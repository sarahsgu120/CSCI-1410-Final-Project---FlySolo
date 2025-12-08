# train_gnn.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Torch Eval Metrics
from torcheval.metrics import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassAUROC, MulticlassAUPRC
)

# Project modules
from src.data import DatasetLoader, GraphParamBuilder
from src.models import GCN

# PyG
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Paths
    # base_path: str = "/home/jchc/Documents/larschan_laboratory/BindGPS/data/datasets"
    base_path: str = "/oscar/data/larschan/shared_data/BindGPS/data/datasets"

    # Data
    p_value: str = "0_1"
    resolution: str = "1kb"
    exclude_features: Tuple[str, ...] = ("clamp", "gaf", "psq")
    features_of_interest: Tuple[str, ...] = (
        "clamp","gaf","psq","h3k27ac","h3k27me3","h3k36me3",
        "h3k4me1","h3k4me2","h3k4me3","h3k9me3","h4k16ac"
    )
    target_column: str = "mre_labels"
    train_size: float = 0.7
    non_mre_size: float = 0.3 #negative samples
    seed: int = 42

    # Model
    model_type: str = "gcn"  # "gcn" or "gat"
    hidden_gnn_size: int = 128
    num_gnn_layers: int = 3
    hidden_linear_size: int = 128
    num_linear_layers: int = 3
    dropout: float = 0.5
    normalize: bool = True
    
    # GAT-specific parameters
    gat_heads: int = 4
    gat_negative_slope: float = 0.2
    gat_concat: bool = True
    gat_edge_dim: int = 2  # contactCount + loop_size_transformed

    # Optimization
    optimizer_type: str = 'adamw' # adam, adamw, sgd, or rmsprop
    lr: float = 5e-4
    weight_decay: float = 5e-4
    use_class_weights: bool = True
    epochs: int = 5

    # NeighborLoader
    num_neighbors_to_sample: int = 20 # TUNABLE
    num_neighbors: List[int] = field(init=False) # don't change; computed automatically (__post_init__)
    batch_size: int = 256
    num_workers: int = 0
    
    # Batch validation (processes all edges)
    val_batch_size: Optional[int] = 1024  # If None, uses full graph evaluation

    # Device / perf
    use_cuda_if_available: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "basic-gnn"
    wandb_entity: Optional[str] = None  # or your entity string

    def __post_init__(self):
        self.num_neighbors = [self.num_neighbors_to_sample] * self.num_gnn_layers

# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For more determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(use_cuda_if_available: bool = True) -> torch.device:
    if use_cuda_if_available and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------------------
# Trainer
# ----------------------------
class GNNTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_global_seed(cfg.seed)
        self.device = get_device(cfg.use_cuda_if_available)

        # Will be populated later
        self.node_df: Optional[pd.DataFrame] = None
        self.edge_df: Optional[pd.DataFrame] = None
        self.data: Optional[Data] = None
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[torch.nn.Module] = None
        
        # W&B
        self.wandb_run = None
        if self.cfg.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

    # ----- Logging -----
    def _init_wandb(self) -> None:
        try:
            wandb.login()
            self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=asdict(self.cfg),
            )
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")
            self.wandb_run = None

    def _log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    # ----- Data -----
    def build_dataset(self) -> None:
        # Load raw dataframes
        loader = DatasetLoader(base_path=self.cfg.base_path)
        node_df, edge_df = loader.load(p_value=self.cfg.p_value, resolution=self.cfg.resolution)

        self.node_df = node_df.copy()
        self.edge_df = edge_df.copy()

        # Feature selection
        feats = [f for f in self.cfg.features_of_interest if f not in self.cfg.exclude_features]
        input_features = self.node_df.loc[:, feats].copy()

        # Target and mask (mre > 0 considered labeled)
        target = self.node_df[self.cfg.target_column].copy()
        self.node_df["mre_mask"] = self.node_df[self.cfg.target_column].apply(lambda x: True if x > 0 else False)
        mre_mask = self.node_df["mre_mask"].astype(bool)

        total_mre_samples = mre_mask.sum()
        self.node_df["non_mre_mask"] = self.node_df[self.cfg.target_column].apply(lambda x: True if x == 0 else False)
        non_mre_mask = self.node_df["non_mre_mask"].astype(bool)

        # Randomly select non-MRE samples based on defined size. #this inflates the level of negative samples
        non_mre_samples = total_mre_samples * self.cfg.non_mre_size
        indices = np.arange(len(non_mre_mask))
        non_mre_indices = indices[non_mre_mask]
        selected_non_mres = np.random.choice(non_mre_indices, size=int(non_mre_samples), replace=False)
        non_mre_mask = np.zeros(len(non_mre_mask), dtype=bool)
        non_mre_mask[selected_non_mres] = True
        self.node_df["non_mre_mask"] = non_mre_mask

        # Combine MRE and non-MRE Samples
        mask = mre_mask | non_mre_mask

        # Stratified split on masked subset
        X_train, X_evaluation, y_train, y_evaluation = train_test_split(
            input_features[mask],
            target[mask],
            train_size=self.cfg.train_size,
            stratify=target[mask],
            random_state=self.cfg.seed,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_evaluation,
            y_evaluation,
            train_size=0.3,
            stratify=y_evaluation,
            random_state=self.cfg.seed,
        )

        # Build boolean masks over the FULL index
        train_mask = pd.Series(False, index=input_features.index)
        val_mask = pd.Series(False, index=input_features.index)
        test_mask = pd.Series(False, index=input_features.index)
        train_mask.loc[X_train.index] = True
        val_mask.loc[X_val.index] = True
        test_mask.loc[X_test.index] = True

        # Build tensors via your helper
        builder = GraphParamBuilder(
            node_df=self.node_df,
            edge_df=self.edge_df,
            target=target,
            mask=mask,
            input_features=input_features,
            seed=self.cfg.seed,
        )
        tensors = builder.convert_to_tensors()

        # Ensure dtypes
        X = tensors["X"]                          # [N, F] float
        y = tensors["y"].to(torch.long)           # [N] long for CE loss
        edge_index = tensors["edge_index"]        # [2, E]
        
        # Edge weights for NeighborSampler (use p-value transformed)
        edge_weight = tensors["edge_pvalue_transformed"]  # [E]
        
        # Edge features (contact count + loop size transformed)
        edge_attr = torch.stack([
            tensors["edge_contactCount"],
            tensors["edge_loop_size_transformed"]
        ], dim=1)  # [E, 2] - 2 edge features

        pyg_data = Data(
            x=X,
            y=y,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            train_mask=torch.tensor(train_mask.to_numpy(), dtype=torch.bool),
            val_mask=torch.tensor(val_mask.to_numpy(), dtype=torch.bool),
            test_mask=torch.tensor(test_mask.to_numpy(), dtype=torch.bool),
        )
        self.data = pyg_data

    def build_loaders(self) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
        assert self.data is not None, "Call build_dataset() first."
        
        # Training loader with neighbor sampling
        train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=list(self.cfg.num_neighbors),
            batch_size=self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        # Validation loader with ALL neighbors (-1 means no sampling limit)
        val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * len(self.cfg.num_neighbors),  # Sample ALL neighbors
            batch_size=self.cfg.val_batch_size if self.cfg.val_batch_size else self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        # Test loader with ALL neighbors
        test_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1] * len(self.cfg.num_neighbors),  # Sample ALL neighbors
            batch_size=self.cfg.val_batch_size if self.cfg.val_batch_size else self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        return train_loader, val_loader, test_loader

    def _get_class_weights(self) -> Optional[torch.Tensor]:
            """Calculates inverse class weights based on training data."""
            if not self.cfg.use_class_weights:
                return None

            # Get only training labels
            y_train = self.data.y[self.data.train_mask].long()
            
            # Count samples per class
            # minlength ensures we get a count even if a class is missing in train set
            counts = torch.bincount(y_train, minlength=self.cfg.num_classes).float()
            
            # Prevent division by zero
            counts[counts == 0] = 1 
            
            # Inverse frequency: Total / (num_classes * class_count)
            # This ensures the expected value of weights is 1.0
            total_samples = len(y_train)
            weights = total_samples / (self.cfg.num_classes * counts)
            
            print(f"Class Counts: {counts.tolist()}")
            print(f"Calculated Class Weights: {weights.tolist()}")
            
            return weights.to(self.device)

    def _get_optimizer(self):
        """Initialize the optimizer based on config."""
        
        if self.cfg.optimizer_type.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            
        elif self.cfg.optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            
        elif self.cfg.optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            
        elif self.cfg.optimizer_type.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer_type}")

    # ----- Model / Optim -----
    def build_model(self) -> None:
        assert self.data is not None, "Call build_dataset() first."

        # Safer class count (works even if labels aren't 0..C-1)
        train_labels = self.data.y[self.data.train_mask]
        unique_labels = torch.unique(train_labels)
        num_classes = int(unique_labels.numel())
        self.cfg.num_classes = num_classes # updating config file

        print(f"Unique train labels: {unique_labels}")
        print(f"Training with num_classes: {num_classes}")
        print(f"Train Samples: {self.data.train_mask.sum()}")
        print(f"Val Samples: {self.data.val_mask.sum()}")
        print(f"Test Samples: {self.data.test_mask.sum()}")
        
        # Check if labels need remapping to 0..C-1 range, will happen when we exclude non-mres
        if unique_labels.min() != 0 or unique_labels.max() != (num_classes - 1):
            print(f"Warning: Labels not in 0..{num_classes-1} range, remapping...")
            # Create mapping from original labels to 0..C-1
            label_mapping = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
            print(f"Label mapping: {label_mapping}")
            
            # Remap all labels in the dataset
            for old_label, new_label in label_mapping.items():
                self.data.y[self.data.y == old_label] = new_label


        # Model
        if self.cfg.model_type.lower() == "gcn":
            self.model = GCN(
                in_channels=self.data.x.size(1),
                out_channels=num_classes,
                hidden_gnn_size=self.cfg.hidden_gnn_size,
                num_gnn_layers=self.cfg.num_gnn_layers,
                hidden_linear_size=self.cfg.hidden_linear_size,
                num_linear_layers=self.cfg.num_linear_layers,
                dropout=self.cfg.dropout,
                normalize=self.cfg.normalize,
            ).to(self.device)
        elif self.cfg.model_type.lower() == "gat":
            self.model = GATModel(
                in_channels=self.data.x.size(1),
                out_channels=num_classes,
                hidden_gnn_size=self.cfg.hidden_gnn_size,
                num_gnn_layers=self.cfg.num_gnn_layers,
                hidden_linear_size=self.cfg.hidden_linear_size,
                num_linear_layers=self.cfg.num_linear_layers,
                heads=self.cfg.gat_heads,
                concat=self.cfg.gat_concat,
                negative_slope=self.cfg.gat_negative_slope,
                dropout=self.cfg.dropout,
                edge_dim=self.cfg.gat_edge_dim,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model_type: {self.cfg.model_type}. Use 'gcn' or 'gat'.")

        # Criterion
        class_weights = self._get_class_weights()
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer 
        self.optimizer = self._get_optimizer()

    # ----- Eval Metrics -----
    def _get_torcheval_metrics(self):
        """
        Initializes a dictionary of torcheval metrics and moves them to the GPU.
        """
        
        metrics = {
            'acc_macro': MulticlassAccuracy(num_classes=self.cfg.num_classes, average="macro"),
            'acc_micro': MulticlassAccuracy(num_classes=self.cfg.num_classes, average="micro"),
            'prec':      MulticlassPrecision(num_classes=self.cfg.num_classes, average="macro"),
            'rec':       MulticlassRecall(num_classes=self.cfg.num_classes, average="macro"),
            'f1':        MulticlassF1Score(num_classes=self.cfg.num_classes, average="macro"),
            # Note: AUROC/AUPRC in torcheval handle logits automatically
            'auroc':     MulticlassAUROC(num_classes=self.cfg.num_classes, average="macro"),
            'aupr':      MulticlassAUPRC(num_classes=self.cfg.num_classes, average="macro"),
        }
        
        # Move all metrics to the correct device (GPU)
        for name, metric in metrics.items():
            metrics[name] = metric.to(self.device)
            
        return metrics

    # ----- Train / Eval -----
    def _train_one_epoch(self, train_loader: NeighborLoader, val_loader: NeighborLoader, epoch: int) -> Tuple[float, float, float, float]:
        assert self.model is not None and self.optimizer is not None

        # initialize metrics
        if not hasattr(self, 'train_metrics'):
            self.train_metrics = self._get_torcheval_metrics()
            
        # Reset states at the start of the epoch
        for metric in self.train_metrics.values():
            metric.reset()
      
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # --- Forward Pass ---
            if self.cfg.model_type.lower() == "gat":
                # GAT models can use edge attributes if available
                edge_attr = getattr(batch, 'edge_attr', None)
                out = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)  # [N_batch, C]
            else:
                out = self.model(batch.x, batch.edge_index)  # [N_batch, C]
            
            # --- Prep Target ---
            mask = batch.train_mask.bool()
            targets = batch.y.to(torch.long)

            # --- Loss ---
            loss = self.criterion(out[mask], targets[mask])
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            
            # --- Metrics Update ---
            preds_masked = out[mask].detach()
            targets_masked = targets[mask]

            for metric in self.train_metrics.values():
                metric.update(preds_masked, targets_masked)
            
            # Clear intermediate variables to free GPU memory
            del out, loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Epoch Results ---
        epoch_loss = total_loss / max(len(train_loader), 1)
        
        train_results = {}
        for name, metric in self.train_metrics.items():
            train_results[name] = metric.compute().item()
        
        # --- Wandb Logging ---
        log_payload = {
            "train/loss": epoch_loss,
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_results.items()}
        }
        self._log(log_payload,step=epoch)

        # validation
        val_loss, val_results = self.evaluate(split="val", loader=val_loader)
        
        return epoch_loss, train_results, val_loss, val_results

    @torch.no_grad()
    def evaluate(self, split="test", loader=None) -> Tuple[float, float]:
        """Evaluate model using NeighborLoader with all neighbors.
        
        Args:
            split: "test" or "val" to specify which nodes to evaluate
            loader: NeighborLoader to use for evaluation (required)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        assert self.model is not None and self.data is not None
        assert loader is not None, "NeighborLoader is required for evaluation"
        
        self.model.eval()
        return self._evaluate_with_loader(loader, split)
    
    def _evaluate_with_loader(self, loader: NeighborLoader, split: str) -> Tuple[float, float]:
        """Evaluate using NeighborLoader with all neighbors (-1 sampling)."""

        eval_metrics = self._get_torcheval_metrics()
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # --- Forward Pass ---
            if self.cfg.model_type.lower() == "gat":
                edge_attr = getattr(batch, 'edge_attr', None)
                out = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)
            else:
                out = self.model(batch.x, batch.edge_index)
            
            # --- Picking Proper Mask ---
            if split == "val":
                mask = batch.val_mask.bool()
            elif split == "test":
                mask = batch.test_mask.bool()
            else:
                mask = batch.train_mask.bool() # fallback
            
            # skip if no nodes in batch
            if mask.sum() == 0:
                continue
            
            # --- Loss & Metrics ---
            batch_logits = out[mask]
            batch_targets = batch.y[mask].to(torch.long)
            
            loss = self.criterion(batch_logits, batch_targets)
            total_loss += float(loss.item())

            # --- Update Metrics ---
            preds_masked = batch_logits.detach()
            targets_masked = batch_targets

            for metric in eval_metrics.values():
                metric.update(preds_masked, targets_masked)

            # Memory cleanup
            del out, batch_logits, batch_targets
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Compute Results ---            
        avg_loss = total_loss / max(len(loader),1)

        results= {}
        for name, metric in eval_metrics.items():
            results[name] = metric.compute().item()

        # --- Logging ---
        log_payload = {
            f"{split}/loss": avg_loss,
            **{f"{split}/{k}": v for k, v in results.items()}
        }
        self._log(log_payload)
        
        return avg_loss, results

    def fit(self) -> None:
        self.build_dataset()
        train_loader, val_loader, test_loader = self.build_loaders()
        self.build_model()

        print(f"Device: {self.device}")
        for epoch in range(self.cfg.epochs):
            train_loss, train_results, val_loss, val_results = self._train_one_epoch(train_loader, val_loader, epoch)

            # --- Metrics to Print ---
            train_acc = train_results.get('acc_macro', 0.0)
            train_f1  = train_results.get('f1', 0.0)
            train_auroc = train_results.get('auroc', 0.0)
            train_aupr = train_results.get('aupr', 0.0)

            val_acc = val_results.get('acc_macro', 0.0)
            val_f1  = val_results.get('f1', 0.0)
            val_auroc = val_results.get('auroc', 0.0)
            val_aupr = val_results.get('aupr', 0.0)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} (F1: {train_f1:.4f}) | "
                  f"Train AUROC: {train_auroc:.4f} | "
                  f"Train AURC: {train_aupr:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} (F1: {val_f1:.4f}) | "  
                  f"Val AUROC: {val_auroc:.4f} | "
                  f"Val AURC: {val_aupr:.4f}")
            
        # --- Test Set Evaluation ---
        test_loss, test_results = self.evaluate(split="test", loader=test_loader)        
        test_acc = test_results.get('acc_macro', 0.0)
        test_f1  = test_results.get('f1', 0.0)
        test_auroc = test_results.get('auroc', 0.0)
        test_aupr = test_results.get('aupr', 0.0)

        print(f"Final Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test F1: {test_f1:.4f} | "
              f"Test AUROC: {test_auroc:.4f} | "
              f"Test AUPR: {test_aupr:.4f}")

        print(f"Full Test Metrics: {test_results}")

    # Convenience single entry
    def run(self) -> None:
        self.fit()
        if self.wandb_run is not None:
            self.wandb_run.finish()

# ----------------------------
# Script entry
# ----------------------------
def main():
    cfg = Config(
        # toggle this on to log to W&B (requires `wandb login`)
        use_wandb=False, 
        seed=42,  # Using default seed for reproducibility
        val_batch_size=1024  # Enable batched validation
    )
    trainer = GNNTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()