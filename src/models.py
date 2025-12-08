import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import SAGEConv, GATConv
import torch_geometric.nn as geom_nn
# from transformers import AutoModel, AutoTokenizer

class GCN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_gnn_size,
            num_gnn_layers,
            hidden_linear_size,
            num_linear_layers,
            out_channels,
            dropout=0.5,
            normalize=True
            ):
        super().__init__()
        
        assert num_gnn_layers >= 1, "num_gnn_layers must be >= 1"
        assert num_linear_layers >= 1, "num_linear_layers counts the output layer; must be >= 1"

        # convolutional layers
        self.convs  = nn.ModuleList(
            [SAGEConv(in_channels, hidden_gnn_size,normalize=normalize)] +
            [SAGEConv(hidden_gnn_size, hidden_gnn_size,normalize=normalize) for _ in range(max(0, num_gnn_layers - 1))]
        )

        # linear layers
        lin_sizes = [hidden_gnn_size] + [hidden_linear_size] * max(0, num_linear_layers - 1) + [out_channels]
        self.linear = nn.ModuleList([
            nn.Linear(lin_sizes[i], lin_sizes[i+1]) for i in range(len(lin_sizes) - 1)
        ])

        self.dropout = dropout

    def forward(self,x,edge_index):
        # No trim

        # convolutional layers
        for conv in self.convs:
            x = conv(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout,training=self.training)

        # linear layers
        for i,lin in enumerate(self.linear):
            x = lin(x)
            if i < (len(self.linear) - 1):
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout,training=self.training)
        
        return x