"""
Predefined WandB sweep configurations for BindGPS project.
Each config includes a project name for organized logging.
"""
#-------------------GCN SWEEPS-------------------
#1A variables - using the v3 agent trainer
gcn_model_no_svm_parameter_first_sweep = { 
    'method': 'grid',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    # 'project': 'gps-gcn-model-no-svm-parameter-test1',
    'project': 'gps-gcn-model-no-svm-parameter-test1_v3',
    'parameters': {
        'model_type': {'values': ['gcn']},
         #parameters to tune
        'hidden_gnn_size': {'values': [128, 256]},        
        'hidden_linear_size': {'values': [128, 256]}, 
        'num_linear_layers': {'values': [2, 3, 4]},  
        'normalize': {'values': [True, False]},

        #constant variables
        'non_mre_size': {'value': 0},
        'seed': {'value': 42},
        'dropout': {'value': 0.3},
        'optimizer_type': {'values': ["adamw"]},
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'num_gnn_layers': {'value': 3},
        'num_neighbors_to_sample': {'value': 10},
        'batch_size': {'value': 256},
        'epochs': {'value': 40},
    }
}

gcn_model_no_svm_parameter_second_sweep = {
    'method': 'grid',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gcn-model-no-svm-parameter-test2_v3',
    'parameters': {
        'model_type': {'values': ['gcn']},
        #parameters to tune 
        # 'num_gnn_layers': {'values': [2, 3]},
        # 'num_neighbors_to_sample': {'values': [10, 20]},  
        # 'optimizer_type': {'values': ["adam", "adamw", "sgd", "rmsprop"]},

        'num_gnn_layers': {'value': 2},
        'num_neighbors_to_sample': {'value': 10},  
        'optimizer_type': {'value': "adamw"},
        
        #constant variables
        'non_mre_size': {'value': 0},
        'seed': {'value': 42},
        'hidden_gnn_size': {'value': 128}, #best loss
        'hidden_linear_size': {'value': 128}, #best loss
        'num_linear_layers': {'value': 3}, #unclear -- most information
        'normalize': {'value': True},
        'dropout': {'value': 0.3},
        'lr': {'value': 0.001},
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 40},
        'batch_size': {'value': 256}
    }
}

gcn_model_no_svm_parameter_third_sweep = {
    'method': 'bayes',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gcn-model-no-svm-parameter-test3_v3',
    'parameters': {
        'model_type': {'values': ['gcn']},
        #parameters to tune 
        # 'num_gnn_layers': {'values': [2, 3]},
        # 'num_neighbors_to_sample': {'values': [10, 20]},  
        # 'optimizer_type': {'values': ["adam", "adamw", "sgd", "rmsprop"]},

        'num_gnn_layers': {'values': [2, 3]},
        'num_neighbors_to_sample': {'values': [10, 20]},  
        'dropout': {'values': [0.3, 0.4, 0.5]},
        'lr': {'values': [0.0005, 0.001, 0.005]},

        #constant variables
        'non_mre_size': {'value': 0}, #HAS THE BEST PERFORMANCE
        'seed': {'value': 42},
        'hidden_gnn_size': {'value': 128}, #best loss
        'hidden_linear_size': {'value': 128}, #best loss
        'num_linear_layers': {'value': 3}, #unclear -- most information
        'normalize': {'value': True}, #GOOD
        'optimizer_type': {'value': "adamw"}, #GOOD
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 40},
        'batch_size': {'value': 256}
    }
}

gcn_model_no_svm_parameter_fourth_sweep = {
    'method': 'bayes',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gcn-model-no-svm-parameter-test4_v3',
    'parameters': {
        'model_type': {'value': 'gcn'},
        #parameters to tune 
        'dropout': {'values': [0.3, 0.4]}, #these are relatively constant, we just want to see what's best
        'lr': {'values': [0.0005, 0.001]}, #these are relatively constant, we just want to see what's best
        'num_linear_layers': {'values': [2, 3]}, #these are relatively constant, we just want to see what's best
        'non_mre_size': {'values': [0.2, 0.1, 0]},
        'seed': {'values': [42, 13]}, #try more seeds in the next sweep

        #constant variables
        'num_gnn_layers': {'value': 2},
        'num_neighbors_to_sample': {'value': 10},
        'hidden_gnn_size': {'value': 128}, #best loss
        'hidden_linear_size': {'value': 128}, #best loss
        'normalize': {'value': True}, #GOOD
        'optimizer_type': {'value': "adamw"}, #GOOD
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 40},
        'batch_size': {'value': 256}
    }
}

gcn_model_no_svm_parameter_fifth_sweep = {
    'method': 'grid',  
    'metric': {'name': 'final_test_accuracy', 'goal': 'maximize'},
    'project': 'gps-gcn-model-no-svm-parameter-test5_v3',
    'parameters': {
        'model_type': {'value': 'gcn'},
        #parameters to tune 
        'num_linear_layers': {'values': [2, 3]}, #these are relatively constant, we just want to see what's best
        'non_mre_size': {'values': [0.2, 0.3, 0.4]},
        'seed': {'values': [42, 123, 2025, 999, 0]}, #try more seeds in the next sweep

        #constant variables
        'seed': {'value': 42}, #try more seeds in the next sweep
        'dropout': {'value': 0.3}, #these are relatively constant, we just want to see what's best
        'lr': {'value': 0.0005}, #these are relatively constant, we just want to see what's best
        'num_gnn_layers': {'value': 2},
        'num_neighbors_to_sample': {'value': 10},
        'hidden_gnn_size': {'value': 128}, #best loss
        'hidden_linear_size': {'value': 128}, #best loss
        'normalize': {'value': True}, #GOOD
        'optimizer_type': {'value': "adamw"}, #GOOD
        'weight_decay': {'value': 0.0005},
        'epochs': {'value': 40},
        'batch_size': {'value': 256}
    }
}


SWEEP_CONFIGS = {
    #GCN Parameter Sweeps - SARAH
    'gcn_model_no_svm_param_sweep1_v3': gcn_model_no_svm_parameter_first_sweep, 
    'gcn_model_no_svm_param_sweep2_v3': gcn_model_no_svm_parameter_second_sweep,
    'gcn_model_no_svm_param_sweep3_v3': gcn_model_no_svm_parameter_third_sweep, 
    'gcn_model_no_svm_param_sweep4_v3': gcn_model_no_svm_parameter_fourth_sweep, 
    'gcn_model_no_svm_param_sweep5_v3': gcn_model_no_svm_parameter_fifth_sweep
}

def get_sweep_config(name):
    """Get a sweep configuration by name"""
    if name not in SWEEP_CONFIGS:
        available = list(SWEEP_CONFIGS.keys())
        raise ValueError(f"Unknown sweep config '{name}'. Available: {available}")
    return SWEEP_CONFIGS[name]
