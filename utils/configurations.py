"""
This module contains experiment configurations for different encoders and training settings.
"""

# Base configuration template
BASE_CONFIG = {
    'input_size': 7,
    'hidden_size': 1024,
    'output_size': 7,
    'dropout_rate': 0.2,
    'learning_rate': 0.01,
    'batch_size': 128,
    'epochs': 500,
    'num_samples': 100000,
    'decay_rate': 0.85,
    'min_lr': 0.00000001
}

# Loss function configurations
LOSS_CONFIGS = {
    'mse': {**BASE_CONFIG, 'loss_fn': 'mse'},
    'mae': {**BASE_CONFIG, 'loss_fn': 'mae'},
    'huber': {**BASE_CONFIG, 'loss_fn': 'huber'},
    'angular_loss': {**BASE_CONFIG, 'loss_fn': 'angular_loss'}
}

# Data type configurations
DATA_CONFIGS = {
    'random': {**BASE_CONFIG, 'data_type': 'random'},
    'equally_spaced': {**BASE_CONFIG, 'data_type': 'equally_spaced'}
}

# Combined configurations for different experiments
EXPERIMENT_CONFIGS = [
    # MSE loss with random data
    {**LOSS_CONFIGS['mse'], 'data_type': 'random'},
    
    # MAE loss with random data
    {**LOSS_CONFIGS['mae'], 'data_type': 'random'},
    
    # Huber loss with random data
    {**LOSS_CONFIGS['huber'], 'data_type': 'random'},
    
    # Angular loss with random data
    {**LOSS_CONFIGS['angular_loss'], 'data_type': 'random'},
    
    # MSE loss with equally spaced data
    {**LOSS_CONFIGS['mse'], 'data_type': 'equally_spaced'},
    
    # MAE loss with equally spaced data
    {**LOSS_CONFIGS['mae'], 'data_type': 'equally_spaced'},
    
    # Huber loss with equally spaced data
    {**LOSS_CONFIGS['huber'], 'data_type': 'equally_spaced'},
    
    # Angular loss with equally spaced data
    {**LOSS_CONFIGS['angular_loss'], 'data_type': 'equally_spaced'}
]

def get_configurations():
    """Return all experiment configurations"""
    return EXPERIMENT_CONFIGS

def get_loss_configurations():
    """Return configurations for different loss functions"""
    return LOSS_CONFIGS

def get_data_configurations():
    """Return configurations for different data types"""
    return DATA_CONFIGS 