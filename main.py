import os
import inspect
from typing import Dict, Any, List, Tuple
import numpy as np
from tinygrad import Tensor, nn, dtypes

from utils.data_generation import generate_dataset
from encoders.base_encoder import BaseEncoder
from database.experiment_db import ExperimentDB

from utils.boilerplate import load_encoders
from utils.losses import get_loss_function

# external libraries
from equidistantpoints import EquidistantPoints

def train_encoder(encoder: BaseEncoder, hyperparameters: Dict[str, Any]) -> Tuple[float, List[float], List[float]]:
    """Train an encoder with given hyperparameters and return losses"""
    # Generate training and validation data
    train_data, val_data, test_data = generate_dataset(num_samples=hyperparameters.get('num_samples', 1000))

    # Get all parameters from the model
    parameters = nn.state.get_parameters(encoder)
    print(parameters)
    print(f"Number of parameters: {len(parameters)}")
    
    # Create optimizer with all parameters
    initial_lr = hyperparameters.get('learning_rate', 0.01)  # Start with a higher learning rate
    optimizer = nn.optim.SGD(parameters, lr=initial_lr)
    
    # Training loop
    batch_size = hyperparameters.get('batch_size', 16)
    epochs = hyperparameters.get('epochs', 20)
    
    train_losses = []
    val_losses = []
    
    # Learning rate decay parameters
    decay_rate = hyperparameters.get('decay_rate', 0.85)  # How much to decay per epoch
    min_lr = hyperparameters.get('min_lr', 0.00000001)  # Minimum learning rate
    
    # Get loss function from hyperparameters
    loss_fn_name = hyperparameters.get('loss_fn', 'mse')
    loss_fn = get_loss_function(loss_fn_name)
    print(f"Using loss function: {loss_fn_name}")
    
    for epoch in range(epochs):
        # Update learning rate with decay
        current_lr = max(initial_lr * (decay_rate ** epoch), min_lr)
        optimizer.lr = current_lr
        
        # Set training mode
        Tensor.training = True
        indices = np.random.permutation(len(train_data))
        epoch_loss = 0
        
        for start_idx in range(0, len(train_data), batch_size):
            batch_indices = indices[start_idx:start_idx+batch_size]
            X_batch = Tensor(train_data[batch_indices], dtype=dtypes.float32)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = encoder(X_batch)
            
            # Calculate loss using the provided loss function
            loss = loss_fn(outputs, X_batch)
            
            # Backward pass
            loss.backward()
            
            # Step optimizer
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.numpy().item()
        
        avg_train_loss = epoch_loss * batch_size / len(train_data)
        train_losses.append(avg_train_loss)
        
        # Validation
        Tensor.training = False
        val_outputs = encoder(Tensor(val_data, dtype=dtypes.float32))
        val_loss = loss_fn(val_outputs, Tensor(val_data, dtype=dtypes.float32)).numpy().item()
        val_losses.append(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {current_lr:.8f}')
    
    # Final evaluation
    Tensor.training = False
    test_outputs = encoder(Tensor(test_data, dtype=dtypes.float32))
    final_test_loss = loss_fn(test_outputs, Tensor(test_data, dtype=dtypes.float32)).numpy().item()
    print(f"Final Test Loss: {final_test_loss:.4f}")
    
    return final_test_loss, train_losses, val_losses

def main():
    # Initialize database
    db = ExperimentDB()
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Load all available encoders
    encoders = load_encoders()
    print(f"Found encoders: {list(encoders.keys())}")
    
    # Define hyperparameter configurations to test
    configurations = [
        {
            'input_size': 7,
            'hidden_size': 1024,
            'output_size': 7,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 500,
            'num_samples': 100000,
            'loss_fn': 'mse',  # Use string identifier instead of function
            'decay_rate': 0.85,
            'min_lr': 0.00000001
        },
        {
            'input_size': 7,
            'hidden_size': 1024,
            'output_size': 7,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 500,
            'num_samples': 100000,
            'loss_fn': 'mae',  # Use string identifier instead of function
            'decay_rate': 0.85,
            'min_lr': 0.00000001
        },
        {
            'input_size': 7,
            'hidden_size': 1024,
            'output_size': 7,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 500,
            'num_samples': 100000,
            'loss_fn': 'huber',  # Use string identifier instead of function
            'decay_rate': 0.85,
            'min_lr': 0.00000001
        },
        {
            'input_size': 7,
            'hidden_size': 1024,
            'output_size': 7,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 500,
            'num_samples': 100000,
            'loss_fn': 'angular_loss',  # Use string identifier instead of function
            'decay_rate': 0.85,
            'min_lr': 0.00000001
        }
    ]
    
    # Run experiments for each encoder and configuration
    for encoder_name, encoder_class in encoders.items():
        print(f"\nRunning experiments for {encoder_name}")
        
        for config in configurations:
            # Check if experiment already exists
            if db.experiment_exists(encoder_name, config):
                print(f"Experiment already exists for {encoder_name} with config: {config}")
                continue
            
            print(f"Training {encoder_name} with config: {config}")
            encoder = encoder_class(**{k: v for k, v in config.items() 
                                    if k in inspect.signature(encoder_class.__init__).parameters})
            
            # Train the encoder and get results
            test_loss, train_losses, val_losses = train_encoder(encoder, config)
            
            # Save results
            experiment_id = db.save_experiment(
                encoder_name, 
                config, 
                train_loss=train_losses[-1],
                val_loss=val_losses[-1] if val_losses else None,
                test_loss=test_loss
            )
            print(f"Experiment {experiment_id} completed with test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main() 