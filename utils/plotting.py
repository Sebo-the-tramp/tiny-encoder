import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tinygrad.tensor import Tensor

def plot_2d_manifold(encoder_outputs: np.ndarray, 
                    true_values: np.ndarray,
                    title: str = "2D Manifold Visualization",
                    save_path: str = None):
    """
    Plot the 2D manifold of encoder outputs vs true values.
    
    Args:
        encoder_outputs: Array of shape (N, D) containing encoder outputs
        true_values: Array of shape (N, D) containing true values
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(12, 6))
    
    # Plot encoder outputs
    plt.subplot(121)
    plt.scatter(encoder_outputs[:, 0], encoder_outputs[:, 1], c='b', alpha=0.5, label='Encoder')
    plt.title(f"{title} - Encoder Outputs")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.legend()
    
    # Plot true values
    plt.subplot(122)
    plt.scatter(true_values[:, 0], true_values[:, 1], c='r', alpha=0.5, label='True')
    plt.title(f"{title} - True Values")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_progress(losses: List[float], 
                         val_losses: List[float] = None,
                         title: str = "Training Progress",
                         save_path: str = None):
    """
    Plot training and validation losses over epochs.
    
    Args:
        losses: List of training losses
        val_losses: Optional list of validation losses
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)
    
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_latent_space(encoder_outputs: np.ndarray,
                          labels: np.ndarray = None,
                          title: str = "Latent Space Visualization",
                          save_path: str = None):
    """
    Visualize the latent space of the encoder using PCA if dimensions > 2.
    
    Args:
        encoder_outputs: Array of shape (N, D) containing encoder outputs
        labels: Optional array of shape (N,) containing labels for coloring
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    from sklearn.decomposition import PCA
    
    # If dimensions > 2, use PCA
    if encoder_outputs.shape[1] > 2:
        pca = PCA(n_components=2)
        encoder_outputs = pca.fit_transform(encoder_outputs)
    
    plt.figure(figsize=(8, 8))
    if labels is not None:
        scatter = plt.scatter(encoder_outputs[:, 0], encoder_outputs[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(encoder_outputs[:, 0], encoder_outputs[:, 1], alpha=0.6)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_reconstruction_error(original: np.ndarray,
                            reconstructed: np.ndarray,
                            title: str = "Reconstruction Error",
                            save_path: str = None):
    """
    Plot the reconstruction error distribution.
    
    Args:
        original: Original input data
        reconstructed: Reconstructed data from the encoder
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    errors = np.mean((original - reconstructed) ** 2, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(np.mean(errors), color='r', linestyle='dashed', 
                label=f'Mean Error: {np.mean(errors):.4f}')
    
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 