from tinygrad import Tensor

def mse_loss(x, y):
    """Mean Squared Error loss"""
    return ((x - y)**2).mean()

def mae_loss(x, y):
    """Mean Absolute Error loss"""
    return (x - y).abs().mean()

def huber_loss(x, y, delta=0.1):
    """Huber loss - combines MSE and MAE for robustness to outliers"""
    diff = x - y
    abs_diff = diff.abs()
    quadratic = (diff**2).mean()
    linear = (abs_diff - 0.5 * delta).mean()
    return (abs_diff <= delta).where(quadratic, linear)

def angular_loss(x, y):
    """Angular loss - measures the angle between vectors"""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    cos_theta = (x * y).sum(axis=1) / (x.norm(dim=1) * y.norm(dim=1))
    return 1 - cos_theta.mean()

# Dictionary mapping loss function names to their implementations
LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'mae': mae_loss,
    'huber': huber_loss,
    'angular_loss': angular_loss
}

def get_loss_function(loss_name):
    """Get a loss function by name"""
    return LOSS_FUNCTIONS.get(loss_name, mse_loss)  # Default to MSE if not found 