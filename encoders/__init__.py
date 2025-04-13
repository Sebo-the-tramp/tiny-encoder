"""
Encoder package containing different implementations of neural network encoders.
Each encoder should inherit from BaseEncoder and implement the required methods.
"""

from .base_encoder import BaseEncoder
from .mlp_encoder import MLPEncoder

__all__ = ['BaseEncoder', 'MLPEncoder'] 