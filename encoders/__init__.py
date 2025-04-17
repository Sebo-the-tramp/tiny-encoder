"""
Encoder package containing different implementations of neural network encoders.
Each encoder should inherit from BaseEncoder and implement the required methods.
"""

from .base_encoder import BaseEncoder

from .mlp_encoder_256 import MLPEncoder256
from .mlp_encoder_512 import MLPEncoder512
from .mlp_encoder_1024 import MLPEncoder1024


__all__ = ['BaseEncoder', 'MLPEncoder'] 