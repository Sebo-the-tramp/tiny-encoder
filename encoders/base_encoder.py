from abc import ABC, abstractmethod
import numpy as np
from tinygrad.tensor import Tensor

class BaseEncoder(ABC):
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.training = True
    
    def train(self):
        """Set the encoder to training mode"""
        self.training = True
    
    def eval(self):
        """Set the encoder to evaluation mode"""
        self.training = False
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass