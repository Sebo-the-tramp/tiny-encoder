from tinygrad import Tensor, nn
from .base_encoder import BaseEncoder

class MLPEncoder(BaseEncoder):
    def __init__(self, input_size=3, hidden_size=1024, output_size=3, dropout_rate=0.2):
        super().__init__(input_size, output_size)
        # Ensure parameters are integers
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.dropout_rate = float(dropout_rate)
        
        # Initialize layers with proper parameter types
        self.layer1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size, bias=True)
    
    def __call__(self, x: Tensor) -> Tensor:            
        # Forward pass with proper tensor operations
        x = self.layer1(x)
        x = x.relu()
        x = self.layer2(x)
        x = x.tanh()  # Constrain output to [-1, 1]
        return x