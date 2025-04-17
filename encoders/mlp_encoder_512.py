from tinygrad import Tensor, nn
from .base_encoder import BaseEncoder
from extra.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class MLPEncoder512(BaseEncoder):

    """Various optimization and other things for training"""
    T_max = 1000
    eta_min = 10e-20

    def __init__(self, input_size=3, hidden_size=512, output_size=3, dropout_rate=0.2):
        super().__init__(input_size, output_size)
        # Ensure parameters are integers
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.dropout_rate = float(dropout_rate)        
        
        # Initialize layers with proper parameter types
        self.layer1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size, bias=True)

        self.opt = nn.optim.Adam(nn.state.get_parameters(self), lr=0.01)
        # self.lr_sched = ReduceLROnPlateau(self.opt, mode="min", factor=0.1, patience=1, threshold=1e-2, threshold_mode="rel")
        # self.lr_sched = CosineAnnealingLR(self.opt, T_max=self.T_max, eta_min=self.eta_min)
    
    def __call__(self, x: Tensor) -> Tensor:            
        # Forward pass with proper tensor operations
        x = self.layer1(x)
        x = x.relu()
        x = self.layer2(x)
        x = x.tanh()  # Constrain output to [-1, 1]
        return x
    
    def loss_fn(self, out: Tensor, Y: Tensor) -> Tensor:        
        return ((out - Y) ** 2).mean()