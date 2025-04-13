# Custom Encoder Experiments

This project provides a framework for experimenting with different neural network encoder architectures. It includes a database to track experiment results and avoid redundant training.

## Project Structure

```
.
├── README.md
├── main.py                 # Main training script
├── encoders/              # Directory containing encoder implementations
│   ├── __init__.py
│   ├── base_encoder.py    # Base class for all encoders
│   └── mlp_encoder.py     # MLP encoder implementation
└── database/             # Database handling
    └── experiment_db.py   # SQLite database manager
```

## Adding New Encoders

To add a new encoder:

1. Create a new file in the `encoders` directory (e.g., `encoders/my_encoder.py`)
2. Implement your encoder class inheriting from `BaseEncoder`
3. Implement the required methods:
   - `forward(self, x: Tensor) -> Tensor`
   - `parameters(self)`

Example:
```python
from .base_encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__(input_size, output_size)
        # Initialize your encoder
    
    def forward(self, x):
        # Implement forward pass
        return x
    
    def parameters(self):
        # Return trainable parameters
        return []
```

## Running Experiments

To run experiments:

1. Make sure your encoder is implemented in the `encoders` directory
2. Configure hyperparameters in `main.py` if needed
3. Run:
```bash
python main.py
```

The script will:
- Automatically discover all encoder implementations
- Check if experiments already exist in the database
- Train only new configurations
- Save results to the SQLite database

## Database Schema

The experiments database (`experiments.db`) contains:
- Encoder name
- Hyperparameters
- Training loss
- Validation loss
- Timestamp

## Requirements

- tinygrad
- numpy
- sqlite3 (included in Python) 