import os
import importlib
import inspect

from typing import Dict, Type
from encoders.base_encoder import BaseEncoder

def load_encoders() -> Dict[str, Type[BaseEncoder]]:
    """Dynamically load all encoder classes from the encoders directory"""
    encoders = {}
    encoders_dir = "encoders"
    
    # Skip __pycache__ and base_encoder
    for file in os.listdir(encoders_dir):
        if file.endswith(".py") and file != "base_encoder.py" and not file.startswith("__"):
            module_name = file[:-3]  # Remove .py extension
            module = importlib.import_module(f"encoders.{module_name}")
            
            # Find all classes that inherit from BaseEncoder
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, BaseEncoder) 
                    and obj != BaseEncoder):
                    encoders[name] = obj
    
    return encoders