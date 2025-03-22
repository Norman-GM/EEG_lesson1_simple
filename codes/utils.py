import os
import random
import numpy as np
import torch
from typing import Optional

def seed_everything(seed: int = 2025) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets the random seed for Python's random module,
    NumPy, PyTorch, and CUDA if available, ensuring reproducible results.
    
    Args:
        seed: Integer seed value (default: 2025)
    """
    # Set Python's random module seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA's random seed if available
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    
    # Disable CUDA benchmarking for reproducibility
    torch.backends.cudnn.benchmark = False
