import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Net(nn.Module):
    """
    Convolutional Neural Network for EEG classification.
    
    This model uses a two-step approach:
    1. Spatial filtering across EEG channels
    2. Temporal filtering across time samples
    
    Followed by fully connected layers for classification.
    
    Architecture:
    - Conv1: Spatial filter across EEG channels
    - Conv2: Temporal filter across time samples
    - FC1: First fully connected layer with 128 units
    - FC2: Output layer with num_classes units
    """
    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        """
        Initialize the network.
        
        Args:
            num_classes: Number of output classes (default: 2)
            input_channels: Number of EEG channels (default: 1)
        """
        super(Net, self).__init__()
        
        # Spatial filter across channels (input_channels x 1 kernel)
        # Input: [batch, 1, channels, time_points]
        # Output: [batch, 16, 1, time_points]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(input_channels, 1))
        
        # Temporal filter across time (1 x 128 kernel)
        # Input: [batch, 16, 1, time_points]
        # Output: [batch, 32, 1, time_points-127]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 128))
        
        # Fully connected layers
        # 3936 is calculated based on the output dimensions after conv2
        self.fc1 = nn.Linear(3936, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor with shape [batch, 1, channels, time_points]
            
        Returns:
            Output tensor with shape [batch, num_classes]
        """
        # Apply spatial filtering (across channels)
        x = F.relu(self.conv1(x))
        
        # Apply temporal filtering (across time)
        x = F.relu(self.conv2(x))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax for classification probabilities
        x = F.softmax(x, dim=1)
        
        return x
