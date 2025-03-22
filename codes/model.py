import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(input_channels, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 128))
        self.fc1 = nn.Linear(3936, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x