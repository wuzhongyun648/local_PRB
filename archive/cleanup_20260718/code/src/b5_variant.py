"""Model components for the B5 (A1+A3+A6+A8) experiment variant."""

import torch.nn as nn


class PaperExplorationNetwork(nn.Module):
    def __init__(self, input_dim, _kernel_size=100, _stride=50, _channels=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.fc2(self.activate(self.fc1(x)))
