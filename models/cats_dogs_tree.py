import torch
import torch.nn as nn


class CatsDogsTree(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.node = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.node(x).squeeze(1)