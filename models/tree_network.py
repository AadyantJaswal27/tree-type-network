import torch
import torch.nn as nn


class TreeNode(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TreeNetwork(nn.Module):
    def __init__(
        self,
        num_classes=10,
        channels=(32, 64, 64),
        feature_dim=128,
        node_hidden=64,
        depth=4
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depth = depth
        self.num_nodes = 2 ** depth - 1

        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, feature_dim),
            nn.ReLU()
        )

        self.nodes = nn.ModuleList([
            TreeNode(feature_dim, node_hidden)
            for _ in range(self.num_nodes)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)

        node_outputs = torch.stack(
            [node(x) for node in self.nodes],
            dim=1
        )

        batch_size = x.size(0)
        class_scores = []

        for class_idx in range(self.num_classes):
            leaf = class_idx
            score = torch.zeros(
                batch_size,
                device=x.device
            )

            node_idx = 0

            for level in range(self.depth):
                bit = (leaf >> (self.depth - 1 - level)) & 1

                if bit == 0:
                    score = score - node_outputs[:, node_idx]
                    node_idx = 2 * node_idx + 1
                else:
                    score = score + node_outputs[:, node_idx]
                    node_idx = 2 * node_idx + 2

            class_scores.append(score)

        return torch.stack(class_scores, dim=1)