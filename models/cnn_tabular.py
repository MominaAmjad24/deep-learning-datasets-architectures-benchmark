import math
import torch
import torch.nn as nn

class TabularCNN(nn.Module):
    """
    Treats tabular features as a 1-channel "image" by padding to a square.
    This is intentionally a baseline to show CNN inductive bias mismatch on tabular data.
    """
    def __init__(self, input_dim, num_classes=2, dropout=0.3):
        super().__init__()
        side = math.ceil(math.sqrt(input_dim))
        self.side = side
        self.padded_dim = side * side

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # compute flattened size after 2 pools
        out_side = side
        out_side = out_side // 2
        out_side = out_side // 2
        out_side = max(out_side, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * out_side * out_side, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, input_dim)
        b = x.size(0)
        if x.size(1) < self.padded_dim:
            pad = self.padded_dim - x.size(1)
            x = torch.cat([x, x.new_zeros((b, pad))], dim=1)

        x = x.view(b, 1, self.side, self.side)
        x = self.features(x)
        x = self.classifier(x)
        return x

