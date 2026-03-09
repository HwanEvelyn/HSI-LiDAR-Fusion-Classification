from __future__ import annotations

import torch
from torch import nn


class ConcatFusionHead(nn.Module):
    def __init__(self, in_dim_a: int, in_dim_b: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim_a + in_dim_b, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([feat_a, feat_b], dim=1)
        return self.classifier(fused)
