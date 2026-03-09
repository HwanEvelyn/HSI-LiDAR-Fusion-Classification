from __future__ import annotations

import torch
from torch import nn


class HsiCnnEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = embed_dim // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected HSI input with shape (B, C, H, W), got {tuple(x.shape)}")
        return self.encoder(x)


class LidarCnnEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128) -> None:
        super().__init__()
        hidden_dim = embed_dim // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected LiDAR input with shape (B, 1, H, W), got {tuple(x.shape)}")
        return self.encoder(x)


class Tokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, add_cls_token: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.add_cls_token = add_cls_token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected feature map with shape (B, C, H, W), got {tuple(x.shape)}")

        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        if self.add_cls_token:
            cls_token = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        return self.norm(tokens)
