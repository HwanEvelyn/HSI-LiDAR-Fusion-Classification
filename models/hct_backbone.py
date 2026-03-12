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
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_spatial_tokens: int,
        add_cls_token: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.add_cls_token = add_cls_token
        token_count = num_spatial_tokens + int(add_cls_token)
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected feature map with shape (B, C, H, W), got {tuple(x.shape)}")

        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        if self.add_cls_token:
            cls_token = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        if tokens.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"Tokenizer 期望 token 数量为 {self.pos_embed.size(1)}，实际得到 {tokens.size(1)}。"
            )

        tokens = tokens + self.pos_embed
        tokens = self.norm(tokens)
        return self.dropout(tokens)
