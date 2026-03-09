from __future__ import annotations

import torch
from torch import nn

from .fusion_blocks import ConcatFusionHead
from .hct_backbone import HsiCnnEncoder, LidarCnnEncoder, Tokenizer


class HCT_BGC(nn.Module):
    def __init__(
        self,
        hsi_in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hsi_encoder = HsiCnnEncoder(hsi_in_channels, embed_dim=embed_dim)
        self.lidar_encoder = LidarCnnEncoder(embed_dim=embed_dim)

        self.hsi_tokenizer = Tokenizer(self.hsi_encoder.out_channels, embed_dim=embed_dim)
        self.lidar_tokenizer = Tokenizer(self.lidar_encoder.out_channels, embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.h_te = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.l_te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.fusion_head = ConcatFusionHead(embed_dim, embed_dim, hidden_dim=mlp_dim, num_classes=num_classes)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        h_feat_map = self.hsi_encoder(hsi)
        l_feat_map = self.lidar_encoder(lidar)

        h_tokens = self.hsi_tokenizer(h_feat_map)
        l_tokens = self.lidar_tokenizer(l_feat_map)

        h_tokens = self.h_te(h_tokens)
        l_tokens = self.l_te(l_tokens)

        h_cls = h_tokens[:, 0]
        l_cls = l_tokens[:, 0]
        return self.fusion_head(h_cls, l_cls)
