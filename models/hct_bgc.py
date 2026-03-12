from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .fusion_blocks import BiDirectionalClassTokenAttention, GatedCrossModalFusion
from .hct_backbone import HsiCnnEncoder, LidarCnnEncoder, Tokenizer


class HCT_BGC(nn.Module):
    def __init__(
        self,
        hsi_in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        fusion_layers: int = 1,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "fusion_layers": fusion_layers,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
        }
        self.hsi_encoder = HsiCnnEncoder(hsi_in_channels, embed_dim=embed_dim)
        self.lidar_encoder = LidarCnnEncoder(embed_dim=embed_dim)

        self.hsi_tokenizer = Tokenizer(self.hsi_encoder.out_channels, embed_dim=embed_dim)
        self.lidar_tokenizer = Tokenizer(self.lidar_encoder.out_channels, embed_dim=embed_dim)

        self.h_te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_layers,
        )
        self.l_te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=num_layers,
        )
        self.fusion_blocks = nn.ModuleList(
            [
                BiDirectionalClassTokenAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(fusion_layers)
            ]
        )
        self.gated_fuse = GatedCrossModalFusion(embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def get_config(self) -> dict[str, int | float]:
        return dict(self.config)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_feat_map = self.hsi_encoder(hsi)
        l_feat_map = self.lidar_encoder(lidar)

        h_tokens = self.hsi_tokenizer(h_feat_map)
        l_tokens = self.lidar_tokenizer(l_feat_map)

        h_tokens = self.h_te(h_tokens)
        l_tokens = self.l_te(l_tokens)

        for block in self.fusion_blocks:
            h_tokens, l_tokens = block(h_tokens, l_tokens)

        h_cls = h_tokens[:, 0]
        l_cls = l_tokens[:, 0]
        fused_token = self.gated_fuse(h_cls, l_cls)
        logits = self.classifier(fused_token)
        return {
            "logits": logits,
            "h_cls": h_cls,
            "l_cls": l_cls,
            "fused_token": fused_token,
        }
