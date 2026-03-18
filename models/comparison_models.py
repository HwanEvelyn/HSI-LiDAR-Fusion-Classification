from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .baseline_cnn import HsiBranch, LidarBranch
from .hct_backbone import HsiCnnEncoder, LidarCnnEncoder, Tokenizer


class HsiOnlyNet(nn.Module):
    def __init__(
        self,
        hsi_in_channels: int,
        num_classes: int,
        hsi_feature_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hsi_branch = HsiBranch(in_channels=hsi_in_channels, out_channels=hsi_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.hsi_branch.out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        del lidar
        h_feat = self.hsi_branch(hsi)
        return self.classifier(h_feat)


class LidarOnlyNet(nn.Module):
    def __init__(
        self,
        lidar_in_channels: int,
        num_classes: int,
        lidar_feature_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.lidar_branch = LidarBranch(in_channels=lidar_in_channels, out_channels=lidar_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.lidar_branch.out_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        del hsi
        l_feat = self.lidar_branch(lidar)
        return self.classifier(l_feat)


class CnnTransformerNoFusion(nn.Module):
    """
    双分支 CNN + Transformer，不使用 Bi-CTA 和 Gate。
    仅将两路 cls token 直接拼接后送入分类头。
    """

    def __init__(
        self,
        hsi_in_channels: int,
        lidar_in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        patch_size: int = 11,
    ) -> None:
        super().__init__()
        num_spatial_tokens = patch_size * patch_size
        self.config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "patch_size": patch_size,
            "num_spatial_tokens": num_spatial_tokens,
            "fusion_mode": "concat_cls",
        }
        self.hsi_encoder = HsiCnnEncoder(hsi_in_channels, embed_dim=embed_dim)
        self.lidar_encoder = LidarCnnEncoder(in_channels=lidar_in_channels, embed_dim=embed_dim)
        self.hsi_tokenizer = Tokenizer(
            self.hsi_encoder.out_channels,
            embed_dim=embed_dim,
            num_spatial_tokens=num_spatial_tokens,
            dropout=dropout,
        )
        self.lidar_tokenizer = Tokenizer(
            self.lidar_encoder.out_channels,
            embed_dim=embed_dim,
            num_spatial_tokens=num_spatial_tokens,
            dropout=dropout,
        )
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def get_config(self) -> dict[str, int | float | str]:
        return dict(self.config)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_feat_map = self.hsi_encoder(hsi)
        l_feat_map = self.lidar_encoder(lidar)
        h_tokens = self.h_te(self.hsi_tokenizer(h_feat_map))
        l_tokens = self.l_te(self.lidar_tokenizer(l_feat_map))
        h_cls = h_tokens[:, 0]
        l_cls = l_tokens[:, 0]
        fused_token = torch.cat([h_cls, l_cls], dim=1)
        logits = self.classifier(fused_token)
        return {
            "logits": logits,
            "h_cls": h_cls,
            "l_cls": l_cls,
            "fused_token": fused_token,
        }
