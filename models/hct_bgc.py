"""
模态内 Transformer Encoder（分 HSI 和 LiDAR），先各自学习本模态内部的长程依赖，再做跨模态融合（fusion_blocks.py）
输入输出：两路 token 序列(B, 1+p^2, D)

"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .fusion_blocks import BiDirectionalClassTokenAttention, GatedCrossModalFusion, SimpleAverageFusion
from .hct_backbone import HsiCnnEncoder, LidarCnnEncoder, Tokenizer


class HCT_BGC(nn.Module):
    def __init__(
        self,
        hsi_in_channels: int,
        lidar_in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        fusion_layers: int = 1,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        patch_size: int = 11,
        disable_gate: bool = False,
    ) -> None:
        super().__init__()
        if fusion_layers not in {1, 2, 3}:
            raise ValueError(f"HCT_BGC-v1 仅支持 1-3 层 Bi-CTA 堆叠，当前收到 fusion_layers={fusion_layers}")
        num_spatial_tokens = patch_size * patch_size
        self.config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "fusion_layers": fusion_layers,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "patch_size": patch_size,
            "num_spatial_tokens": num_spatial_tokens,
            "disable_gate": disable_gate,
            "fusion_mode": "average" if disable_gate else "gated",
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
        self.fusion_mode = "average" if disable_gate else "gated"
        self.cls_fusion = SimpleAverageFusion() if disable_gate else GatedCrossModalFusion(embed_dim)
        # 分类头：输入 fused_token(B, D),输出 logits(B, num_classes)
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
        if self.fusion_mode == "gated":
            gate = self.cls_fusion.compute_gate(h_cls, l_cls)
            fused_token = gate * h_cls + (1.0 - gate) * l_cls
        else:
            gate = torch.full_like(h_cls, 0.5)
            fused_token = self.cls_fusion(h_cls, l_cls)
        logits = self.classifier(fused_token)
        return {
            "logits": logits,   # 最终分类分数
            "h_cls": h_cls,     # hsi 的最终摘要特征
            "l_cls": l_cls,     # lidar。。
            "fused_token": fused_token,     # 融合后的联合表征
            "gate": gate,       # 门控权重；平均融合时固定为 0.5
        }
