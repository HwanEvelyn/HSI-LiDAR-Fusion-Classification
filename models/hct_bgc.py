"""
模态内 Transformer Encoder（分 HSI 和 LiDAR），先各自学习本模态内部的长程依赖，再做跨模态融合（fusion_blocks.py）
输入输出：两路 token 序列(B, 1+p^2, D)

"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .fusion_blocks import (
    BiDirectionalClassTokenAttention,
    CrossTokenAttention,
    FeedForwardBlock,
    GatedCrossModalFusion,
    SimpleAverageFusion,
)
from .hct_backbone import HsiCnnEncoder, LidarCnnEncoder, Tokenizer


class CrossScaleClassTokenInteraction(nn.Module):
    """
    跨尺度 cls-token 交互：
    - local_cls <- attend(context_tokens)
    - context_cls <- attend(local_tokens)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.local_queries_context = CrossTokenAttention(embed_dim, num_heads, dropout=dropout, conservative=True)
        self.context_queries_local = CrossTokenAttention(embed_dim, num_heads, dropout=dropout, conservative=True)
        self.local_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)
        self.context_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)

    def forward(
        self,
        local_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_cls, local_rest = local_tokens[:, :1], local_tokens[:, 1:]
        context_cls, context_rest = context_tokens[:, :1], context_tokens[:, 1:]

        local_cls = self.local_queries_context(local_cls, context_tokens)
        context_cls = self.context_queries_local(context_cls, local_tokens)
        local_cls = self.local_ffn(local_cls)
        context_cls = self.context_ffn(context_cls)

        local_tokens = torch.cat([local_cls, local_rest], dim=1)
        context_tokens = torch.cat([context_cls, context_rest], dim=1)
        return local_tokens, context_tokens


class LocalDominantScaleFusion(nn.Module):
    """
    可学习尺度融合：
    - 学习 local/context 的逐维尺度门控
    - 最终形式为 local 主导，context 仅作为残差修正
    """

    def __init__(self, embed_dim: int, init_strength: float = 0.15) -> None:
        super().__init__()
        self.scale_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.context_strength = nn.Parameter(torch.full((1, embed_dim), init_strength))

    def compute_gate(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale_gate(torch.cat([local_cls, context_cls], dim=1)))

    def forward(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.compute_gate(local_cls, context_cls)
        residual = gate * (context_cls - local_cls)
        strength = torch.sigmoid(self.context_strength)
        fused = local_cls + strength * residual
        return fused, gate


class GatedScaleFusion(nn.Module):
    """
    可学习尺度融合：
    - 不强制 local 主导
    - 由门控决定 local/context 的逐维占比
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.scale_gate = nn.Linear(embed_dim * 2, embed_dim)

    def compute_gate(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale_gate(torch.cat([local_cls, context_cls], dim=1)))

    def forward(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.compute_gate(local_cls, context_cls)
        fused = gate * local_cls + (1.0 - gate) * context_cls
        return fused, gate


class AverageScaleFusion(nn.Module):
    """固定平均的尺度融合基线。"""

    def forward(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.full_like(local_cls, 0.5)
        fused = 0.5 * (local_cls + context_cls)
        return fused, gate


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
        context_patch_size: int | None = None,
        context_token_size: int = 0,
        scale_fusion_mode: str = "residual",
        disable_gate: bool = False,
        encoder_variant: str = "hetero",
        use_conservative_fusion: bool = False,
        use_aux_heads: bool = False,
        aux_head_mode: str = "linear",
    ) -> None:
        super().__init__()
        if fusion_layers not in {1, 2, 3}:
            raise ValueError(f"HCT_BGC-v1 仅支持 1-3 层 Bi-CTA 堆叠，当前收到 fusion_layers={fusion_layers}")
        if context_patch_size is not None and context_patch_size < patch_size:
            raise ValueError("context_patch_size 必须 >= patch_size")
        if context_token_size < 0:
            raise ValueError("context_token_size 必须 >= 0")
        self.patch_size = patch_size
        self.context_patch_size = context_patch_size
        self.use_multiscale = context_patch_size is not None and context_patch_size > patch_size
        num_spatial_tokens = patch_size * patch_size
        self.context_token_size = context_token_size if context_token_size > 0 else (
            context_patch_size if self.use_multiscale else patch_size
        )
        context_spatial_tokens = (
            self.context_token_size * self.context_token_size if self.use_multiscale else num_spatial_tokens
        )
        self.config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "fusion_layers": fusion_layers,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "patch_size": patch_size,
            "context_patch_size": context_patch_size if context_patch_size is not None else patch_size,
            "context_token_size": self.context_token_size,
            "multiscale_mode": "cross_scale_residual" if self.use_multiscale else "single_scale",
            "scale_fusion_mode": scale_fusion_mode,
            "num_spatial_tokens": num_spatial_tokens,
            "context_spatial_tokens": context_spatial_tokens,
            "disable_gate": disable_gate,
            "fusion_mode": "average" if disable_gate else "gated",
            "encoder_variant": encoder_variant,
            "use_conservative_fusion": use_conservative_fusion,
            "use_aux_heads": use_aux_heads,
            "aux_head_mode": aux_head_mode,
        }
        self.hsi_encoder = HsiCnnEncoder(hsi_in_channels, embed_dim=embed_dim, variant=encoder_variant)
        self.lidar_encoder = LidarCnnEncoder(in_channels=lidar_in_channels, embed_dim=embed_dim, variant=encoder_variant)

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
        if self.use_multiscale:
            self.hsi_context_tokenizer = Tokenizer(
                self.hsi_encoder.out_channels,
                embed_dim=embed_dim,
                num_spatial_tokens=context_spatial_tokens,
                dropout=dropout,
            )
            self.lidar_context_tokenizer = Tokenizer(
                self.lidar_encoder.out_channels,
                embed_dim=embed_dim,
                num_spatial_tokens=context_spatial_tokens,
                dropout=dropout,
            )
        else:
            self.hsi_context_tokenizer = None
            self.lidar_context_tokenizer = None

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
                    conservative=use_conservative_fusion,
                )
                for _ in range(fusion_layers)
            ]
        )
        if self.use_multiscale:
            self.hsi_cross_scale = CrossScaleClassTokenInteraction(embed_dim, num_heads, mlp_dim, dropout)
            self.lidar_cross_scale = CrossScaleClassTokenInteraction(embed_dim, num_heads, mlp_dim, dropout)
            if scale_fusion_mode == "residual":
                self.hsi_scale_fusion = LocalDominantScaleFusion(embed_dim)
                self.lidar_scale_fusion = LocalDominantScaleFusion(embed_dim)
            elif scale_fusion_mode == "gated":
                self.hsi_scale_fusion = GatedScaleFusion(embed_dim)
                self.lidar_scale_fusion = GatedScaleFusion(embed_dim)
            elif scale_fusion_mode == "average":
                self.hsi_scale_fusion = AverageScaleFusion()
                self.lidar_scale_fusion = AverageScaleFusion()
            else:
                raise ValueError(f"Unsupported scale_fusion_mode: {scale_fusion_mode}")
        else:
            self.hsi_cross_scale = None
            self.lidar_cross_scale = None
            self.hsi_scale_fusion = None
            self.lidar_scale_fusion = None
        self.fusion_mode = "average" if disable_gate else "gated"
        self.cls_fusion = SimpleAverageFusion() if disable_gate else GatedCrossModalFusion(
            embed_dim,
            conservative=use_conservative_fusion,
        )
        self.use_aux_heads = use_aux_heads
        self.aux_head_mode = aux_head_mode
        # 分类头：输入 fused_token(B, D),输出 logits(B, num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )
        if use_aux_heads:
            if aux_head_mode == "linear":
                self.hsi_aux_classifier = nn.Linear(embed_dim, num_classes)
                self.lidar_aux_classifier = nn.Linear(embed_dim, num_classes)
            elif aux_head_mode == "mlp":
                self.hsi_aux_classifier = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, num_classes),
                )
                self.lidar_aux_classifier = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, num_classes),
                )
            else:
                raise ValueError(f"Unsupported aux_head_mode: {aux_head_mode}")
        else:
            self.hsi_aux_classifier = None
            self.lidar_aux_classifier = None

    def get_config(self) -> dict[str, int | float]:
        return dict(self.config)

    def _center_crop(self, x: torch.Tensor, crop_size: int) -> torch.Tensor:
        if x.size(-1) < crop_size or x.size(-2) < crop_size:
            raise ValueError(
                f"输入 patch 尺寸 {tuple(x.shape[-2:])} 小于请求裁剪尺寸 {crop_size}"
            )
        if x.size(-1) == crop_size and x.size(-2) == crop_size:
            return x
        top = (x.size(-2) - crop_size) // 2
        left = (x.size(-1) - crop_size) // 2
        return x[..., top : top + crop_size, left : left + crop_size]

    def _pool_context_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_multiscale:
            return feat
        if feat.size(-1) == self.context_token_size and feat.size(-2) == self.context_token_size:
            return feat
        return F.adaptive_avg_pool2d(feat, output_size=(self.context_token_size, self.context_token_size))

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_multiscale:
            hsi_local = self._center_crop(hsi, self.patch_size)
            lidar_local = self._center_crop(lidar, self.patch_size)

            h_local_feat = self.hsi_encoder(hsi_local)
            l_local_feat = self.lidar_encoder(lidar_local)
            h_context_feat = self._pool_context_feat(self.hsi_encoder(hsi))
            l_context_feat = self._pool_context_feat(self.lidar_encoder(lidar))

            h_local_tokens = self.h_te(self.hsi_tokenizer(h_local_feat))
            l_local_tokens = self.l_te(self.lidar_tokenizer(l_local_feat))
            h_context_tokens = self.h_te(self.hsi_context_tokenizer(h_context_feat))
            l_context_tokens = self.l_te(self.lidar_context_tokenizer(l_context_feat))

            for block in self.fusion_blocks:
                h_local_tokens, l_local_tokens = block(h_local_tokens, l_local_tokens)
                h_context_tokens, l_context_tokens = block(h_context_tokens, l_context_tokens)

            h_local_tokens, h_context_tokens = self.hsi_cross_scale(h_local_tokens, h_context_tokens)
            l_local_tokens, l_context_tokens = self.lidar_cross_scale(l_local_tokens, l_context_tokens)

            h_cls, h_scale_gate = self.hsi_scale_fusion(h_local_tokens[:, 0], h_context_tokens[:, 0])
            l_cls, l_scale_gate = self.lidar_scale_fusion(l_local_tokens[:, 0], l_context_tokens[:, 0])
        else:
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
            h_scale_gate = None
            l_scale_gate = None
        if self.fusion_mode == "gated":
            gate = self.cls_fusion.compute_gate(h_cls, l_cls)
            fused_token = self.cls_fusion(h_cls, l_cls)
        else:
            gate = torch.full_like(h_cls, 0.5)
            fused_token = self.cls_fusion(h_cls, l_cls)
        logits = self.classifier(fused_token)
        outputs = {
            "logits": logits,   # 最终分类分数
            "h_cls": h_cls,     # hsi 的最终摘要特征
            "l_cls": l_cls,     # lidar。。
            "fused_token": fused_token,     # 融合后的联合表征
            "gate": gate,       # 门控权重；平均融合时固定为 0.5
        }
        if h_scale_gate is not None and l_scale_gate is not None:
            outputs["h_scale_gate"] = h_scale_gate
            outputs["l_scale_gate"] = l_scale_gate
        if self.use_aux_heads:
            outputs["h_logits"] = self.hsi_aux_classifier(h_cls)
            outputs["l_logits"] = self.lidar_aux_classifier(l_cls)
        return outputs
