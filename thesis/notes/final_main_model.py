"""
Final HCT-BGC main architecture summary.

This file is a compact, code-level reference for the final main protocol only:
- local patch: 11 x 11
- context patch: 17 x 17
- context token grid: 11 x 11
- encoder: light_hetero
- scale fusion: local-dominant residual fusion
- cross-modal fusion: conservative gated fusion

It intentionally excludes dataset splitting, normalization, PCA, ablation switches,
and alternative parameter variants.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FinalHCTBGCConfig:
    patch_size: int = 11
    context_patch_size: int = 17
    context_token_size: int = 11
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    fusion_layers: int = 1
    mlp_dim: int = 256
    dropout: float = 0.1
    use_aux_heads: bool = True


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class HsiLightHeteroEncoder(nn.Module):
    """HSI light_hetero encoder: 1 x 1 spectral mixing + 3 x 3 spatial encoding."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        hidden_dim = max(embed_dim // 2, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LidarLightHeteroEncoder(nn.Module):
    """LiDAR light_hetero encoder: spatial convolution + residual structure encoding."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        stem_dim = max(embed_dim // 4, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            ResidualConvBlock(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Tokenizer(nn.Module):
    """Feature map -> [CLS token + spatial tokens] + positional embedding."""

    def __init__(self, embed_dim: int, spatial_size: int, dropout: float) -> None:
        super().__init__()
        token_count = spatial_size * spatial_size + 1
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, _, _, _ = x.shape
        spatial_tokens = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_token, spatial_tokens], dim=1)
        tokens = tokens + self.pos_embed
        return self.dropout(self.norm(tokens))


class CrossTokenAttention(nn.Module):
    """Conservative CLS-token attention to another token sequence."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, cls_token: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(cls_token)
        kv = self.norm_kv(context_tokens)
        updated, _ = self.attn(q, kv, kv, need_weights=False)
        updated = self.residual_scale * self.dropout(updated)
        return cls_token + updated


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class BiDirectionalClassTokenAttention(nn.Module):
    """Same-scale HSI-LiDAR interaction by bidirectional CLS-token attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.hsi_cls_queries_lidar = CrossTokenAttention(embed_dim, num_heads, dropout)
        self.lidar_cls_queries_hsi = CrossTokenAttention(embed_dim, num_heads, dropout)
        self.hsi_cls_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout)
        self.lidar_cls_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout)

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_cls, h_rest = h_tokens[:, :1], h_tokens[:, 1:]
        l_cls, l_rest = l_tokens[:, :1], l_tokens[:, 1:]

        h_cls = self.hsi_cls_queries_lidar(h_cls, l_tokens)
        l_cls = self.lidar_cls_queries_hsi(l_cls, h_tokens)
        h_cls = self.hsi_cls_ffn(h_cls)
        l_cls = self.lidar_cls_ffn(l_cls)

        return torch.cat([h_cls, h_rest], dim=1), torch.cat([l_cls, l_rest], dim=1)


class CrossScaleClassTokenInteraction(nn.Module):
    """Same-modality local-context CLS interaction."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.local_queries_context = CrossTokenAttention(embed_dim, num_heads, dropout)
        self.context_queries_local = CrossTokenAttention(embed_dim, num_heads, dropout)
        self.local_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout)
        self.context_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout)

    def forward(self, local_tokens: torch.Tensor, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_cls, local_rest = local_tokens[:, :1], local_tokens[:, 1:]
        context_cls, context_rest = context_tokens[:, :1], context_tokens[:, 1:]

        local_cls = self.local_queries_context(local_cls, context_tokens)
        context_cls = self.context_queries_local(context_cls, local_tokens)
        local_cls = self.local_ffn(local_cls)
        context_cls = self.context_ffn(context_cls)

        return torch.cat([local_cls, local_rest], dim=1), torch.cat([context_cls, context_rest], dim=1)


class LocalDominantScaleFusion(nn.Module):
    """local-dominant residual scale fusion: local + controlled context correction."""

    def __init__(self, embed_dim: int, init_strength: float = 0.15) -> None:
        super().__init__()
        self.scale_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.context_strength = nn.Parameter(torch.full((1, embed_dim), init_strength))

    def forward(self, local_cls: torch.Tensor, context_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.scale_gate(torch.cat([local_cls, context_cls], dim=1)))
        residual = gate * (context_cls - local_cls)
        fused_cls = local_cls + torch.sigmoid(self.context_strength) * residual
        return fused_cls, gate


class ConservativeGatedCrossModalFusion(nn.Module):
    """Gated HSI-LiDAR fusion constrained around average fusion."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_strength = nn.Parameter(torch.tensor(0.25))

    def compute_gate(self, h_cls: torch.Tensor, l_cls: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([h_cls, l_cls], dim=1)))

    def forward(self, h_cls: torch.Tensor, l_cls: torch.Tensor) -> torch.Tensor:
        gate = self.compute_gate(h_cls, l_cls)
        gated = gate * h_cls + (1.0 - gate) * l_cls
        base = 0.5 * (h_cls + l_cls)
        strength = torch.clamp(self.fusion_strength, min=0.0, max=1.0)
        return base + strength * (gated - base)


class FinalHCTBGC(nn.Module):
    """Single-file reference implementation of the final main HCT-BGC protocol."""

    def __init__(self, hsi_in_channels: int, lidar_in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cfg = FinalHCTBGCConfig()
        cfg = self.cfg

        self.hsi_encoder = HsiLightHeteroEncoder(hsi_in_channels, cfg.embed_dim)
        self.lidar_encoder = LidarLightHeteroEncoder(lidar_in_channels, cfg.embed_dim)

        self.hsi_tokenizer = Tokenizer(cfg.embed_dim, cfg.patch_size, cfg.dropout)
        self.lidar_tokenizer = Tokenizer(cfg.embed_dim, cfg.patch_size, cfg.dropout)
        self.hsi_context_tokenizer = Tokenizer(cfg.embed_dim, cfg.context_token_size, cfg.dropout)
        self.lidar_context_tokenizer = Tokenizer(cfg.embed_dim, cfg.context_token_size, cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.h_te = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.l_te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.mlp_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=cfg.num_layers,
        )

        self.fusion_blocks = nn.ModuleList(
            [
                BiDirectionalClassTokenAttention(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout)
                for _ in range(cfg.fusion_layers)
            ]
        )
        self.hsi_cross_scale = CrossScaleClassTokenInteraction(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout)
        self.lidar_cross_scale = CrossScaleClassTokenInteraction(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout)
        self.hsi_scale_fusion = LocalDominantScaleFusion(cfg.embed_dim)
        self.lidar_scale_fusion = LocalDominantScaleFusion(cfg.embed_dim)
        self.cls_fusion = ConservativeGatedCrossModalFusion(cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, num_classes),
        )
        self.hsi_aux_classifier = nn.Linear(cfg.embed_dim, num_classes)
        self.lidar_aux_classifier = nn.Linear(cfg.embed_dim, num_classes)

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        top = (x.size(-2) - cfg.patch_size) // 2
        left = (x.size(-1) - cfg.patch_size) // 2
        return x[..., top : top + cfg.patch_size, left : left + cfg.patch_size]

    def _pool_context_feat(self, feat: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return F.adaptive_avg_pool2d(feat, output_size=(cfg.context_token_size, cfg.context_token_size))

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> dict[str, torch.Tensor]:
        hsi_local = self._center_crop(hsi)
        lidar_local = self._center_crop(lidar)

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
        gate = self.cls_fusion.compute_gate(h_cls, l_cls)
        fused_token = self.cls_fusion(h_cls, l_cls)
        logits = self.classifier(fused_token)

        return {
            "logits": logits,
            "h_cls": h_cls,
            "l_cls": l_cls,
            "fused_token": fused_token,
            "gate": gate,
            "h_scale_gate": h_scale_gate,
            "l_scale_gate": l_scale_gate,
            "h_logits": self.hsi_aux_classifier(h_cls),
            "l_logits": self.lidar_aux_classifier(l_cls),
        }
