"""
两路特征融合：Bi-CTA + gated fusion
"""
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


class CrossTokenAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_token: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(cls_token)
        kv = self.norm_kv(context_tokens)
        updated, _ = self.attn(q, kv, kv, need_weights=False)
        return cls_token + self.dropout(updated)


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1) -> None:
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
    """
    Bi-CTA 模块：通过双向 CLS token 交互完成跨模态上下文交换。
    输入：h_tokens、 l_tokens(B, N, D)
    输出：更新后的两路 token 序列(B, N, D)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hsi_cls_queries_lidar = CrossTokenAttention(embed_dim, num_heads, dropout=dropout)
        self.lidar_cls_queries_hsi = CrossTokenAttention(embed_dim, num_heads, dropout=dropout)
        self.hsi_cls_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)
        self.lidar_cls_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h_tokens.dim() != 3 or l_tokens.dim() != 3:
            raise ValueError("BiDirectionalClassTokenAttention 期望输入 token 张量形状为 (B, N, D)")

        h_cls, h_rest = h_tokens[:, :1], h_tokens[:, 1:]
        l_cls, l_rest = l_tokens[:, :1], l_tokens[:, 1:]

        # 每个模态的 CLS token 从另一模态 token 序列中提取上下文信息。
        h_cls = self.hsi_cls_queries_lidar(h_cls, l_tokens)
        l_cls = self.lidar_cls_queries_hsi(l_cls, h_tokens)

        h_cls = self.hsi_cls_ffn(h_cls)
        l_cls = self.lidar_cls_ffn(l_cls)

        h_tokens = torch.cat([h_cls, h_rest], dim=1)
        l_tokens = torch.cat([l_cls, l_rest], dim=1)
        return h_tokens, l_tokens


class GatedCrossModalFusion(nn.Module):
    """
    Gated Fuse 模块：学习逐维门控权重来融合 HSI/LiDAR 的 CLS token。
    - g = sigmoid(Wg [h_cls; l_cls])
    - fused = g * h_cls + (1-g) * l_cls
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def compute_gate(self, cls_h: torch.Tensor, cls_l: torch.Tensor) -> torch.Tensor:
        if cls_h.shape != cls_l.shape:
            raise ValueError("GatedCrossModalFusion 期望 cls_h 和 cls_l 形状一致")
        return torch.sigmoid(self.gate(torch.cat([cls_h, cls_l], dim=1)))

    def forward(self, cls_h: torch.Tensor, cls_l: torch.Tensor) -> torch.Tensor:
        if cls_h.shape != cls_l.shape:
            raise ValueError("GatedCrossModalFusion 期望 cls_h 和 cls_l 形状一致")
        gate = self.compute_gate(cls_h, cls_l)
        return gate * cls_h + (1.0 - gate) * cls_l


class SimpleAverageFusion(nn.Module):
    """简单融合基线：直接对 HSI/LiDAR 的 CLS token 做等权平均。"""

    def forward(self, cls_h: torch.Tensor, cls_l: torch.Tensor) -> torch.Tensor:
        if cls_h.shape != cls_l.shape:
            raise ValueError("SimpleAverageFusion 期望 cls_h 和 cls_l 形状一致")
        return 0.5 * (cls_h + cls_l)


BiCTAFusionBlock = BiDirectionalClassTokenAttention
GatedFuse = GatedCrossModalFusion
