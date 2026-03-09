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


class BiCTAFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.h_to_l = CrossTokenAttention(embed_dim, num_heads, dropout=dropout)
        self.l_to_h = CrossTokenAttention(embed_dim, num_heads, dropout=dropout)
        self.h_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)
        self.l_ffn = FeedForwardBlock(embed_dim, mlp_dim, dropout=dropout)

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h_tokens.dim() != 3 or l_tokens.dim() != 3:
            raise ValueError("BiCTAFusionBlock expects token tensors with shape (B, N, D)")

        h_cls, h_rest = h_tokens[:, :1], h_tokens[:, 1:]
        l_cls, l_rest = l_tokens[:, :1], l_tokens[:, 1:]

        h_cls = self.h_to_l(h_cls, l_tokens)
        l_cls = self.l_to_h(l_cls, h_tokens)

        h_cls = self.h_ffn(h_cls)
        l_cls = self.l_ffn(l_cls)

        h_tokens = torch.cat([h_cls, h_rest], dim=1)
        l_tokens = torch.cat([l_cls, l_rest], dim=1)
        return h_tokens, l_tokens


class GatedFuse(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, cls_h: torch.Tensor, cls_l: torch.Tensor) -> torch.Tensor:
        if cls_h.shape != cls_l.shape:
            raise ValueError("GatedFuse expects cls_h and cls_l to have the same shape")
        gate = torch.sigmoid(self.gate(torch.cat([cls_h, cls_l], dim=1)))
        return gate * cls_h + (1.0 - gate) * cls_l
