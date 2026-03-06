"""
    最小可运行的baseline（双分支CNN融合网络）
    包括：
    HsiBranch:3D Conv提取光谱特征 + 2D Conv提取空间特征
    LidarBranch：Conv2D提取空间特征
    BaselineFusionNet
    入口：BaselineFusionNet
"""
from __future__ import annotations

import torch
from torch import nn


class HsiBranch(nn.Module):
    """
    HSI分支
    输入： (B, C, H, W) B = batch size, C = 光谱通道数, H/W = patch大小(如11 * 11)
    """
    def __init__(
        self,
        in_channels: int,
        spectral_channels: int = 8,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        # 3D Conv提取光谱特征，卷积核大小(7, 3, 3)覆盖整个光谱维度，padding保持空间维度不变
        self.spectral = nn.Sequential(
            nn.Conv3d(1, spectral_channels, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(spectral_channels, spectral_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
        )
        # 2D Conv提取空间特征，卷积核大小(3, 3)覆盖整个空间维度，padding保持光谱维度不变
        self.spatial = nn.Sequential(
            nn.Conv2d(spectral_channels * in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected HSI input with shape (B, C, H, W), got {tuple(x.shape)}")
        x = x.unsqueeze(1)   # (B, C, H, W) -> (B, 1, C, H, W)，后者是Conv 3D的期望输入
        x = self.spectral(x)
        b, c3d, bands, h, w = x.shape
        x = x.reshape(b, c3d * bands, h, w)
        x = self.spatial(x)
        return torch.flatten(x, 1)


class LidarBranch(nn.Module):
    """
    LiDAR分支
    输入： (B, 1, H, W) B = batch size, 1 = LiDAR通道数, H/W = patch大小(如11 * 11)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 32) -> None:
        super().__init__()
        # Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected LiDAR input with shape (B, 1, H, W), got {tuple(x.shape)}")
        x = self.encoder(x)
        return torch.flatten(x, 1)


class BaselineFusionNet(nn.Module):
    """
    双分支融合网络

    """
    def __init__(
        self,
        hsi_in_channels: int,
        num_classes: int,
        hsi_feature_dim: int = 64,
        lidar_feature_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hsi_branch = HsiBranch(
            in_channels=hsi_in_channels,
            out_channels=hsi_feature_dim,
        )
        self.lidar_branch = LidarBranch(out_channels=lidar_feature_dim)
        fusion_dim = self.hsi_branch.out_channels + self.lidar_branch.out_channels
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        # HSI patch 经过 HSI encoder，得到 h_feat, LiDAR同理, 然后俩特征在特征维拼接，得到 fused，最后经过分类器得到每个类别的打分
        h_feat = self.hsi_branch(hsi)
        l_feat = self.lidar_branch(lidar)
        fused = torch.cat([h_feat, l_feat], dim=1)
        return self.classifier(fused)  # classifier: Linear -> ReLU -> Dropout -> Linear, 输出的是每个类别的打分，不是概率，后面训练的时候会直接交给crossEntropyLoss
