"""
    pipeline:
    1.读取 data_hl.mat文件，提取HSI、LiDAR和GT数据
    2.Norm → PCA → Patch 
    3.划分训练样本列表 + patch数据集 √
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import pad_reflect

@dataclass
class IndexItem:  # 一个样本的结构，(r, c) = 像素位置，y = 类别标签
    r: int
    c: int
    y: int

def bulid_index(gt: np.ndarray, train_ratio: float, seed: int = 0) -> Tuple[List[IndexItem], List[IndexItem], int]:
    """
    从GT(H, W)中构建训练集和测试集的标签列表，忽略label=0的像素

    args:
        gt: (H, W) 每个像素的类别标签
        train_ratio: 训练集占总数据的比例
        seed: 随机种子，保证划分的可重复性
    returns:
        train_items, test_items, num_classes
    """
    rng = np.random.default_rng(seed)
    labels = gt[gt > 0]  # 只考虑标签大于0的像素
    num_classes = int(labels.max())  # 类别数等于最大标签值
    
    train_items: List[IndexItem] = []
    test_items: List[IndexItem] = []

    for cls in range(1, num_classes + 1):
        coords = np.argwhere(gt == cls)  # (N_cls, 2) 获取这个类别的所有像素坐标
        rng.shuffle(coords)  # 打乱
        n_train = int(len(coords) * train_ratio)  # 划分训练/测试
        train_coords = coords[:n_train]  # 前train_ratio的像素作为训练集
        test_coords = coords[n_train:]  # 剩余的像素作为测试集

        for r, c in train_coords:
            train_items.append(IndexItem(int(r), int(c), cls - 1))  # 存储训练集索引，标签减1变成0-based
        for r, c in test_coords:
            test_items.append(IndexItem(int(r), int(c), cls - 1))  # 存储测试集索引

    rng.shuffle(train_items)  # 打乱训练集索引顺序
    rng.shuffle(test_items)  # 打乱测试集索引顺序
    return train_items, test_items, num_classes

class HsiLidarPatchDataset(Dataset):
    """
    根据 IndexItem 提取 patch 数据
    """
    def __init__(
            self,
            hsi: np.ndarray,
            lidar: np.ndarray,
            items: List[IndexItem],
            patch_size: int,
    ) -> None:
        self.items = items
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.hsi_pad = pad_reflect(hsi, self.pad)  # (H+2*pad, W+2*pad, C)
        self.lidar_pad = pad_reflect(lidar, self.pad)  # (H+2*pad, W+2*pad)

    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        r, c, y = item.r, item.c, item.y
        rp, cp = r + self.pad, c + self.pad  # 加上pad偏移

        hsi_patch = self.hsi_pad[rp - self.pad:rp + self.pad + 1, cp - self.pad:cp + self.pad + 1, :]  # (patch_size, patch_size, C)
        lidar_patch = self.lidar_pad[rp - self.pad:rp + self.pad + 1, cp - self.pad:cp + self.pad + 1]  # (patch_size, patch_size)

        # 转换成torch.Tensor，hsi_patch (C, patch_size, patch_size)，lidar_patch (1, patch_size, patch_size)，y (scalar)
        hsi_t = torch.from_numpy(hsi_patch).permute(2, 0, 1).contiguous()  # (C, patch_size, patch_size)
        lidar_t = torch.from_numpy(lidar_patch).unsqueeze(0).contiguous()  # (1, patch_size, patch_size)
        y_t = torch.tensor(y, dtype=torch.long)  # (scalar)
        return hsi_t, lidar_t, y_t