"""
    pipeline:
    1.读取 data_hl.mat文件，提取HSI、LiDAR和GT数据 √
    2.Norm → PCA → Patch 
    3.划分训练样本列表 + patch数据集
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from scipy.io import loadmat

# define the data structure for the loaded .mat file
@dataclass
class MatData:
    hsi:np.ndarray    # (H,W,C)
    lidar:np.ndarray  # (H,W)
    gt:np.ndarray     # (H,W)

def load_houston_hl(mat_path: str) -> MatData:
    """
    Load HSI, LiDAR, and GT arrays from Houston_2013_data_hl.mat file 
    
    Args:
        mat_path: Path to .mat file.

    Returns:
        MatData with np.float32 for hsi/lidar and np.int64 for gt
    """
    mat = loadmat(mat_path)

    data = mat["data"]
    gt = mat["AllTrueLabel"]

    data = np.asarray(data, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.int64)

    hsi = data[:, :, :-1]  # (H, W, C) 前144个通道是HSI
    lidar = data[:, :, -1]  # (H, W) 最后一个通道是LiDAR

    return MatData(hsi=hsi, lidar=lidar, gt=gt)

