"""
    pipeline:
    1.读取 data_hl.mat文件，提取HSI、LiDAR和GT数据
    2.Norm → PCA → Patch √
    3.划分训练样本列表 + patch数据集
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class PCAResult:
    x_pca: np.ndarray  # (H, W, b)
    mean: np.ndarray  # (C,)
    components: np.ndarray  # (b, C)

def pca_reduce(x: np.ndarray, n_components: int) -> PCAResult:
    """
    pca降维，保留n_components个主成分，SVD改成了基于协方差矩阵的特征分解，好像说会快一些
    
    args:
        x: HSI cube (H, W, C)
        n_components: b个主成分 int
    returns:
        PCAResult with x_pca (H, W, b), mean (C,), components (b, C)
    """
    x = np.asarray(x, dtype=np.float32)
    h, w, c = x.shape
    x_2d = x.reshape(-1, c)  # (H*W, C) pca需要二维数据

    mean = x_2d.mean(axis=0)  # (C,)
    x_centered = x_2d - mean  # (H*W, C) 中心化

    # U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)  # SVD分解
    # components = Vt[:n_components]  # (b, C) 前b个主成分
    cov = (x_centered.T @ x_centered) / max(x_centered.shape[0] - 1, 1)  # (C, C)
    eigvals, eigvecs = np.linalg.eigh(cov)       # eigvecs: (C, C)

    idx = np.argsort(eigvals)[::-1]              # descending
    eigvecs = eigvecs[:, idx]                    # (C, C)

    components = eigvecs[:, :n_components].T     # (b, C)
    x_pca_2d = x_centered @ components.T         # (N, b) = (H*W, b) 降维后的数据，投影到新空间
    x_pca = x_pca_2d.reshape(h, w, n_components) # (H, W, b) 恢复成原来的空间结构

    return PCAResult(
        x_pca=x_pca.astype(np.float32),
        mean=mean.astype(np.float32),
        components=components.astype(np.float32)
    )

def zscore_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    z-score标准化

    args:
        x: 输入数据 (H, W, C) 或者 (H, W)
        eps: 防止除以0的小常数
    returns:
        标准化后的数据 (H, W, C)
    """
    x = x.astype(np.float32)
    if(x.ndim == 3):
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        std = x.reshape(-1, x.shape[-1]).std(axis=0)
        std = np.maximum(std, eps)
        return (x - mean) / std
    mean = float(x.mean())
    std = float(x.std())
    std = max(std, eps)
    return (x - mean) / std

def pad_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    """
    填充边界，使用反射填充

    args:
        x: 输入数据 (H, W, C) 或者 (H, W)
        pad: 填充大小 int

    returns:
        填充后的数据 (H+2*pad, W+2*pad, C) 或者 (H+2*pad, W+2*pad)
    """
    if pad <= 0:
        return x
    if x.ndim == 3:
        return np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    return np.pad(x, ((pad, pad), (pad, pad)), mode='reflect')

