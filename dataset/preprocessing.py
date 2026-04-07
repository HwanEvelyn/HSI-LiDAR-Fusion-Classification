"""
    标准化 z-score + PCA + 边界 padding
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    x_pca: np.ndarray
    mean: np.ndarray
    components: np.ndarray


@dataclass
class ZScoreStats:
    mean: np.ndarray | float
    std: np.ndarray | float


def fit_zscore(x: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> ZScoreStats:
    x = np.asarray(x, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape[:2]:
            raise ValueError("mask shape must match the spatial shape of x")

    if x.ndim == 3:
        flat = x.reshape(-1, x.shape[-1]) if mask is None else x[mask]
        mean = flat.mean(axis=0)
        std = np.maximum(flat.std(axis=0), eps)
        return ZScoreStats(mean=mean.astype(np.float32), std=std.astype(np.float32))

    flat = x.reshape(-1) if mask is None else x[mask]
    mean = float(flat.mean())
    std = max(float(flat.std()), eps)
    return ZScoreStats(mean=mean, std=std)


def apply_zscore(x: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - stats.mean) / stats.std


def zscore_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return apply_zscore(x, fit_zscore(x, mask=None, eps=eps))


def zscore_norm_with_mask(
    x: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, ZScoreStats]:
    """
    用训练像素统计均值方差
    """
    stats = fit_zscore(x, mask=mask, eps=eps)
    return apply_zscore(x, stats), stats


def _fit_pca_from_flat(x_2d: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    x_2d = np.asarray(x_2d, dtype=np.float64)
    if not np.isfinite(x_2d).all():
        raise ValueError("PCA 输入包含 NaN 或 Inf，无法继续拟合")

    mean = x_2d.mean(axis=0)
    x_centered = x_2d - mean
    cov = np.einsum("ni,nj->ij", x_centered, x_centered) / max(x_centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    components = eigvecs[:, :n_components].T
    return mean.astype(np.float32), components.astype(np.float32)


def pca_project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("PCA 投影输入包含 NaN 或 Inf，无法继续计算")

    h, w, c = x.shape
    x_centered = x.reshape(-1, c) - mean
    x_pca_2d = np.einsum("nc,dc->nd", x_centered, components)
    return x_pca_2d.reshape(h, w, components.shape[0]).astype(np.float32)


def pca_reduce(x: np.ndarray, n_components: int) -> PCAResult:
    x = np.asarray(x, dtype=np.float32)
    h, w, c = x.shape
    mean, components = _fit_pca_from_flat(x.reshape(-1, c), n_components)
    x_pca = pca_project(x, mean, components)
    return PCAResult(x_pca=x_pca, mean=mean, components=components)


def pca_reduce_with_mask(
    x: np.ndarray,
    n_components: int,
    mask: np.ndarray | None = None,
) -> PCAResult:
    """
    只用训练像素拟合 PCA，再投影整景
    """
    x = np.asarray(x, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != x.shape[:2]:
            raise ValueError("mask shape must match the first two dimensions of x")
        x_fit = x[mask]
    else:
        x_fit = x.reshape(-1, x.shape[-1])

    mean, components = _fit_pca_from_flat(x_fit, n_components)
    x_pca = pca_project(x, mean, components)
    return PCAResult(x_pca=x_pca, mean=mean, components=components)


def pad_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return x
    if x.ndim == 3:
        return np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
