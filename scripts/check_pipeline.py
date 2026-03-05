from __future__ import annotations

import numpy as np

from dataset.mat_loader import load_houston_hl
from dataset.preprocessing import zscore_norm, pca_reduce
from dataset.patch_dataset import bulid_index, HsiLidarPatchDataset

def main() -> None:
    data = load_houston_hl(
        "data/raw/Houston 2013/Houston_2013_data_hl.mat"
    )
    hsi, lidar, gt = data.hsi, data.lidar, data.gt

    print(hsi.shape)    # (349,1905,144)
    print(lidar.shape)  # (349,1905)
    print(gt.shape)     # (349,1905)

    hsi_norm = zscore_norm(data.hsi)
    lidar_norm = zscore_norm(data.lidar)

    # 1.Norm
    hsi_n = zscore_norm(hsi)
    lidar_n = zscore_norm(lidar)

    # 检查 labeled pixel的均值和标准差
    labeled_pixels_hsi = hsi_n[gt != 0]
    labeled_pixels_lidar = lidar_n[gt != 0]

    print(f"HSI labeled pixels - Mean: {labeled_pixels_hsi.mean()}, Std: {labeled_pixels_hsi.std()}")
    print(f"LiDAR labeled pixels - Mean: {labeled_pixels_lidar.mean()}, Std: {labeled_pixels_lidar.std()}")

    # 2.PCA
    pca_result = pca_reduce(hsi_n, n_components=30)
    x_pca = pca_result.x_pca
    print("pca shape:",x_pca.shape)  # (349, 1905, 30)

    # pca检查：mean应该接近0，std应该接近1
    assert np.isfinite(x_pca).all(), "PCA result contains inf values"
    print("PCA stats:", float(x_pca.mean()), float(x_pca.std()),float(x_pca.min()), float(x_pca.max()))

    # 3.检查组件的形状
    print("PCA components shape:", pca_result.components.shape,"mean shape", pca_result.mean.shape)  # (30, 144)
    assert pca_result.components.shape == (30, hsi.shape[-1])

    print("check passed")

if __name__ == "__main__":
    main()