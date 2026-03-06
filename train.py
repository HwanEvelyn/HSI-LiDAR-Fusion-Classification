"""
    训练流程的脚本，把数据、模型、优化器、loss、训练循环串起来
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dataset.mat_loader import load_houston_hl
from dataset.patch_dataset import HsiLidarPatchDataset, bulid_index
from dataset.preprocessing import pca_reduce, zscore_norm
from models.baseline_cnn import BaselineFusionNet
from utils.metrics import confusion_matrix, oa_aa_kappa


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    hsi_channels: int
    num_classes: int


def build_dataloaders(
    mat_path: str,
    patch_size: int,
    pca_components: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> SplitLoaders:
    """
    负责整个数据准备流程：
    
    GT 全部有标签像素
    先切成 train/test
    再从 train 中切出一部分 val
    """
    data = load_houston_hl(mat_path)
    hsi = zscore_norm(data.hsi)
    lidar = zscore_norm(data.lidar)

    if pca_components > 0 and pca_components < hsi.shape[-1]:
        hsi = pca_reduce(hsi, n_components=pca_components).x_pca

    train_items, test_items, num_classes = bulid_index(data.gt, train_ratio=train_ratio, seed=seed)  # 按照标签划分train/test样本索引
    
    # 构造patch dataset,它会把每个像素点变成一个 patch 样本，并输出：hsi_t: (C, patch, patch),lidar_t: (1, patch, patch),y_t: 标签
    train_dataset_full = HsiLidarPatchDataset(hsi, lidar, train_items, patch_size)
    test_dataset = HsiLidarPatchDataset(hsi, lidar, test_items, patch_size)

    val_size = int(len(train_dataset_full) * val_ratio)
    train_size = len(train_dataset_full) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Adjust train_ratio or val_ratio.")
    if val_size == 0:
        val_size = min(1, len(train_dataset_full) - 1)
        train_size = len(train_dataset_full) - val_size

    generator = torch.Generator().manual_seed(seed)
    # 从训练集中切一部分出来做验证集
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    # 用 DataLoader 包装成可迭代的加载器，训练时会自动打乱训练集，验证和测试集不打乱
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return SplitLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        hsi_channels=hsi.shape[-1],
        num_classes=num_classes,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,     # 如果传入optimizier就训练，否则只评估
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    训练和验证共用的单轮循环
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    for hsi, lidar, target in loader:
        hsi = hsi.to(device=device, dtype=torch.float32, non_blocking=True)
        lidar = lidar.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.long, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)   # 清梯度

        with torch.set_grad_enabled(is_train):
            # 前向
            logits = model(hsi, lidar)
            loss = criterion(logits, target)
            if is_train:
                # 反向传播、更新参数
                loss.backward()
                optimizer.step()

        batch_size = target.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return avg_loss, y_true, y_pred


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """
    对run_epoch的结果进行评估，计算混淆矩阵和相关指标
    返回结果是一个字典，包含loss、OA、AA和Kappa
    """
    loss, y_true, y_pred = run_epoch(model, loader, criterion, device, optimizer=None)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    metrics = oa_aa_kappa(cm)
    metrics["loss"] = loss
    return metrics


def parse_args() -> argparse.Namespace:
    """
    命令行参数
    """
    parser = argparse.ArgumentParser(description="Minimal HSI-LiDAR baseline training")
    parser.add_argument("--mat-path", type=str, default="data/raw/Houston 2013/Houston_2013_data_hl.mat")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """
    主函数，负责整体流程控制：
    1.读参数、设置随机种子
    2.选device
    3.准备数据，构造dataloader
    4.构造模型
    5.定义loss和optimizer
    6.epoch循环
    7.测试
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(
        mat_path=str(mat_path),
        patch_size=args.patch_size,
        pca_components=args.pca_components,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = BaselineFusionNet(
        hsi_in_channels=loaders.hsi_channels,
        num_classes=loaders.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Train/Val/Test batches: {len(loaders.train)}/{len(loaders.val)}/{len(loaders.test)} | "
        f"HSI channels: {loaders.hsi_channels} | Classes: {loaders.num_classes}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders.train, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_split(model, loaders.val, criterion, device, num_classes=loaders.num_classes)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_oa={val_metrics['oa']:.4f} | "
            f"val_aa={val_metrics['aa']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f}"
        )

    test_metrics = evaluate_split(model, loaders.test, criterion, device, num_classes=loaders.num_classes)
    print(
        "Test | "
        f"loss={test_metrics['loss']:.4f} | "
        f"oa={test_metrics['oa']:.4f} | "
        f"aa={test_metrics['aa']:.4f} | "
        f"kappa={test_metrics['kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
