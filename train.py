from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dataset.mat_loader import build_official_houston_split, load_houston_hl
from dataset.patch_dataset import HsiLidarPatchDataset, bulid_index
from dataset.preprocessing import pca_reduce, pca_reduce_with_mask, zscore_norm, zscore_norm_with_mask
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
    data_root: str,
    patch_size: int,
    pca_components: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    split_mode: str,
    preprocess_scope: str,
) -> SplitLoaders:
    data = load_houston_hl(data_root)

    if split_mode == "official":
        train_items, test_items, num_classes = build_official_houston_split(data)
    else:
        train_items, test_items, num_classes = bulid_index(
            data.gt,
            train_ratio=train_ratio,
            seed=seed,
        )

    train_mask = np.zeros_like(data.gt, dtype=bool)
    for item in train_items:
        train_mask[item.r, item.c] = True

    if preprocess_scope == "train":
        hsi, _ = zscore_norm_with_mask(data.hsi, mask=train_mask)
        lidar, _ = zscore_norm_with_mask(data.lidar, mask=train_mask)
        if pca_components > 0 and pca_components < hsi.shape[-1]:
            hsi = pca_reduce_with_mask(hsi, n_components=pca_components, mask=train_mask).x_pca
    else:
        hsi = zscore_norm(data.hsi)
        lidar = zscore_norm(data.lidar)
        if pca_components > 0 and pca_components < hsi.shape[-1]:
            hsi = pca_reduce(hsi, n_components=pca_components).x_pca

    train_dataset_full = HsiLidarPatchDataset(hsi, lidar, train_items, patch_size)
    test_dataset = HsiLidarPatchDataset(hsi, lidar, test_items, patch_size)

    val_size = int(len(train_dataset_full) * val_ratio)
    train_size = len(train_dataset_full) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Adjust split settings or val_ratio.")
    if val_size == 0 and len(train_dataset_full) > 1:
        val_size = 1
        train_size = len(train_dataset_full) - 1

    generator = torch.Generator().manual_seed(seed)
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
    optimizer: Adam | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
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
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(hsi, lidar)
            loss = criterion(logits, target)
            if is_train:
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
    loss, y_true, y_pred = run_epoch(model, loader, criterion, device, optimizer=None)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    metrics = oa_aa_kappa(cm)
    metrics["loss"] = loss
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HSI-LiDAR baseline training")
    parser.add_argument("--data-root", type=str, default="data/raw/Houston 2013/2013_DFTC")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--split-mode", type=str, choices=["random", "official"], default="official")
    parser.add_argument("--preprocess-scope", type=str, choices=["full", "train"], default="train")
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
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(
        data_root=str(data_root),
        patch_size=args.patch_size,
        pca_components=args.pca_components,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_mode=args.split_mode,
        preprocess_scope=args.preprocess_scope,
    )

    model = BaselineFusionNet(
        hsi_in_channels=loaders.hsi_channels,
        num_classes=loaders.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Train/Val/Test batches: {len(loaders.train)}/{len(loaders.val)}/{len(loaders.test)} | "
        f"HSI channels: {loaders.hsi_channels} | Classes: {loaders.num_classes} | "
        f"split={args.split_mode} | preprocess={args.preprocess_scope}"
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
