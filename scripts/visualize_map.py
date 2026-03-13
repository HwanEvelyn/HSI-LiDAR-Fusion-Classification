from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.mat_loader import build_official_houston_split, load_houston_hl
from dataset.patch_dataset import HsiLidarPatchDataset, IndexItem
from dataset.preprocessing import pca_reduce, pca_reduce_with_mask, zscore_norm, zscore_norm_with_mask
from train import create_model, log_device_info, maybe_fallback_from_mps, resolve_device, should_pin_memory
from utils.logger import SimpleLogger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize prediction map for a trained model")
    parser.add_argument("--checkpoint", type=str, default="results/run_baseline_compare/best.pth")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--region", type=str, choices=["test", "labeled"], default="labeled")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def build_preprocessed_data(train_args: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_houston_hl(train_args["data_root"])
    train_items, test_items, _ = build_official_houston_split(data)

    train_mask = np.zeros_like(data.gt, dtype=bool)
    for item in train_items:
        train_mask[item.r, item.c] = True

    if train_args["preprocess_scope"] == "train":
        hsi, _ = zscore_norm_with_mask(data.hsi, mask=train_mask)
        lidar, _ = zscore_norm_with_mask(data.lidar, mask=train_mask)
        if 0 < int(train_args["pca_components"]) < hsi.shape[-1]:
            hsi = pca_reduce_with_mask(hsi, n_components=int(train_args["pca_components"]), mask=train_mask).x_pca
    else:
        hsi = zscore_norm(data.hsi)
        lidar = zscore_norm(data.lidar)
        if 0 < int(train_args["pca_components"]) < hsi.shape[-1]:
            hsi = pca_reduce(hsi, n_components=int(train_args["pca_components"])).x_pca

    return hsi, lidar, data.train_gt, data.test_gt


def build_items(label_map: np.ndarray) -> list[IndexItem]:
    coords = np.argwhere(label_map > 0)
    return [IndexItem(int(r), int(c), int(label_map[r, c]) - 1) for r, c in coords]


def colorize_map(pred_map: np.ndarray, num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", num_classes)
    colored = np.zeros((*pred_map.shape, 3), dtype=np.float32)
    valid = pred_map > 0
    if np.any(valid):
        colors = cmap((pred_map[valid] - 1) / max(num_classes - 1, 1))[:, :3]
        colored[valid] = colors
    return colored


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output) if args.output is not None else checkpoint_path.parent / "classification_map.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_path.parent / "visualize_log.txt")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    set_seed(int(train_args["seed"]))

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)

    hsi, lidar, train_gt, test_gt = build_preprocessed_data(train_args)
    target_map = test_gt if args.region == "test" else np.where(train_gt > 0, train_gt, test_gt)
    items = build_items(target_map)

    dataset = HsiLidarPatchDataset(
        hsi=hsi,
        lidar=lidar,
        items=items,
        patch_size=int(train_args["patch_size"]),
    )
    model = create_model(train_args, int(checkpoint["hsi_channels"]), int(checkpoint["num_classes"]))
    device = maybe_fallback_from_mps(model, device, logger)
    if device.type == "cpu":
        logger.log("Using device: cpu")

    loader = DataLoader(
        dataset,
        batch_size=int(train_args["batch_size"]),
        shuffle=False,
        num_workers=int(train_args["num_workers"]),
        pin_memory=should_pin_memory(device),
    )

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_map = np.zeros_like(target_map, dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for hsi_batch, lidar_batch, _ in loader:
            hsi_batch = hsi_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            lidar_batch = lidar_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(hsi_batch, lidar_batch)
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            batch_items = items[cursor : cursor + len(preds)]
            for item, pred in zip(batch_items, preds):
                pred_map[item.r, item.c] = int(pred)
            cursor += len(preds)

    colored = colorize_map(pred_map, int(checkpoint["num_classes"]))
    plt.imsave(output_path, colored)
    logger.log(f"Saved classification map to {output_path}")


if __name__ == "__main__":
    main()
