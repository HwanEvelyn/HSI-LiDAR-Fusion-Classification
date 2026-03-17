from __future__ import annotations

import argparse
import json
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
from train import create_model, log_device_info, maybe_fallback_from_mps, resolve_device, should_pin_memory, unpack_model_outputs
from utils.logger import SimpleLogger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出论文用 classification map 和 gate 统计")
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--hct-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/paper_figures/maps")
    parser.add_argument("--region", type=str, choices=["test", "labeled"], default="test")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--zoom-size", type=int, default=80, help="放大区域的边长")
    parser.add_argument("--zoom-row", type=int, default=-1, help="放大区域左上角行坐标，<0 表示自动选择")
    parser.add_argument("--zoom-col", type=int, default=-1, help="放大区域左上角列坐标，<0 表示自动选择")
    return parser.parse_args()


def build_preprocessed_data(train_args: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_houston_hl(train_args["data_root"])
    train_items, _, _ = build_official_houston_split(data)

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


def colorize_map(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", num_classes)
    colored = np.ones((*label_map.shape, 3), dtype=np.float32)
    valid = label_map > 0
    if np.any(valid):
        colors = cmap((label_map[valid] - 1) / max(num_classes - 1, 1))[:, :3]
        colored[valid] = colors
    return colored


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def predict_map(
    checkpoint: dict,
    target_map: np.ndarray,
    hsi: np.ndarray,
    lidar: np.ndarray,
    device: torch.device,
    logger: SimpleLogger,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    items = build_items(target_map)
    train_args = checkpoint["args"]
    dataset = HsiLidarPatchDataset(
        hsi=hsi,
        lidar=lidar,
        items=items,
        patch_size=int(train_args["patch_size"]),
    )
    model = create_model(train_args, int(checkpoint["hsi_channels"]), int(checkpoint["num_classes"]))
    device = maybe_fallback_from_mps(model, device, logger)
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
    gate_mean_map = np.full(target_map.shape, np.nan, dtype=np.float32)
    gate_class_sum = None
    gate_class_count = None
    if train_args["model"] == "hct_bgc":
        num_classes = int(checkpoint["num_classes"])
        embed_dim = int(checkpoint["model_config"].get("embed_dim", 1))
        gate_class_sum = np.zeros((num_classes, embed_dim), dtype=np.float64)
        gate_class_count = np.zeros(num_classes, dtype=np.int64)

    cursor = 0
    with torch.no_grad():
        for hsi_batch, lidar_batch, target_batch in loader:
            hsi_batch = hsi_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            lidar_batch = lidar_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = unpack_model_outputs(model(hsi_batch, lidar_batch))
            logits = outputs["logits"]
            preds = logits.argmax(dim=1).cpu().numpy() + 1
            gates = outputs.get("gate")
            gate_mean = gates.mean(dim=1).cpu().numpy() if gates is not None else None
            gate_np = gates.cpu().numpy() if gates is not None else None

            batch_items = items[cursor : cursor + len(preds)]
            for idx, (item, pred) in enumerate(zip(batch_items, preds)):
                pred_map[item.r, item.c] = int(pred)
                if gate_mean is not None:
                    gate_mean_map[item.r, item.c] = float(gate_mean[idx])
                    cls = int(target_batch[idx].item())
                    gate_class_sum[cls] += gate_np[idx]
                    gate_class_count[cls] += 1
            cursor += len(preds)

    gate_class_pref = None
    if gate_class_sum is not None and gate_class_count is not None:
        gate_class_pref = np.divide(
            gate_class_sum,
            np.maximum(gate_class_count[:, None], 1),
            out=np.zeros_like(gate_class_sum),
            where=gate_class_count[:, None] > 0,
        )
    return pred_map, gate_mean_map if gate_class_pref is not None else None, gate_class_pref


def save_gate_stats(
    output_dir: Path,
    gate_mean_map: np.ndarray | None,
    gate_class_pref: np.ndarray | None,
) -> None:
    if gate_mean_map is None or gate_class_pref is None:
        return

    valid = np.isfinite(gate_mean_map)
    payload = {
        "global_gate_mean": float(np.nanmean(gate_mean_map)),
        "global_gate_std": float(np.nanstd(gate_mean_map)),
        "per_class_gate_mean": [
            {
                "class_index": class_idx,
                "mean_gate": float(class_pref.mean()),
                "gate_dim_values": [float(value) for value in class_pref.tolist()],
            }
            for class_idx, class_pref in enumerate(gate_class_pref)
        ],
    }
    with (output_dir / "gate_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.imshow(gate_mean_map, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Mean Gate Weight")
    plt.title("HCT-BGC Gate Mean Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "gate_mean_map.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    class_means = gate_class_pref.mean(axis=1)
    plt.bar(np.arange(len(class_means)), class_means)
    plt.xlabel("Class Index")
    plt.ylabel("Mean Gate Weight")
    plt.title("Per-Class Gate Preference")
    plt.tight_layout()
    plt.savefig(output_dir / "gate_per_class.png", dpi=300)
    plt.close()


def save_panel_figure(output_path: Path, gt_map: np.ndarray, baseline_map: np.ndarray, hct_map: np.ndarray, num_classes: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("GT", gt_map),
        ("Baseline-CNN", baseline_map),
        ("HCT-BGC-v1", hct_map),
    ]
    for ax, (title, panel_map) in zip(axes, panels):
        ax.imshow(colorize_map(panel_map, num_classes))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def build_zoom_triptych(gt_crop: np.ndarray, baseline_crop: np.ndarray, hct_crop: np.ndarray, num_classes: int) -> np.ndarray:
    gap = 3
    gt_rgb = colorize_map(gt_crop, num_classes)
    baseline_rgb = colorize_map(baseline_crop, num_classes)
    hct_rgb = colorize_map(hct_crop, num_classes)
    height, width, _ = gt_rgb.shape
    canvas = np.ones((height, width * 3 + gap * 2, 3), dtype=np.float32)
    canvas[:, 0:width] = gt_rgb
    canvas[:, width + gap : width * 2 + gap] = baseline_rgb
    canvas[:, width * 2 + gap * 2 : width * 3 + gap * 2] = hct_rgb
    return canvas


def choose_zoom_region(
    gt_map: np.ndarray,
    baseline_map: np.ndarray,
    hct_map: np.ndarray,
    zoom_size: int,
    manual_row: int,
    manual_col: int,
) -> tuple[int, int, int]:
    height, width = gt_map.shape
    zoom_size = min(zoom_size, height, width)
    if manual_row >= 0 and manual_col >= 0:
        row0 = min(max(manual_row, 0), max(height - zoom_size, 0))
        col0 = min(max(manual_col, 0), max(width - zoom_size, 0))
        return row0, col0, zoom_size

    # 优先在更复杂、常见易混淆类别中找 baseline 与 HCT 差异大的区域。
    focus_classes = {6, 7, 8, 9, 10, 11, 12}  # Residential, Commercial, Road, Highway, Railway, Parking1, Parking2 (1-based)
    valid = gt_map > 0
    focus_mask = np.isin(gt_map, list(focus_classes)) & valid
    disagreement = ((baseline_map != gt_map) | (hct_map != gt_map) | (baseline_map != hct_map)) & valid
    score_map = disagreement.astype(np.float32) + 0.5 * focus_mask.astype(np.float32)

    stride = max(zoom_size // 4, 8)
    best_score = -1.0
    best_coord = (0, 0)
    for row0 in range(0, max(height - zoom_size + 1, 1), stride):
        for col0 in range(0, max(width - zoom_size + 1, 1), stride):
            row1 = row0 + zoom_size
            col1 = col0 + zoom_size
            crop_score = float(score_map[row0:row1, col0:col1].sum())
            if crop_score > best_score:
                best_score = crop_score
                best_coord = (row0, col0)

    return best_coord[0], best_coord[1], zoom_size


def save_comparison_figure(
    output_path: Path,
    gt_map: np.ndarray,
    baseline_map: np.ndarray,
    hct_map: np.ndarray,
    num_classes: int,
    zoom_row: int,
    zoom_col: int,
    zoom_size: int,
) -> None:
    row0, col0, zoom_size = choose_zoom_region(
        gt_map=gt_map,
        baseline_map=baseline_map,
        hct_map=hct_map,
        zoom_size=zoom_size,
        manual_row=zoom_row,
        manual_col=zoom_col,
    )
    row1 = row0 + zoom_size
    col1 = col0 + zoom_size

    gt_crop = gt_map[row0:row1, col0:col1]
    baseline_crop = baseline_map[row0:row1, col0:col1]
    hct_crop = hct_map[row0:row1, col0:col1]
    zoom_canvas = build_zoom_triptych(gt_crop, baseline_crop, hct_crop, num_classes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("GT", gt_map),
        ("Baseline", baseline_map),
        ("HCT-BGC", hct_map),
    ]
    for ax, (title, panel_map) in zip(axes.flat[:3], panels):
        ax.imshow(colorize_map(panel_map, num_classes))
        rect = plt.Rectangle((col0, row0), zoom_size, zoom_size, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.set_title(title)
        ax.axis("off")

    axes[1, 1].imshow(zoom_canvas)
    axes[1, 1].set_title("Zoom Region: GT | Baseline | HCT-BGC")
    axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_dir / "visualize_log.txt")

    baseline_ckpt = load_checkpoint(Path(args.baseline_checkpoint))
    hct_ckpt = load_checkpoint(Path(args.hct_checkpoint))
    train_args = hct_ckpt["args"]
    set_seed(int(train_args["seed"]))

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)

    hsi, lidar, train_gt, test_gt = build_preprocessed_data(train_args)
    target_map = test_gt if args.region == "test" else np.where(train_gt > 0, train_gt, test_gt)

    baseline_pred_map, _, _ = predict_map(baseline_ckpt, target_map, hsi, lidar, device, logger)
    hct_pred_map, gate_mean_map, gate_class_pref = predict_map(hct_ckpt, target_map, hsi, lidar, device, logger)

    save_panel_figure(output_dir / "classification_map_panel.png", target_map, baseline_pred_map, hct_pred_map, int(hct_ckpt["num_classes"]))
    save_comparison_figure(
        output_path=output_dir / "classification_comparison.png",
        gt_map=target_map,
        baseline_map=baseline_pred_map,
        hct_map=hct_pred_map,
        num_classes=int(hct_ckpt["num_classes"]),
        zoom_row=args.zoom_row,
        zoom_col=args.zoom_col,
        zoom_size=args.zoom_size,
    )
    plt.imsave(output_dir / "classification_map_gt.png", colorize_map(target_map, int(hct_ckpt["num_classes"])))
    plt.imsave(output_dir / "classification_map_baseline.png", colorize_map(baseline_pred_map, int(hct_ckpt["num_classes"])))
    plt.imsave(output_dir / "classification_map_hct_bgc_v1.png", colorize_map(hct_pred_map, int(hct_ckpt["num_classes"])))
    save_gate_stats(output_dir, gate_mean_map, gate_class_pref)
    logger.log(f"Saved paper figure assets to {output_dir}")


if __name__ == "__main__":
    main()
