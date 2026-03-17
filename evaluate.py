from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from train import (
    build_dataloaders,
    create_model,
    evaluate_split,
    log_device_info,
    maybe_fallback_from_mps,
    resolve_device,
)
from utils.logger import SimpleLogger
from utils.metrics import per_class_accuracy
from utils.seed import set_seed

HOUSTON_2013_CLASS_NAMES = [
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Trees",
    "Soil",
    "Water",
    "Residential",
    "Commercial",
    "Road",
    "Highway",
    "Railway",
    "Parking lot 1",
    "Parking lot 2",
    "Tennis court",
    "Running track",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 checkpoint，并导出每类精度与混淆矩阵图")
    parser.add_argument("--checkpoint", type=str, default="", help="单模型评估用 checkpoint 路径")
    parser.add_argument("--baseline-checkpoint", type=str, default="", help="Baseline checkpoint 路径")
    parser.add_argument("--hct-checkpoint", type=str, default="", help="HCT-BGC checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="", help="评估结果输出目录")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--hct-label", type=str, default="HCT-BGC")
    return parser.parse_args()


def get_class_names(num_classes: int) -> list[str]:
    if num_classes == len(HOUSTON_2013_CLASS_NAMES):
        return HOUSTON_2013_CLASS_NAMES.copy()
    return [f"Class {idx}" for idx in range(num_classes)]


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def get_train_arg(train_args: dict[str, Any], key: str, default: Any) -> Any:
    return train_args.get(key, default)


def build_loaders_from_checkpoint(train_args: dict[str, Any], device: torch.device):
    patch_size = int(get_train_arg(train_args, "patch_size", 11))
    val_spatial_buffer = int(get_train_arg(train_args, "val_spatial_buffer", patch_size // 2))
    if val_spatial_buffer < 0:
        val_spatial_buffer = patch_size // 2
    return build_dataloaders(
        data_root=str(get_train_arg(train_args, "data_root", "data/raw/Houston 2013/2013_DFTC")),
        patch_size=patch_size,
        pca_components=int(get_train_arg(train_args, "pca_components", 30)),
        train_ratio=float(get_train_arg(train_args, "train_ratio", 0.6)),
        batch_size=int(get_train_arg(train_args, "batch_size", 64)),
        num_workers=int(get_train_arg(train_args, "num_workers", 4)),
        split_seed=int(get_train_arg(train_args, "split_seed", get_train_arg(train_args, "seed", 42))),
        split_mode=str(get_train_arg(train_args, "split_mode", "official")),
        preprocess_scope=str(get_train_arg(train_args, "preprocess_scope", "train")),
        device=device,
        val_ratio=float(get_train_arg(train_args, "val_ratio", 0.2)),
        val_spatial_buffer=val_spatial_buffer,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    logger: SimpleLogger,
    artifact_prefix: str,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    train_args = checkpoint["args"]
    set_seed(int(get_train_arg(train_args, "seed", 42)))

    loaders = build_loaders_from_checkpoint(train_args, device)
    model = create_model(train_args, int(checkpoint["hsi_channels"]), int(checkpoint["num_classes"]))
    device = maybe_fallback_from_mps(model, device, logger)
    if device.type == "cpu":
        logger.log("Using device: cpu")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_split(
        model,
        loaders.test,
        criterion,
        device,
        num_classes=int(checkpoint["num_classes"]),
    )
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_acc = per_class_accuracy(cm)
    class_names = get_class_names(cm.shape[0])

    save_confusion_matrix_png(
        cm=cm,
        class_names=class_names,
        output_path=output_dir / f"{artifact_prefix}confusion_matrix.png",
        title=f"{artifact_prefix.rstrip('_')} Confusion Matrix" if artifact_prefix else "Confusion Matrix",
    )
    save_per_class_accuracy_csv(
        class_names=class_names,
        class_acc=class_acc,
        cm=cm,
        output_path=output_dir / f"{artifact_prefix}per_class_accuracy.csv",
    )

    summary = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "loss": float(metrics["loss"]),
        "oa": float(metrics["oa"]),
        "aa": float(metrics["aa"]),
        "kappa": float(metrics["kappa"]),
        "checkpoint": str(checkpoint_path),
    }
    with (output_dir / f"{artifact_prefix}eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.log(
        f"{artifact_prefix or 'single'}Evaluate | "
        f"loss={summary['loss']:.4f} | "
        f"oa={summary['oa']:.4f} | "
        f"aa={summary['aa']:.4f} | "
        f"kappa={summary['kappa']:.4f}"
    )

    return {
        "summary": summary,
        "confusion_matrix": cm,
        "class_acc": class_acc,
        "class_names": class_names,
    }


def save_per_class_accuracy_csv(
    class_names: list[str],
    class_acc: np.ndarray,
    cm: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "accuracy", "support"])
        for idx, (name, acc) in enumerate(zip(class_names, class_acc.tolist())):
            writer.writerow([idx, name, f"{acc:.6f}", int(cm[idx].sum())])


def save_comparison_csv(
    class_names: list[str],
    baseline_acc: np.ndarray,
    hct_acc: np.ndarray,
    output_path: Path,
    baseline_label: str,
    hct_label: str,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", baseline_label, hct_label, "delta"])
        for idx, name in enumerate(class_names):
            delta = float(hct_acc[idx] - baseline_acc[idx])
            writer.writerow(
                [
                    idx,
                    name,
                    f"{float(baseline_acc[idx]):.6f}",
                    f"{float(hct_acc[idx]):.6f}",
                    f"{delta:.6f}",
                ]
            )


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    threshold = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.checkpoint:
        return Path(args.checkpoint).resolve().parent
    if args.baseline_checkpoint and args.hct_checkpoint:
        return Path("results/evaluation_compare")
    raise ValueError("必须提供 --output-dir，或至少提供 --checkpoint / --baseline-checkpoint / --hct-checkpoint")


def validate_args(args: argparse.Namespace) -> None:
    single_mode = bool(args.checkpoint)
    compare_mode = bool(args.baseline_checkpoint and args.hct_checkpoint)
    if single_mode == compare_mode:
        raise ValueError("请二选一：使用 --checkpoint 单模型评估，或同时使用 --baseline-checkpoint 与 --hct-checkpoint 对比评估")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_dir / "evaluate_log.txt")

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)

    if args.checkpoint:
        evaluate_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            output_dir=output_dir,
            device=device,
            logger=logger,
            artifact_prefix="",
        )
        return

    baseline_result = evaluate_checkpoint(
        checkpoint_path=Path(args.baseline_checkpoint),
        output_dir=output_dir,
        device=device,
        logger=logger,
        artifact_prefix="baseline_",
    )
    hct_result = evaluate_checkpoint(
        checkpoint_path=Path(args.hct_checkpoint),
        output_dir=output_dir,
        device=device,
        logger=logger,
        artifact_prefix="hct_bgc_",
    )

    if baseline_result["class_names"] != hct_result["class_names"]:
        raise RuntimeError("Baseline 与 HCT-BGC 的类别定义不一致，无法生成对比表")

    save_comparison_csv(
        class_names=baseline_result["class_names"],
        baseline_acc=baseline_result["class_acc"],
        hct_acc=hct_result["class_acc"],
        output_path=output_dir / "per_class_accuracy.csv",
        baseline_label=args.baseline_label,
        hct_label=args.hct_label,
    )

    logger.log(f"Saved per-class comparison CSV to {output_dir / 'per_class_accuracy.csv'}")


if __name__ == "__main__":
    main()
