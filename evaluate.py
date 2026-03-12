from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train import build_dataloaders, create_model, evaluate_split, log_device_info, resolve_device
from utils.logger import SimpleLogger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HSI-LiDAR baseline checkpoint")
    parser.add_argument("--checkpoint", type=str, default="results/run_baseline/best.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_dir / "evaluate_log.txt")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    set_seed(int(train_args["seed"]))

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)

    loaders = build_dataloaders(
        data_root=train_args["data_root"],
        patch_size=int(train_args["patch_size"]),
        pca_components=int(train_args["pca_components"]),
        train_ratio=float(train_args["train_ratio"]),
        batch_size=int(train_args["batch_size"]),
        num_workers=int(train_args["num_workers"]),
        seed=int(train_args["seed"]),
        split_mode=str(train_args["split_mode"]),
        preprocess_scope=str(train_args["preprocess_scope"]),
        device=device,
    )

    model = create_model(train_args, int(checkpoint["hsi_channels"]), int(checkpoint["num_classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_split(
        model,
        loaders.test,
        criterion,
        device,
        num_classes=int(checkpoint["num_classes"]),
    )

    cm_path = output_dir / "confusion_matrix.npy"
    np.save(cm_path, metrics["confusion_matrix"])

    summary = {
        "epoch": int(checkpoint["epoch"]),
        "loss": float(metrics["loss"]),
        "oa": float(metrics["oa"]),
        "aa": float(metrics["aa"]),
        "kappa": float(metrics["kappa"]),
        "confusion_matrix_path": str(cm_path),
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.log(
        "Evaluate | "
        f"loss={summary['loss']:.4f} | "
        f"oa={summary['oa']:.4f} | "
        f"aa={summary['aa']:.4f} | "
        f"kappa={summary['kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
