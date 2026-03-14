from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
    r"train_ce=(?P<train_ce>[0-9.]+)\s+\|\s+"
    r"train_contrastive=(?P<train_contrastive>[0-9.]+)\s+\|\s+"
    r"train_total=(?P<train_total>[0-9.]+)\s+\|\s+"
    r"val_loss=(?P<val_loss>[0-9.]+)\s+\|\s+"
    r"val_oa=(?P<val_oa>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 train_log.txt 生成训练曲线")
    parser.add_argument("log_files", nargs="+", help="一个或多个 train_log.txt")
    parser.add_argument("--output-dir", type=str, default="results/paper_figures/curves")
    return parser.parse_args()


def parse_log(log_path: Path) -> dict[str, list[float]]:
    all_points: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        all_points.append(
            {
                "epoch": int(match.group("epoch")),
                "train_ce": float(match.group("train_ce")),
                "train_contrastive": float(match.group("train_contrastive")),
                "train_total": float(match.group("train_total")),
                "val_oa": float(match.group("val_oa")),
            }
        )

    if not all_points:
        return {"epoch": [], "train_ce": [], "train_contrastive": [], "train_total": [], "val_oa": []}

    runs: list[list[dict[str, float]]] = []
    current_run: list[dict[str, float]] = []
    previous_epoch = -1
    for point in all_points:
        if current_run and point["epoch"] <= previous_epoch:
            runs.append(current_run)
            current_run = []
        current_run.append(point)
        previous_epoch = point["epoch"]
    if current_run:
        runs.append(current_run)

    last_run = runs[-1]
    curves = {key: [point[key] for point in last_run] for key in last_run[0].keys()}
    return curves


def plot_single_curve(curves: dict[str, list[float]], name: str, output_dir: Path) -> None:
    epochs = curves["epoch"]
    if not epochs:
        raise ValueError(f"{name} 没有解析到任何 epoch 曲线")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    same_ce_total = all(abs(a - b) < 1e-10 for a, b in zip(curves["train_total"], curves["train_ce"]))
    if same_ce_total:
        axes[0].plot(epochs, curves["train_total"], label="train_total (= train_ce)", linewidth=2, color="tab:blue")
    else:
        axes[0].plot(epochs, curves["train_total"], label="train_total", linewidth=2, color="tab:blue")
        axes[0].plot(epochs, curves["train_ce"], label="train_ce", linewidth=2, linestyle="--", color="tab:orange")
    if any(value > 0 for value in curves["train_contrastive"]):
        axes[0].plot(epochs, curves["train_contrastive"], label="train_contrastive", linewidth=2, color="tab:red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{name} Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, curves["val_oa"], color="tab:green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val OA")
    axes[1].set_title(f"{name} Validation OA")

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_training_curves.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for log_file in args.log_files:
        log_path = Path(log_file)
        curves = parse_log(log_path)
        plot_single_curve(curves, log_path.parent.name, output_dir)


if __name__ == "__main__":
    main()
