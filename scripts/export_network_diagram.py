from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 HCT-BGC-v1 网络结构草图")
    parser.add_argument("--output", type=str, default="results/paper_figures/method/hct_bgc_v1_diagram.png")
    return parser.parse_args()


def add_box(ax, xy, width, height, text, facecolor):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    add_box(ax, (0.5, 5.5), 1.6, 1.0, "HSI Patch", "#d9edf7")
    add_box(ax, (0.5, 1.5), 1.6, 1.0, "LiDAR Patch", "#d9edf7")
    add_box(ax, (2.6, 5.5), 1.8, 1.0, "CNN Encoder", "#fcf8e3")
    add_box(ax, (2.6, 1.5), 1.8, 1.0, "CNN Encoder", "#fcf8e3")
    add_box(ax, (4.9, 5.5), 2.0, 1.0, "Tokenization\n+ CLS + Pos", "#f2dede")
    add_box(ax, (4.9, 1.5), 2.0, 1.0, "Tokenization\n+ CLS + Pos", "#f2dede")
    add_box(ax, (7.4, 5.5), 1.8, 1.0, "TE", "#dff0d8")
    add_box(ax, (7.4, 1.5), 1.8, 1.0, "TE", "#dff0d8")
    add_box(ax, (9.7, 3.2), 2.3, 1.6, "Bi-CTA Fusion\nBlock × Lf", "#f5e79e")
    add_box(ax, (12.5, 3.2), 1.7, 1.6, "Gated Fuse", "#f7d6e0")
    add_box(ax, (12.5, 0.7), 1.7, 1.2, "MLP\nClassifier", "#e1d5e7")

    arrowprops = dict(arrowstyle="->", linewidth=1.6)
    ax.annotate("", xy=(2.6, 6.0), xytext=(2.1, 6.0), arrowprops=arrowprops)
    ax.annotate("", xy=(2.6, 2.0), xytext=(2.1, 2.0), arrowprops=arrowprops)
    ax.annotate("", xy=(4.9, 6.0), xytext=(4.4, 6.0), arrowprops=arrowprops)
    ax.annotate("", xy=(4.9, 2.0), xytext=(4.4, 2.0), arrowprops=arrowprops)
    ax.annotate("", xy=(7.4, 6.0), xytext=(6.9, 6.0), arrowprops=arrowprops)
    ax.annotate("", xy=(7.4, 2.0), xytext=(6.9, 2.0), arrowprops=arrowprops)
    ax.annotate("", xy=(9.7, 4.2), xytext=(9.2, 6.0), arrowprops=arrowprops)
    ax.annotate("", xy=(9.7, 3.8), xytext=(9.2, 2.0), arrowprops=arrowprops)
    ax.annotate("", xy=(12.5, 4.0), xytext=(12.0, 4.0), arrowprops=arrowprops)
    ax.annotate("", xy=(13.35, 1.9), xytext=(13.35, 3.2), arrowprops=arrowprops)

    ax.text(10.85, 5.0, "HSI cls ↔ LiDAR tokens\nResidual + FFN", ha="center", va="center", fontsize=9)
    ax.text(13.35, 5.2, "g = sigmoid(Wg[h;l])\nfused = g*h + (1-g)*l", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
