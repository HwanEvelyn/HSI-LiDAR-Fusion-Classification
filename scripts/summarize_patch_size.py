from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 patch size 实验并生成表格与 OA 曲线图")
    parser.add_argument("summary_files", nargs="+", help="多个 results/repeats/*/summary.json 文件")
    parser.add_argument("--output-dir", type=str, default="results/paper_figures/patch_size")
    return parser.parse_args()


def load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_patch_size(summary_path: Path) -> int:
    repeat_config = summary_path.parent / "repeat_config.json"
    if not repeat_config.exists():
        raise FileNotFoundError(f"缺少 repeat_config.json: {repeat_config}")
    with repeat_config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    patch_size = config["train_args"].get("patch_size")
    if patch_size is None:
        raise ValueError(f"{repeat_config} 中没有 patch_size")
    return int(patch_size)


def format_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value * 100:.2f} ± {std_value * 100:.2f}"


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "| Patch | OA | AA | Kappa |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['patch_size']} | "
            f"{format_pm(row['test_oa_mean'], row['test_oa_std'])} | "
            f"{format_pm(row['test_aa_mean'], row['test_aa_std'])} | "
            f"{format_pm(row['test_kappa_mean'], row['test_kappa_std'])} |"
        )
    return "\n".join(lines) + "\n"


def plot_patch_vs_oa(rows: list[dict], output_path: Path) -> None:
    patch_sizes = [row["patch_size"] for row in rows]
    oa_means = [row["test_oa_mean"] * 100 for row in rows]
    oa_stds = [row["test_oa_std"] * 100 for row in rows]

    plt.figure(figsize=(6, 4))
    plt.errorbar(patch_sizes, oa_means, yerr=oa_stds, marker="o", linewidth=2, capsize=4)
    plt.xlabel("Patch Size")
    plt.ylabel("OA (%)")
    plt.title("Patch Size vs OA")
    plt.xticks(patch_sizes)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for summary_file in args.summary_files:
        summary_path = Path(summary_file)
        summary = load_summary(summary_path)
        patch_size = infer_patch_size(summary_path)
        aggregate = summary["aggregate"]
        rows.append(
            {
                "patch_size": patch_size,
                "test_oa_mean": float(aggregate["test_oa"]["mean"]),
                "test_oa_std": float(aggregate["test_oa"]["std"]),
                "test_aa_mean": float(aggregate["test_aa"]["mean"]),
                "test_aa_std": float(aggregate["test_aa"]["std"]),
                "test_kappa_mean": float(aggregate["test_kappa"]["mean"]),
                "test_kappa_std": float(aggregate["test_kappa"]["std"]),
            }
        )

    rows.sort(key=lambda row: row["patch_size"])
    markdown = build_markdown(rows)
    (output_dir / "patch_size_table.md").write_text(markdown, encoding="utf-8")
    plot_patch_vs_oa(rows, output_dir / "patch_size_vs_OA.png")

    print(markdown)
    print(f"Saved table to {output_dir / 'patch_size_table.md'}")
    print(f"Saved plot to {output_dir / 'patch_size_vs_OA.png'}")


if __name__ == "__main__":
    main()
