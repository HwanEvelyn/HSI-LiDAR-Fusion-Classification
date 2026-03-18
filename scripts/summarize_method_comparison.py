from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总不同方法对比实验")
    parser.add_argument("run_dirs", nargs="+", help="多个实验目录，例如 results/compare_methods/hsi_only")
    parser.add_argument("--output", type=str, default="thesis/notes/method_comparison_table.md")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_name(run_dir: Path, config: dict) -> str:
    model = config.get("model", run_dir.name)
    mapping = {
        "hsi_only": "HSI-only",
        "lidar_only": "LiDAR-only",
        "baseline": "CNN baseline",
        "cnn_transformer": "CNN+Transformer",
        "hct_bgc": "Your model",
    }
    return mapping.get(model, model)


def main() -> None:
    args = parse_args()
    rows: list[str] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        metrics = load_json(run_dir / "best_metrics.json")
        config = load_json(run_dir / "model_config.json") if (run_dir / "model_config.json").exists() else {}
        name = infer_name(run_dir, config)
        rows.append(
            f"| {name} | {metrics.get('test_oa', 0.0) * 100:.2f} | {metrics.get('test_aa', 0.0) * 100:.2f} | {metrics.get('test_kappa', 0.0) * 100:.2f} |"
        )

    markdown = "\n".join(
        [
            "| Model | OA | AA | Kappa |",
            "| --- | ---: | ---: | ---: |",
            *rows,
        ]
    ) + "\n"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
