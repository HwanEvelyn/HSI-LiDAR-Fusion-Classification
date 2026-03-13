from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_config(run_dir: Path) -> dict:
    config_path = run_dir / "model_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_setting(run_dir: Path, metrics: dict, model_config: dict) -> str:
    name = run_dir.name
    if name == "baseline":
        return "baseline"
    fusion_layers = model_config.get("fusion_layers", "-")
    gate_status = "off" if model_config.get("disable_gate") else "on"
    if model_config.get("use_contrastive"):
        return (
            f"fusion_layers={fusion_layers}, "
            f"gate={gate_status}, "
            f"contrastive(w={model_config.get('contrastive_weight')}, tau={model_config.get('temperature')})"
        )
    return f"fusion_layers={fusion_layers}, gate={gate_status}"


def to_row(run_dir: Path, metrics: dict, model_config: dict) -> str:
    return (
        f"| {run_dir.name} | {infer_setting(run_dir, metrics, model_config)} | "
        f"{metrics.get('epoch', '-')} | "
        f"{metrics.get('val_oa', 0.0):.4f} | "
        f"{metrics.get('val_aa', 0.0):.4f} | "
        f"{metrics.get('val_kappa', 0.0):.4f} | "
        f"{metrics.get('test_oa', 0.0):.4f} | "
        f"{metrics.get('test_aa', 0.0):.4f} | "
        f"{metrics.get('test_kappa', 0.0):.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总多个实验目录的 ablation 结果")
    parser.add_argument("run_dirs", nargs="+", help="每个实验输出目录，例如 results/ablation_full/baseline")
    parser.add_argument("--output", type=str, default="", help="可选：输出 markdown 文件路径")
    args = parser.parse_args()

    rows: list[str] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        metrics = load_metrics(run_dir)
        model_config = load_model_config(run_dir)
        rows.append(to_row(run_dir, metrics, model_config))

    markdown = "\n".join(
        [
            "| Run | Setting | Best Epoch | Val OA | Val AA | Val Kappa | Test OA | Test AA | Test Kappa |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )

    print(markdown)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
