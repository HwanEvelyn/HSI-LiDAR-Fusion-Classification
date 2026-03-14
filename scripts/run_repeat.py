from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


SUMMARY_KEYS = [
    "val_oa",
    "val_aa",
    "val_kappa",
    "test_oa",
    "test_aa",
    "test_kappa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重复运行实验并统计均值/标准差")
    parser.add_argument("--config", type=str, required=True, help="实验配置 YAML 文件")
    parser.add_argument("--output-root", type=str, default="", help="可选：覆盖配置中的输出目录")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="重复运行的随机种子列表")
    parser.add_argument("--python", type=str, default=sys.executable, help="用于调用 train.py 的 Python 解释器")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不真正执行")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是字典")
    if "name" not in config or "train_args" not in config:
        raise ValueError("配置文件必须包含 name 和 train_args")
    if not isinstance(config["train_args"], dict):
        raise ValueError("train_args 必须是字典")
    return config


def build_command(python_executable: str, train_args: dict[str, Any], seed: int, output_dir: Path) -> list[str]:
    command = [python_executable, "train.py"]
    merged_args = dict(train_args)
    merged_args["seed"] = seed
    merged_args["save_dir"] = str(output_dir)

    for key, value in merged_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        command.extend([flag, str(value)])
    return command


def load_best_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少结果文件: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_runs": len(metrics_list),
        "runs": metrics_list,
        "aggregate": {},
    }
    for key in SUMMARY_KEYS:
        values = [float(metrics[key]) for metrics in metrics_list]
        summary["aggregate"][key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    run_name = str(config["name"])
    output_root = Path(args.output_root) if args.output_root else Path(str(config.get("output_root", "results/repeats"))) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "repeat_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "name": run_name,
                "seeds": args.seeds,
                "split_seed": config["train_args"].get("split_seed", 42),
                "train_args": config["train_args"],
            },
            f,
            indent=2,
        )

    collected_metrics: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = output_root / f"seed{seed}"
        command = build_command(args.python, dict(config["train_args"]), seed, run_dir)
        print("Running:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
            metrics = load_best_metrics(run_dir)
            metrics["seed"] = seed
            collected_metrics.append(metrics)

    if args.dry_run:
        return

    summary = summarize_metrics(collected_metrics)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {output_root / 'summary.json'}")
    for key in SUMMARY_KEYS:
        stats = summary["aggregate"][key]
        print(f"{key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


if __name__ == "__main__":
    main()
