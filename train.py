from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.mat_loader import build_official_houston_split, load_dataset
from dataset.patch_dataset import (
    HsiLidarPatchDataset,
    IndexItem,
    build_index_fewshot,
    build_index_three_way,
    split_items_spatial_holdout,
)
from dataset.preprocessing import pca_reduce, pca_reduce_with_mask, zscore_norm, zscore_norm_with_mask
from models.baseline_cnn import BaselineFusionNet
from models.comparison_models import CnnTransformerNoFusion, HsiOnlyNet, LidarOnlyNet
from models.hct_bgc import HCT_BGC
from utils.logger import SimpleLogger
from utils.metrics import confusion_matrix, oa_aa_kappa, per_class_accuracy
from utils.seed import set_seed


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    hsi_channels: int
    lidar_channels: int
    num_classes: int
    split_sizes: Dict[str, int]


@dataclass
class EpochStats:
    total_loss: float
    ce_loss: float
    contrastive_loss: float
    aux_loss: float
    y_true: np.ndarray
    y_pred: np.ndarray


def create_model(args: argparse.Namespace | dict, hsi_channels: int, lidar_channels: int, num_classes: int) -> nn.Module:
    model_name = args["model"] if isinstance(args, dict) else args.model
    fusion_layers = args.get("fusion_layers", 1) if isinstance(args, dict) else args.fusion_layers
    if model_name == "baseline":
        return BaselineFusionNet(
            hsi_in_channels=hsi_channels,
            lidar_in_channels=lidar_channels,
            num_classes=num_classes,
        )
    if model_name == "hsi_only":
        return HsiOnlyNet(
            hsi_in_channels=hsi_channels,
            num_classes=num_classes,
        )
    if model_name == "lidar_only":
        return LidarOnlyNet(
            lidar_in_channels=lidar_channels,
            num_classes=num_classes,
        )
    if model_name == "cnn_transformer":
        return CnnTransformerNoFusion(
            hsi_in_channels=hsi_channels,
            lidar_in_channels=lidar_channels,
            num_classes=num_classes,
            embed_dim=args.get("embed_dim", 128) if isinstance(args, dict) else args.embed_dim,
            num_heads=args.get("num_heads", 4) if isinstance(args, dict) else args.num_heads,
            num_layers=args.get("num_layers", 2) if isinstance(args, dict) else args.num_layers,
            dropout=args.get("dropout", 0.1) if isinstance(args, dict) else args.dropout,
            patch_size=args.get("patch_size", 11) if isinstance(args, dict) else args.patch_size,
        )
    if model_name == "hct_bgc":
        return HCT_BGC(
            hsi_in_channels=hsi_channels,
            lidar_in_channels=lidar_channels,
            num_classes=num_classes,
            embed_dim=args.get("embed_dim", 128) if isinstance(args, dict) else args.embed_dim,
            num_heads=args.get("num_heads", 4) if isinstance(args, dict) else args.num_heads,
            num_layers=args.get("num_layers", 2) if isinstance(args, dict) else args.num_layers,
            fusion_layers=fusion_layers,
            dropout=args.get("dropout", 0.1) if isinstance(args, dict) else args.dropout,
            patch_size=args.get("patch_size", 11) if isinstance(args, dict) else args.patch_size,
            context_patch_size=(
                args.get("context_patch_size", 0) if isinstance(args, dict) else args.context_patch_size
            ) or None,
            context_token_size=args.get("context_token_size", 0) if isinstance(args, dict) else args.context_token_size,
            scale_fusion_mode=args.get("scale_fusion_mode", "residual") if isinstance(args, dict) else args.scale_fusion_mode,
            disable_gate=args.get("disable_gate", False) if isinstance(args, dict) else args.disable_gate,
            encoder_variant=args.get("encoder_variant", "hetero") if isinstance(args, dict) else args.encoder_variant,
            use_conservative_fusion=args.get("use_conservative_fusion", False) if isinstance(args, dict) else args.use_conservative_fusion,
            use_aux_heads=args.get("use_aux_heads", False) if isinstance(args, dict) else args.use_aux_heads,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def unpack_model_outputs(outputs: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError("Model output dictionary must contain a 'logits' key")
        return outputs
    return {"logits": outputs}


def collect_model_config(
    model: nn.Module,
    args: argparse.Namespace,
    hsi_channels: int,
    lidar_channels: int,
    num_classes: int,
) -> Dict[str, object]:
    extraction_patch_size = max(args.patch_size, args.context_patch_size)
    effective_val_spatial_buffer = args.val_spatial_buffer if args.val_spatial_buffer >= 0 else extraction_patch_size // 2
    config: Dict[str, object] = {
        "model": args.model,
        "hsi_in_channels": hsi_channels,
        "lidar_in_channels": lidar_channels,
        "num_classes": num_classes,
        "use_contrastive": args.use_contrastive,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "split_seed": args.split_seed,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "context_patch_size": args.context_patch_size,
        "context_token_size": args.context_token_size,
        "scale_fusion_mode": args.scale_fusion_mode,
        "val_ratio": args.val_ratio,
        "selection_metric": args.selection_metric,
        "preprocess_scope": args.preprocess_scope,
        "split_mode": args.split_mode,
        "val_spatial_buffer": effective_val_spatial_buffer,
        "train_augment": args.train_augment,
        "label_smoothing": args.label_smoothing,
        "early_stopping_patience": args.early_stopping_patience,
        "encoder_variant": args.encoder_variant,
        "use_conservative_fusion": args.use_conservative_fusion,
        "use_aux_heads": args.use_aux_heads,
        "aux_weight": args.aux_weight,
    }
    if hasattr(model, "get_config"):
        config.update(model.get_config())
    return config


def info_nce_loss(h_cls: torch.Tensor, l_cls: torch.Tensor, temperature: float) -> torch.Tensor:
    if h_cls.shape != l_cls.shape:
        raise ValueError("InfoNCE 期望 h_cls 和 l_cls 形状一致")
    if h_cls.dim() != 2:
        raise ValueError("InfoNCE 期望输入形状为 (B, D)")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    h_norm = F.normalize(h_cls, dim=1)
    l_norm = F.normalize(l_cls, dim=1)
    logits = (h_norm @ l_norm.T) / temperature
    labels = torch.arange(h_cls.size(0), device=h_cls.device)
    loss_h_to_l = F.cross_entropy(logits, labels)
    loss_l_to_h = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_h_to_l + loss_l_to_h)


def mps_is_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def should_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def build_dataloaders(
    data_root: str,
    patch_size: int,
    pca_components: int,
    train_ratio: float,
    batch_size: int,
    num_workers: int,
    split_seed: int,
    split_mode: str,
    preprocess_scope: str,
    device: torch.device,
    val_ratio: float,
    val_spatial_buffer: int,
    train_augment: str,
    train_per_class: int,
    val_per_class: int,
) -> SplitLoaders:
    data = load_dataset(data_root)

    if split_mode == "fewshot":
        train_items, val_items, test_items, num_classes = build_index_fewshot(
            data.gt,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            seed=split_seed,
        )
    elif split_mode == "official" and data.dataset_name == "houston":
        official_train_items, test_items, num_classes = build_official_houston_split(data)
        train_items, val_items = split_items_spatial_holdout(
            official_train_items,
            holdout_ratio=val_ratio,
            buffer_radius=val_spatial_buffer,
            seed=split_seed,
        )
    elif split_mode == "official":
        coords = np.argwhere(data.gt > 0)
        all_items = [IndexItem(int(r), int(c), int(data.gt[r, c]) - 1) for r, c in coords]
        train_val_items, test_items = split_items_spatial_holdout(
            all_items,
            holdout_ratio=1.0 - train_ratio - val_ratio,
            buffer_radius=val_spatial_buffer,
            seed=split_seed,
        )
        inner_val_ratio = val_ratio / max(train_ratio + val_ratio, 1e-8)
        train_items, val_items = split_items_spatial_holdout(
            train_val_items,
            holdout_ratio=inner_val_ratio,
            buffer_radius=val_spatial_buffer,
            seed=split_seed + 1,
        )
        num_classes = int(data.gt.max())
    else:
        train_items, val_items, test_items, num_classes = build_index_three_way(
            data.gt,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=split_seed,
        )

    if len(train_items) == 0 or len(val_items) == 0 or len(test_items) == 0:
        raise RuntimeError(
            f"数据划分失败，得到的样本数为 train={len(train_items)}, val={len(val_items)}, test={len(test_items)}"
        )

    train_mask = np.zeros_like(data.gt, dtype=bool)
    for item in train_items:
        train_mask[item.r, item.c] = True

    if preprocess_scope == "train":
        hsi, _ = zscore_norm_with_mask(data.hsi, mask=train_mask)
        lidar, _ = zscore_norm_with_mask(data.lidar, mask=train_mask)
        if pca_components > 0 and pca_components < hsi.shape[-1]:
            hsi = pca_reduce_with_mask(hsi, n_components=pca_components, mask=train_mask).x_pca
    else:
        hsi = zscore_norm(data.hsi)
        lidar = zscore_norm(data.lidar)
        if pca_components > 0 and pca_components < hsi.shape[-1]:
            hsi = pca_reduce(hsi, n_components=pca_components).x_pca

    train_dataset_full = HsiLidarPatchDataset(hsi, lidar, train_items, patch_size, augment_mode=train_augment)
    val_dataset = HsiLidarPatchDataset(hsi, lidar, val_items, patch_size)
    test_dataset = HsiLidarPatchDataset(hsi, lidar, test_items, patch_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": should_pin_memory(device),
    }
    train_loader = DataLoader(train_dataset_full, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return SplitLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        hsi_channels=hsi.shape[-1],
        lidar_channels=1 if lidar.ndim == 2 else lidar.shape[-1],
        num_classes=num_classes,
        split_sizes={
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_contrastive: bool,
    contrastive_weight: float,
    temperature: float,
    use_aux_heads: bool,
    aux_weight: float,
    optimizer: Adam | None = None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    aux_loss_sum = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    for hsi, lidar, target in loader:
        hsi = hsi.to(device=device, dtype=torch.float32, non_blocking=True)
        lidar = lidar.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.long, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = unpack_model_outputs(model(hsi, lidar))
            logits = outputs["logits"]
            ce_loss = criterion(logits, target)
            contrastive_loss = torch.zeros((), device=device)
            aux_loss = torch.zeros((), device=device)
            if use_contrastive:
                if "h_cls" not in outputs or "l_cls" not in outputs:
                    raise KeyError("启用对比损失时，模型输出必须包含 'h_cls' 和 'l_cls'")
                contrastive_loss = info_nce_loss(outputs["h_cls"], outputs["l_cls"], temperature=temperature)
            if use_aux_heads:
                if "h_logits" not in outputs or "l_logits" not in outputs:
                    raise KeyError("启用辅助头时，模型输出必须包含 'h_logits' 和 'l_logits'")
                aux_loss = 0.5 * (criterion(outputs["h_logits"], target) + criterion(outputs["l_logits"], target))
            loss = ce_loss + contrastive_weight * contrastive_loss + aux_weight * aux_loss
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = target.size(0)
        total_loss_sum += float(loss.item()) * batch_size
        ce_loss_sum += float(ce_loss.item()) * batch_size
        contrastive_loss_sum += float(contrastive_loss.item()) * batch_size
        aux_loss_sum += float(aux_loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    avg_total = total_loss_sum / max(total_samples, 1)
    avg_ce = ce_loss_sum / max(total_samples, 1)
    avg_contrastive = contrastive_loss_sum / max(total_samples, 1)
    avg_aux = aux_loss_sum / max(total_samples, 1)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return EpochStats(
        total_loss=avg_total,
        ce_loss=avg_ce,
        contrastive_loss=avg_contrastive,
        aux_loss=avg_aux,
        y_true=y_true,
        y_pred=y_pred,
    )

# 评估
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    epoch_stats = run_epoch(
        model,
        loader,
        criterion,
        device,
        use_contrastive=False,
        contrastive_weight=0.0,
        temperature=1.0,
        use_aux_heads=False,
        aux_weight=0.0,
        optimizer=None,
    )
    cm = confusion_matrix(epoch_stats.y_true, epoch_stats.y_pred, num_classes=num_classes)
    metrics = oa_aa_kappa(cm)
    metrics["loss"] = epoch_stats.ce_loss
    metrics["confusion_matrix"] = cm
    return metrics


def get_selection_score(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric not in {"val_oa", "val_kappa"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    metric_name = selection_metric.replace("val_", "")
    return float(metrics[metric_name])


def save_final_eval_artifacts(output_dir: Path, metrics: Dict[str, float]) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_acc = per_class_accuracy(cm)

    with (output_dir / "test_confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    with (output_dir / "test_confusion_matrix.csv").open("w", encoding="utf-8") as f:
        for row in cm.tolist():
            f.write(",".join(str(value) for value in row) + "\n")

    per_class_payload = [
        {
            "class_index": idx,
            "accuracy": float(acc),
            "support": int(cm[idx].sum()),
        }
        for idx, acc in enumerate(class_acc.tolist())
    ]
    with (output_dir / "test_per_class_accuracy.json").open("w", encoding="utf-8") as f:
        json.dump(per_class_payload, f, indent=2)

    with (output_dir / "test_per_class_accuracy.csv").open("w", encoding="utf-8") as f:
        f.write("class_index,accuracy,support\n")
        for item in per_class_payload:
            f.write(f"{item['class_index']},{item['accuracy']:.6f},{item['support']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HSI-LiDAR baseline training")
    parser.add_argument("--data-root", type=str, default="data/raw/Houston 2013/2013_DFTC")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "hsi_only", "lidar_only", "cnn_transformer", "hct_bgc"],
        default="baseline",
    )
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--context-patch-size", type=int, default=0)
    parser.add_argument("--context-token-size", type=int, default=0)
    parser.add_argument("--scale-fusion-mode", type=str, choices=["residual", "gated", "average"], default="residual")
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--encoder-variant", type=str, choices=["simple", "hetero", "light_hetero"], default="hetero")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--use-conservative-fusion", action="store_true")
    parser.add_argument("--use-aux-heads", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, choices=["random", "official", "fewshot"], default="official")
    parser.add_argument("--preprocess-scope", type=str, choices=["full", "train"], default="train")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-per-class", type=int, default=0)
    parser.add_argument("--val-per-class", type=int, default=5)
    parser.add_argument("--val-spatial-buffer", type=int, default=-1)
    parser.add_argument("--selection-metric", type=str, choices=["val_oa", "val_kappa"], default="val_oa")
    parser.add_argument("--train-augment", type=str, choices=["none", "d4"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", type=str, default="results/run_baseline")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and check your GPU driver."
            )
        return torch.device("cuda")
    if device_arg == "mps":
        if not mps_is_available():
            raise RuntimeError(
                "MPS was requested with --device mps, but torch.backends.mps.is_available() is False. "
                "Install a recent macOS-compatible PyTorch build and run on Apple Silicon."
            )
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device_info(logger: SimpleLogger, device: torch.device) -> None:
    logger.log(
        f"PyTorch version: {torch.__version__} | CUDA build: {torch.version.cuda} | "
        f"MPS available: {mps_is_available()}"
    )
    if device.type == "cuda":
        logger.log(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True


def maybe_fallback_from_mps(model: nn.Module, device: torch.device, logger: SimpleLogger) -> torch.device:
    if device.type != "mps":
        return device
    module_types = sorted({type(module).__name__ for module in model.modules()})
    logger.log(
        "Warning: MPS is available, but this project's models are not stable on the current "
        "PyTorch + Apple Silicon runtime and may abort at the native runtime level. "
        f"Detected model modules: {', '.join(module_types)}. Falling back to CPU."
    )
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.use_contrastive and args.model != "hct_bgc":
        raise ValueError("--use-contrastive 当前仅支持 --model hct_bgc")
    if args.split_mode == "fewshot" and args.train_per_class <= 0:
        raise ValueError("--split-mode fewshot 时，--train-per-class 必须 > 0")
    if args.context_patch_size < 0:
        raise ValueError("--context-patch-size 必须 >= 0")
    if args.context_patch_size and args.context_patch_size < args.patch_size:
        raise ValueError("--context-patch-size 必须 >= --patch-size")
    if args.context_token_size < 0:
        raise ValueError("--context-token-size 必须 >= 0")
    if args.context_token_size and args.context_patch_size == 0:
        raise ValueError("--context-token-size 仅在设置 --context-patch-size 时有效")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_dir / "train_log.txt")

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)
    if args.preprocess_scope != "train":
        logger.log("Warning: preprocess_scope=full 会使用全图统计量，存在数据泄漏风险；论文实验建议使用 preprocess_scope=train。")

    extraction_patch_size = max(args.patch_size, args.context_patch_size)
    loaders = build_dataloaders(
        data_root=str(data_root),
        patch_size=extraction_patch_size,
        pca_components=args.pca_components,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        preprocess_scope=args.preprocess_scope,
        device=device,
        val_ratio=args.val_ratio,
        val_spatial_buffer=args.val_spatial_buffer if args.val_spatial_buffer >= 0 else extraction_patch_size // 2,
        train_augment=args.train_augment,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )

    model = create_model(args, loaders.hsi_channels, loaders.lidar_channels, loaders.num_classes)
    device = maybe_fallback_from_mps(model, device, logger)
    if device.type == "cpu":
        logger.log("Using device: cpu")
    model = model.to(device)
    model_config = collect_model_config(model, args, loaders.hsi_channels, loaders.lidar_channels, loaders.num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.log(
        f"Train/Val/Test batches: {len(loaders.train)}/{len(loaders.val)}/{len(loaders.test)} | "
        f"Train/Val/Test samples: {loaders.split_sizes['train']}/{loaders.split_sizes['val']}/{loaders.split_sizes['test']} | "
        f"HSI channels: {loaders.hsi_channels} | Classes: {loaders.num_classes} | "
        f"split={args.split_mode} | preprocess={args.preprocess_scope} | "
        f"model={args.model} | select_by={args.selection_metric}"
    )
    logger.log(f"Model config: {json.dumps(model_config, sort_keys=True)}")
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    best_score = float("-inf")
    best_metrics: Dict[str, float] | None = None
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            loaders.train,
            criterion,
            device,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            use_aux_heads=args.use_aux_heads,
            aux_weight=args.aux_weight,
            optimizer=optimizer,
        )
        val_metrics = evaluate_split(model, loaders.val, criterion, device, num_classes=loaders.num_classes)
        logger.log(
            f"Epoch {epoch:03d} | "
            f"train_ce={train_stats.ce_loss:.4f} | "
            f"train_contrastive={train_stats.contrastive_loss:.4f} | "
            f"train_aux={train_stats.aux_loss:.4f} | "
            f"train_total={train_stats.total_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_oa={val_metrics['oa']:.4f} | "
            f"val_aa={val_metrics['aa']:.4f} | "
            f"val_kappa={val_metrics['kappa']:.4f}"
        )

        current_score = get_selection_score(val_metrics, args.selection_metric)
        if current_score > best_score:
            best_score = current_score
            epochs_without_improve = 0
            best_metrics = {
                "epoch": epoch,
                "train_ce": float(train_stats.ce_loss),
                "train_contrastive": float(train_stats.contrastive_loss),
                "train_aux": float(train_stats.aux_loss),
                "train_total": float(train_stats.total_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_oa": float(val_metrics["oa"]),
                "val_aa": float(val_metrics["aa"]),
                "val_kappa": float(val_metrics["kappa"]),
                "selection_metric": args.selection_metric,
                "selection_score": current_score,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "model_config": model_config,
                    "hsi_channels": loaders.hsi_channels,
                    "lidar_channels": loaders.lidar_channels,
                    "num_classes": loaders.num_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metrics": best_metrics,
                },
                output_dir / "best.pth",
            )
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)
        else:
            epochs_without_improve += 1

        if args.early_stopping_patience > 0 and epochs_without_improve >= args.early_stopping_patience:
            logger.log(
                f"Early stopping triggered at epoch {epoch:03d} | "
                f"no improvement for {epochs_without_improve} epochs on {args.selection_metric}"
            )
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without producing best metrics")

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_split(model, loaders.test, criterion, device, num_classes=loaders.num_classes)
    save_final_eval_artifacts(output_dir, test_metrics)
    best_metrics.update(
        {
            "test_loss": float(test_metrics["loss"]),
            "test_oa": float(test_metrics["oa"]),
            "test_aa": float(test_metrics["aa"]),
            "test_kappa": float(test_metrics["kappa"]),
        }
    )
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    logger.log(
        "Best | "
        f"epoch={best_metrics['epoch']} | "
        f"val_oa={best_metrics['val_oa']:.4f} | "
        f"val_aa={best_metrics['val_aa']:.4f} | "
        f"val_kappa={best_metrics['val_kappa']:.4f}"
    )
    logger.log(
        "Test | "
        f"test_oa={best_metrics['test_oa']:.4f} | "
        f"test_aa={best_metrics['test_aa']:.4f} | "
        f"test_kappa={best_metrics['test_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
