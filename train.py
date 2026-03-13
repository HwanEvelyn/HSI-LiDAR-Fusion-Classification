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

from dataset.mat_loader import build_official_houston_split, load_houston_hl
from dataset.patch_dataset import HsiLidarPatchDataset, bulid_index
from dataset.preprocessing import pca_reduce, pca_reduce_with_mask, zscore_norm, zscore_norm_with_mask
from models.baseline_cnn import BaselineFusionNet
from models.hct_bgc import HCT_BGC
from utils.logger import SimpleLogger
from utils.metrics import confusion_matrix, oa_aa_kappa
from utils.seed import set_seed


@dataclass
class SplitLoaders:
    train: DataLoader
    test: DataLoader
    hsi_channels: int
    num_classes: int


@dataclass
class EpochStats:
    total_loss: float
    ce_loss: float
    contrastive_loss: float
    y_true: np.ndarray
    y_pred: np.ndarray


def create_model(args: argparse.Namespace | dict, hsi_channels: int, num_classes: int) -> nn.Module:
    model_name = args["model"] if isinstance(args, dict) else args.model
    fusion_layers = args.get("fusion_layers", 1) if isinstance(args, dict) else args.fusion_layers
    if model_name == "baseline":
        return BaselineFusionNet(
            hsi_in_channels=hsi_channels,
            num_classes=num_classes,
        )
    if model_name == "hct_bgc":
        return HCT_BGC(
            hsi_in_channels=hsi_channels,
            num_classes=num_classes,
            embed_dim=args.get("embed_dim", 128) if isinstance(args, dict) else args.embed_dim,
            num_heads=args.get("num_heads", 4) if isinstance(args, dict) else args.num_heads,
            num_layers=args.get("num_layers", 2) if isinstance(args, dict) else args.num_layers,
            fusion_layers=fusion_layers,
            dropout=args.get("dropout", 0.1) if isinstance(args, dict) else args.dropout,
            patch_size=args.get("patch_size", 11) if isinstance(args, dict) else args.patch_size,
            disable_gate=args.get("disable_gate", False) if isinstance(args, dict) else args.disable_gate,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def unpack_model_outputs(outputs: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError("Model output dictionary must contain a 'logits' key")
        return outputs
    return {"logits": outputs}


def collect_model_config(model: nn.Module, args: argparse.Namespace, hsi_channels: int, num_classes: int) -> Dict[str, object]:
    config: Dict[str, object] = {
        "model": args.model,
        "hsi_in_channels": hsi_channels,
        "num_classes": num_classes,
        "use_contrastive": args.use_contrastive,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
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
    seed: int,
    split_mode: str,
    preprocess_scope: str,
    device: torch.device,
) -> SplitLoaders:
    data = load_houston_hl(data_root)

    if split_mode == "official":
        train_items, test_items, num_classes = build_official_houston_split(data)
    else:
        train_items, test_items, num_classes = bulid_index(
            data.gt,
            train_ratio=train_ratio,
            seed=seed,
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

    train_dataset_full = HsiLidarPatchDataset(hsi, lidar, train_items, patch_size)
    test_dataset = HsiLidarPatchDataset(hsi, lidar, test_items, patch_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": should_pin_memory(device),
    }
    train_loader = DataLoader(train_dataset_full, shuffle=True, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return SplitLoaders(
        train=train_loader,
        test=test_loader,
        hsi_channels=hsi.shape[-1],
        num_classes=num_classes,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_contrastive: bool,
    contrastive_weight: float,
    temperature: float,
    optimizer: Adam | None = None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    contrastive_loss_sum = 0.0
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
            if use_contrastive:
                if "h_cls" not in outputs or "l_cls" not in outputs:
                    raise KeyError("启用对比损失时，模型输出必须包含 'h_cls' 和 'l_cls'")
                contrastive_loss = info_nce_loss(outputs["h_cls"], outputs["l_cls"], temperature=temperature)
            # 总损失 = CE + weight * Contrative
            loss = ce_loss + contrastive_weight * contrastive_loss
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = target.size(0)
        total_loss_sum += float(loss.item()) * batch_size
        ce_loss_sum += float(ce_loss.item()) * batch_size
        contrastive_loss_sum += float(contrastive_loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    avg_total = total_loss_sum / max(total_samples, 1)
    avg_ce = ce_loss_sum / max(total_samples, 1)
    avg_contrastive = contrastive_loss_sum / max(total_samples, 1)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return EpochStats(
        total_loss=avg_total,
        ce_loss=avg_ce,
        contrastive_loss=avg_contrastive,
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
        optimizer=None,
    )
    cm = confusion_matrix(epoch_stats.y_true, epoch_stats.y_pred, num_classes=num_classes)
    metrics = oa_aa_kappa(cm)
    metrics["loss"] = epoch_stats.ce_loss
    metrics["confusion_matrix"] = cm
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HSI-LiDAR baseline training")
    parser.add_argument("--data-root", type=str, default="data/raw/Houston 2013/2013_DFTC")
    parser.add_argument("--model", type=str, choices=["baseline", "hct_bgc"], default="baseline")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, choices=["random", "official"], default="official")
    parser.add_argument("--preprocess-scope", type=str, choices=["full", "train"], default="train")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/run_baseline")
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.use_contrastive and args.model != "hct_bgc":
        raise ValueError("--use-contrastive 当前仅支持 --model hct_bgc")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(output_dir / "train_log.txt")

    device = resolve_device(args.device)
    logger.log(f"Using device: {device}")
    log_device_info(logger, device)

    loaders = build_dataloaders(
        data_root=str(data_root),
        patch_size=args.patch_size,
        pca_components=args.pca_components,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_mode=args.split_mode,
        preprocess_scope=args.preprocess_scope,
        device=device,
    )

    model = create_model(args, loaders.hsi_channels, loaders.num_classes).to(device)
    model_config = collect_model_config(model, args, loaders.hsi_channels, loaders.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.log(
        f"Train/Test batches: {len(loaders.train)}/{len(loaders.test)} | "
        f"HSI channels: {loaders.hsi_channels} | Classes: {loaders.num_classes} | "
        f"split={args.split_mode} | preprocess={args.preprocess_scope} | "
        f"model={args.model}"
    )
    logger.log(f"Model config: {json.dumps(model_config, sort_keys=True)}")
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    best_oa = -1.0
    best_metrics: Dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            loaders.train,
            criterion,
            device,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            optimizer=optimizer,
        )
        test_metrics = evaluate_split(model, loaders.test, criterion, device, num_classes=loaders.num_classes)
        logger.log(
            f"Epoch {epoch:03d} | "
            f"train_ce={train_stats.ce_loss:.4f} | "
            f"train_contrastive={train_stats.contrastive_loss:.4f} | "
            f"train_total={train_stats.total_loss:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_oa={test_metrics['oa']:.4f} | "
            f"test_aa={test_metrics['aa']:.4f} | "
            f"test_kappa={test_metrics['kappa']:.4f}"
        )

        if test_metrics["oa"] > best_oa:
            best_oa = float(test_metrics["oa"])
            best_metrics = {
                "epoch": epoch,
                "train_ce": float(train_stats.ce_loss),
                "train_contrastive": float(train_stats.contrastive_loss),
                "train_total": float(train_stats.total_loss),
                "test_loss": float(test_metrics["loss"]),
                "oa": float(test_metrics["oa"]),
                "aa": float(test_metrics["aa"]),
                "kappa": float(test_metrics["kappa"]),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "model_config": model_config,
                    "hsi_channels": loaders.hsi_channels,
                    "num_classes": loaders.num_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metrics": best_metrics,
                },
                output_dir / "best.pth",
            )
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)

    if best_metrics is None:
        raise RuntimeError("Training finished without producing best metrics")

    logger.log(
        "Best | "
        f"epoch={best_metrics['epoch']} | "
        f"oa={best_metrics['oa']:.4f} | "
        f"aa={best_metrics['aa']:.4f} | "
        f"kappa={best_metrics['kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
