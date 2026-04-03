"""
train_transport_surrogate.py — Train a surrogate for passive scalar transport.

Maps wind fields (u,v,w,k) + terrain + BC profiles → T, q fields.
Reuses existing UNet3D or FNO3D architectures with adapted I/O channels.

Usage:
    python train_transport_surrogate.py \
        --data-dir /path/to/fuxicfd/extracted \
        --output /path/to/models/transport_surrogate \
        --model unet \
        --epochs 100 \
        --batch-size 2 \
        --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset_fuxicfd import FuXiCFDTransportDataset, create_manifest
from src.model_unet3d import UNet3D
from src.model_fno3d import FNO3D

logger = logging.getLogger(__name__)

# ── Device selection ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_rmse(pred: torch.Tensor, target: torch.Tensor,
                 scales: list[float]) -> dict[str, float]:
    """Per-variable RMSE in physical units.

    pred, target: (B, 2, nz, ny, nx) normalized
    scales: [T_SCALE, Q_SCALE] to convert back to physical units
    """
    metrics = {}
    names = ["T", "q"]
    for i, (name, scale) in enumerate(zip(names, scales)):
        diff = (pred[:, i] - target[:, i]) * scale
        rmse = torch.sqrt(diff.pow(2).mean()).item()
        metrics[f"rmse_{name}"] = rmse
    return metrics


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = get_device()
    logger.info(f"Device: {device}")

    out_dir = Path(args.output) / f"{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest_path = Path(args.data_dir) / "split_manifest.json"
    if not manifest_path.exists():
        logger.info("Creating train/val/test manifest...")
        create_manifest(args.data_dir, manifest_path)

    # Datasets
    train_ds = FuXiCFDTransportDataset(
        args.data_dir, split="train", manifest=manifest_path, augment=True)
    val_ds = FuXiCFDTransportDataset(
        args.data_dir, split="val", manifest=manifest_path, augment=False)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2,
        drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)

    # Model
    # UNet3D expects (B, C, D, H, W) — here D=nz=27, H=ny=300, W=nx=300
    if args.model == "unet":
        model = UNet3D(in_channels=8, out_channels=2, base_features=32)
    elif args.model == "fno":
        model = FNO3D(in_channels=8, out_channels=2, width=32,
                      modes=(8, 16, 16), n_layers=4)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model}, params: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Physical scales for metrics
    from src.dataset_fuxicfd import T_SCALE, Q_SCALE
    phys_scales = [T_SCALE, Q_SCALE]

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_rmse_T": [], "val_rmse_q": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = criterion(pred, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_metrics = {"rmse_T": [], "rmse_q": []}

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_losses.append(loss.item())

                m = compute_rmse(pred.float(), y.float(), phys_scales)
                for k, v in m.items():
                    val_metrics[k].append(v)

        avg_val = np.mean(val_losses)
        avg_rmse_T = np.mean(val_metrics["rmse_T"])
        avg_rmse_q = np.mean(val_metrics["rmse_q"])

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={avg_train:.6f} val={avg_val:.6f} | "
            f"T_rmse={avg_rmse_T:.3f}K q_rmse={avg_rmse_q:.6f}kg/kg | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_rmse_T"].append(avg_rmse_T)
        history["val_rmse_q"].append(avg_rmse_q)

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "rmse_T": avg_rmse_T,
                "rmse_q": avg_rmse_q,
                "model_type": args.model,
                "n_params": n_params,
            }, out_dir / "best_model.pt")
            logger.info(f"  -> New best model saved (val_loss={avg_val:.6f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
            }, out_dir / f"checkpoint_ep{epoch:03d}.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to {out_dir / 'best_model.pt'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train transport surrogate on FuXi-CFD dataset")
    parser.add_argument("--data-dir", required=True,
                        help="Root dir with case_NNNNNN/ folders")
    parser.add_argument("--output", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--model", choices=["unet", "fno"], default="unet",
                        help="Model architecture (default: unet)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
