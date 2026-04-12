"""
train_wind_vit.py — Train TerrainViT for 3D wind downscaling on 9k dataset.

Supports 3 architecture variants (S1/S2/S3) with S4 training recipe:
  S1 "film":   v1 encoder + FiLM vertical decoder
  S2 "factor": Factored 2D transformer + 1D vertical MLP
  S3 "cross":  Cross-attention terrain × ERA5 tokens

S4 training recipe:
  - Linear warmup (10 epochs) + cosine decay (no restart)
  - Weight decay 0.1, patience 50
  - Charbonnier + amplitude + divergence losses

Usage:
    python train_wind_vit.py \
        --data-dir /path/to/training_9k \
        --output /path/to/models \
        --variant film \
        --preset base \
        --epochs 200
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.dataset_wind9k import Wind9kDataset, WIND_UV_SCALE, WIND_W_SCALE, T_SCALE, Q_SCALE
from src.model_vit import build_vit

logger = logging.getLogger(__name__)


# ── Loss functions ────────────────────────────────────────────────────────────

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier loss: sqrt((pred - target)^2 + eps^2).

    More robust to outliers than MSE, used in FuXi-CFD.
    """
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frequency-domain amplitude loss.

    Penalizes differences in spectral amplitude to preserve large-scale
    spatial structures. Applied on the 2D spatial dimensions (ny, nx).
    pred, target: (B, C, ny, nx, nz)
    """
    # Average over z levels for 2D spectral comparison
    pred_2d = pred.mean(dim=-1)    # (B, C, ny, nx)
    target_2d = target.mean(dim=-1)

    # 2D FFT on spatial dims
    pred_fft = torch.fft.rfft2(pred_2d, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target_2d, dim=(-2, -1))

    # Amplitude difference
    pred_amp = pred_fft.abs()
    target_amp = target_fft.abs()

    return (pred_amp - target_amp).abs().mean()


def divergence_loss(pred: torch.Tensor, z_levels: torch.Tensor,
                    dx: float = 75.0) -> torch.Tensor:
    """Soft divergence-free penalty: ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0.

    pred: (B, 5, ny, nx, nz) — channels [u, v, w, T, q]
    z_levels: (nz,) — vertical level heights [m]
    dx: grid spacing [m] (128 cells over ~9.6km domain)
    """
    u = pred[:, 0]  # (B, ny, nx, nz)
    v = pred[:, 1]
    w = pred[:, 2]

    # Central differences
    du_dx = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dx)
    dv_dy = (v[:, 2:, :, :] - v[:, :-2, :, :]) / (2.0 * dx)

    dz = z_levels[2:] - z_levels[:-2]
    dw_dz = (w[:, :, :, 2:] - w[:, :, :, :-2]) / dz[None, None, None, :]

    # Trim to common interior
    nz = min(du_dx.shape[3], dw_dz.shape[3])
    ny = min(du_dx.shape[1], dv_dy.shape[1])
    nx = min(du_dx.shape[2], dv_dy.shape[2])

    div = (du_dx[:, 1:ny+1, :nx, 1:nz+1]
           + dv_dy[:, :ny, 1:nx+1, 1:nz+1]
           + dw_dz[:, 1:ny+1, 1:nx+1, :nz])

    return div.pow(2).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Per-variable RMSE in physical units.

    pred, target: (B, 5, ny, nx, nz) — normalized
    """
    scales = [WIND_UV_SCALE, WIND_UV_SCALE, WIND_W_SCALE, T_SCALE, Q_SCALE]
    names = ["u", "v", "w", "T", "q"]
    metrics = {}
    for i, (name, scale) in enumerate(zip(names, scales)):
        diff = (pred[:, i] - target[:, i]) * scale
        metrics[f"rmse_{name}"] = torch.sqrt(diff.pow(2).mean()).item()
    return metrics


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = get_device()
    logger.info(f"Training ViT on {device}")

    out_dir = Path(args.output) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset_yaml = Path(args.data_dir) / "dataset.yaml"
    train_ds = Wind9kDataset(args.data_dir, dataset_yaml, split="train",
                             use_residual=True, augment=True)
    val_ds = Wind9kDataset(args.data_dir, dataset_yaml, split="val",
                           use_residual=True, augment=False)

    logger.info(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=True)

    # Model
    variant = getattr(args, "variant", "film")
    model = build_vit(variant=variant, preset=args.preset,
                      nz=32, img_size=128, patch_size=8)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: TerrainViT-{variant}-{args.preset}, params: {n_params:,}")

    # S4: AdamW with higher weight decay + linear warmup + cosine decay (no restart)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    warmup_epochs = 10
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(1e-6 / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Z levels for divergence loss
    z_levels = torch.tensor([
        5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
        120, 140, 160, 180, 200, 250, 300, 400, 500, 600, 700,
        800, 900, 1000, 1200, 1400, 1600, 1800, 2000,
    ], dtype=torch.float32, device=device)

    # Loss weights
    lambda_amp = args.lambda_amp
    lambda_div = args.lambda_div

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_rmse_u": [], "val_rmse_v": [], "val_rmse_w": [],
        "val_rmse_T": [], "val_rmse_q": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for terrain, era5, target in train_loader:
            terrain = terrain.to(device)
            era5 = era5.to(device)
            target = target.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(terrain, era5)
                loss_charb = charbonnier_loss(pred, target)
                loss = loss_charb

                if lambda_amp > 0:
                    loss = loss + lambda_amp * amplitude_loss(pred, target)

                if lambda_div > 0:
                    loss = loss + lambda_div * divergence_loss(
                        pred.float(), z_levels)

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
        val_metrics_acc = {f"rmse_{v}": [] for v in ["u", "v", "w", "T", "q"]}

        with torch.no_grad():
            for terrain, era5, target in val_loader:
                terrain = terrain.to(device)
                era5 = era5.to(device)
                target = target.to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(terrain, era5)
                    loss = charbonnier_loss(pred, target)
                val_losses.append(loss.item())

                m = compute_rmse(pred.float(), target.float())
                for k, v in m.items():
                    val_metrics_acc[k].append(v)

        avg_val = np.mean(val_losses)
        elapsed = time.time() - t0

        rmse_str = "  ".join(
            f"{k}={np.mean(v):.4f}" for k, v in val_metrics_acc.items())

        if epoch % 5 == 0 or epoch <= 5:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train={avg_train:.6f}  val={avg_val:.6f}  "
                f"{rmse_str}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.0f}s"
            )

        # History
        history["train_loss"].append(float(avg_train))
        history["val_loss"].append(float(avg_val))
        for k, v in val_metrics_acc.items():
            history[f"val_{k}"].append(float(np.mean(v)))

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "rmse": {k: float(np.mean(v)) for k, v in val_metrics_acc.items()},
                "variant": variant,
                "preset": args.preset,
                "n_params": n_params,
            }, out_dir / "best_model.pt")
            logger.info(f"  -> Best model saved (val_loss={avg_val:.6f})")
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
            }, out_dir / f"checkpoint_ep{epoch:03d}.pt")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Train TerrainViT for 3D wind downscaling")
    parser.add_argument("--data-dir", required=True,
                        help="Root dir with site_NNNNN_case_tsNNN/ folders + dataset.yaml")
    parser.add_argument("--output", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--run-name", default=None,
                        help="Run name for output subdir (auto from variant+preset)")
    parser.add_argument("--variant", choices=["film", "factor", "cross"],
                        default="film", help="Architecture variant (S1/S2/S3)")
    parser.add_argument("--preset", choices=["small", "base"],
                        default="base", help="ViT size preset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    # Loss weights
    parser.add_argument("--lambda-amp", type=float, default=0.1,
                        help="Weight for frequency amplitude loss")
    parser.add_argument("--lambda-div", type=float, default=0.05,
                        help="Weight for divergence penalty")

    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"vit_{args.variant}_{args.preset}"
    train(args)


if __name__ == "__main__":
    main()
