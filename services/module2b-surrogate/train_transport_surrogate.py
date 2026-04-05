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

# FuXi-CFD grid constants
HF_Z_LEVELS = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    106.5, 114.95, 125.94, 140.22, 158.78, 182.91, 214.29,
], dtype=np.float64)
DX = 30.0  # meters


# ── Physics losses (PINO) ────────────────────────────────────────────────────

def transport_residual_loss(
    pred: torch.Tensor,
    inp: torch.Tensor,
    kappa: float = 5.0,
) -> torch.Tensor:
    """PDE residual loss: U·∇φ - κ∇²φ ≈ 0 for passive scalar transport.

    pred: (B, 2, nz, ny, nx) — normalized T/T_SCALE, q/Q_SCALE
    inp:  (B, 8, nz, ny, nx) — channels [u/15, v/15, w/5, k/5, dem/500, z0, T_prof/300, q_prof/0.01]

    Computes the steady-state residual R = U·∇φ - κ∇²φ using central differences.
    Penalizes R² averaged over interior cells.
    """
    from src.dataset_fuxicfd import (WIND_UV_SCALE, WIND_W_SCALE,
        T_RESIDUAL_SCALE, Q_RESIDUAL_SCALE, T_PROFILE_SCALE, Q_PROFILE_SCALE)

    # Denormalize velocity (from input channels 0-2)
    u = inp[:, 0] * WIND_UV_SCALE   # (B, nz, ny, nx) m/s
    v = inp[:, 1] * WIND_UV_SCALE
    w = inp[:, 2] * WIND_W_SCALE

    # Reconstruct absolute fields: phi = profile + residual * scale
    # Input channels 6,7 contain profile/T_PROFILE_SCALE, profile/Q_PROFILE_SCALE
    # pred channels contain (phi - profile) / RESIDUAL_SCALE
    scales = [T_RESIDUAL_SCALE, Q_RESIDUAL_SCALE]
    profile_scales = [T_PROFILE_SCALE, Q_PROFILE_SCALE]
    profile_channels = [6, 7]
    dx = DX
    dy = DX
    z_levels = torch.from_numpy(HF_Z_LEVELS).to(pred.device, dtype=pred.dtype)

    total_loss = torch.tensor(0.0, device=pred.device)

    for ch, (res_scale, prof_scale, prof_ch) in enumerate(
            zip(scales, profile_scales, profile_channels)):
        # Reconstruct absolute scalar: phi = profile + predicted_residual * res_scale
        profile = inp[:, prof_ch] * prof_scale  # (B, nz, ny, nx) in physical units
        phi = profile + pred[:, ch] * res_scale  # (B, nz, ny, nx)

        # ∂φ/∂x: central differences along nx (dim=3)
        dphi_dx = (phi[:, :, :, 2:] - phi[:, :, :, :-2]) / (2.0 * dx)
        # ∂φ/∂y: central differences along ny (dim=2)
        dphi_dy = (phi[:, :, 2:, :] - phi[:, :, :-2, :]) / (2.0 * dy)
        # ∂φ/∂z: central differences along nz (dim=1), non-uniform
        dz = (z_levels[2:] - z_levels[:-2])  # (nz-2,)
        dphi_dz = (phi[:, 2:, :, :] - phi[:, :-2, :, :]) / dz[None, :, None, None]

        # ∂²φ/∂x²
        d2phi_dx2 = (phi[:, :, :, 2:] - 2 * phi[:, :, :, 1:-1] + phi[:, :, :, :-2]) / dx**2
        # ∂²φ/∂y²
        d2phi_dy2 = (phi[:, :, 2:, :] - 2 * phi[:, :, 1:-1, :] + phi[:, :, :-2, :]) / dy**2
        # ∂²φ/∂z²
        dz_sq = ((z_levels[2:] - z_levels[1:-1]) * (z_levels[1:-1] - z_levels[:-2]))
        d2phi_dz2 = (phi[:, 2:, :, :] - 2 * phi[:, 1:-1, :, :] + phi[:, :-2, :, :]) / dz_sq[None, :, None, None]

        # Trim all to common interior (nz-2, ny-2, nx-2)
        nz = min(dphi_dx.shape[1], dphi_dy.shape[1], dphi_dz.shape[1])
        ny = min(dphi_dx.shape[2], dphi_dy.shape[2], dphi_dz.shape[2])
        nx = min(dphi_dx.shape[3], dphi_dy.shape[3], dphi_dz.shape[3])

        # Advection: U·∇φ
        advection = (
            u[:, 1:nz+1, 1:ny+1, 1:nx+1] * dphi_dx[:, :nz, :ny, :nx]
            + v[:, 1:nz+1, 1:ny+1, 1:nx+1] * dphi_dy[:, :nz, :ny, :nx]
            + w[:, 1:nz+1, 1:ny+1, 1:nx+1] * dphi_dz[:, :nz, :ny, :nx]
        )

        # Diffusion: κ∇²φ
        diffusion = kappa * (
            d2phi_dx2[:, 1:nz+1, 1:ny+1, :][:, :, :, :nx]
            + d2phi_dy2[:, 1:nz+1, :, 1:nx+1][:, :, :ny, :]
            + d2phi_dz2[:, :, 1:ny+1, 1:nx+1][:, :nz, :, :][:, :, :ny, :nx]
        )

        # Residual: R = advection - diffusion ≈ 0
        residual = advection - diffusion
        total_loss = total_loss + residual.pow(2).mean()

    return total_loss


def boundary_consistency_loss(
    pred: torch.Tensor,
    inp: torch.Tensor,
) -> torch.Tensor:
    """Penalize predicted residuals being non-zero at inflow boundaries.

    At inflow faces, T = T_profile (residual = 0) and q = q_profile (residual = 0).
    pred: (B, 2, nz, ny, nx) — predicted residuals (should be ~0 at inflow)
    inp:  (B, 8, nz, ny, nx) — ch 0-1: u/15, v/15
    """
    from src.dataset_fuxicfd import WIND_UV_SCALE

    u = inp[:, 0] * WIND_UV_SCALE
    v = inp[:, 1] * WIND_UV_SCALE

    loss = torch.tensor(0.0, device=pred.device)

    for ch in range(2):  # T=0, q=1
        phi = pred[:, ch]  # (B, nz, ny, nx) — residual, should be 0 at inflow

        # West (x=0): inflow where u > 0 → residual should be 0
        mask_w = (u[:, :, :, 0] > 0).float()
        loss = loss + (mask_w * phi[:, :, :, 0].pow(2)).mean()

        # East (x=-1): inflow where u < 0
        mask_e = (u[:, :, :, -1] < 0).float()
        loss = loss + (mask_e * phi[:, :, :, -1].pow(2)).mean()

        # South (y=0): inflow where v > 0
        mask_s = (v[:, :, 0, :] > 0).float()
        loss = loss + (mask_s * phi[:, :, 0, :].pow(2)).mean()

        # North (y=-1): inflow where v < 0
        mask_n = (v[:, :, -1, :] < 0).float()
        loss = loss + (mask_n * phi[:, :, -1, :].pow(2)).mean()

    return loss


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

    out_dir = Path(args.output) / args.run_name
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
        model = UNet3D(in_channels=8, out_channels=2,
                       base_features=args.base_features)
    elif args.model == "fno":
        model = FNO3D(in_channels=8, out_channels=2, width=args.width,
                      modes=tuple(args.modes), n_layers=args.n_layers)
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

    # Physics loss config
    use_physics = args.lambda_pde > 0 or args.lambda_bc > 0
    if use_physics:
        logger.info(f"PINO mode: lambda_pde={args.lambda_pde}, lambda_bc={args.lambda_bc}")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_rmse_T": [], "val_rmse_q": []}
    if use_physics:
        history["train_loss_data"] = []
        history["train_loss_pde"] = []
        history["train_loss_bc"] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        train_data_losses = []
        train_pde_losses = []
        train_bc_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss_data = criterion(pred, y)
                loss = loss_data

                if args.lambda_pde > 0:
                    loss_pde = transport_residual_loss(pred.float(), x.float())
                    loss = loss + args.lambda_pde * loss_pde
                else:
                    loss_pde = torch.tensor(0.0)

                if args.lambda_bc > 0:
                    loss_bc = boundary_consistency_loss(pred.float(), x.float())
                    loss = loss + args.lambda_bc * loss_bc
                else:
                    loss_bc = torch.tensor(0.0)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            if use_physics:
                train_data_losses.append(loss_data.item())
                train_pde_losses.append(loss_pde.item())
                train_bc_losses.append(loss_bc.item())

        scheduler.step()
        avg_train = np.mean(train_losses)
        if use_physics:
            history["train_loss_data"].append(float(np.mean(train_data_losses)))
            history["train_loss_pde"].append(float(np.mean(train_pde_losses)))
            history["train_loss_bc"].append(float(np.mean(train_bc_losses)))

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
    parser.add_argument("--run-name", default=None,
                        help="Run name for output subdir (default: auto from model+params)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    # UNet3D hyperparams
    parser.add_argument("--base-features", type=int, default=32,
                        help="UNet3D base channel width (default: 32)")
    # FNO3D hyperparams
    parser.add_argument("--width", type=int, default=32,
                        help="FNO hidden channel dimension (default: 32)")
    parser.add_argument("--modes", type=int, nargs=3, default=[8, 16, 16],
                        help="FNO Fourier modes (nz ny nx, default: 8 16 16)")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="FNO number of spectral layers (default: 4)")
    # Physics-informed (PINO) losses
    parser.add_argument("--lambda-pde", type=float, default=0.0,
                        help="Weight for PDE residual loss (0 = pure data, >0 = PINO)")
    parser.add_argument("--lambda-bc", type=float, default=0.0,
                        help="Weight for boundary consistency loss")

    args = parser.parse_args()
    if args.run_name is None:
        physics_tag = ""
        if args.lambda_pde > 0 or args.lambda_bc > 0:
            physics_tag = "_pino"
        if args.model == "unet":
            args.run_name = f"unet_f{args.base_features}_lr{args.lr}{physics_tag}"
        else:
            args.run_name = f"fno_w{args.width}_m{'x'.join(map(str,args.modes))}_l{args.n_layers}_lr{args.lr}{physics_tag}"
    train(args)


if __name__ == "__main__":
    main()
