"""
train.py — Unified training script for SF PoC wind downscaling.

Supports 3 model architectures:
  --model mlp    : MLP per-node baseline
  --model unet   : U-Net 3D on regular grid
  --model gnn    : GATv2 on k-NN graph

Usage
-----
    cd services/module2b-surrogate
    python train.py \
        --model     mlp \
        --data-dir  ../../data/cfd-database/sf_poc/ \
        --dataset   ../../data/cfd-database/sf_poc/dataset.yaml \
        --output    ../../data/models/module2b_poc/ \
        --epochs    100
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLP training loop
# ---------------------------------------------------------------------------

def train_mlp(args):
    from src.dataset_sf import SFUnstructuredDataset
    from src.model_mlp import MLPSpeedup

    device = get_device()
    logger.info("Training MLP on %s", device)

    train_ds = SFUnstructuredDataset(args.data_dir, args.dataset, split="train")
    val_ds = SFUnstructuredDataset(args.data_dir, args.dataset, split="val")

    # MLP processes each case independently (variable N, no batching across cases)
    model = MLPSpeedup().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    best_val_loss = float("inf")
    output_dir = Path(args.output) / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for i in range(len(train_ds)):
            sample = train_ds[i]
            node_feat = sample["node_features"].to(device)
            global_feat = sample["global_features"].to(device)
            target = sample["target"].to(device)

            pred = model(node_feat, global_feat)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_ds)):
                sample = val_ds[i]
                pred = model(
                    sample["node_features"].to(device),
                    sample["global_features"].to(device),
                )
                val_loss += criterion(pred, sample["target"].to(device)).item()
        val_loss /= max(len(val_ds), 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                epoch + 1, args.epochs, train_loss, val_loss,
                optimizer.param_groups[0]["lr"],
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info("MLP training done. Best val_loss=%.6f", best_val_loss)
    return best_val_loss


# ---------------------------------------------------------------------------
# U-Net 3D training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Physics losses (PINN)
# ---------------------------------------------------------------------------

# Z-levels for dz computation (must match export_sf_dataset.py)
_Z_LEVELS = np.geomspace(5, 5000, 32).astype(np.float32)


def divergence_loss(
    pred: torch.Tensor, inner_pad: int, dx: float = 31.25,
) -> torch.Tensor:
    """Penalize ∇·U ≠ 0 on the predicted velocity field.

    pred: (B, 5, Ny, Nx, Nz) — channels u, v, w, T, q (normalized)
    dx: horizontal grid spacing [m] (4000m / 128 = 31.25m)

    Computes du/dx + dv/dy + dw/dz using central differences.
    """
    # Extract velocity channels and denormalize
    u = pred[:, 0] * 20.0   # (B, Ny, Nx, Nz) in m/s
    v = pred[:, 1] * 20.0
    w = pred[:, 2] * 20.0

    # Crop to inner zone (but keep 1-pixel border for finite differences)
    p = max(inner_pad - 1, 0)
    if p > 0:
        u = u[:, p:-p, p:-p, :]
        v = v[:, p:-p, p:-p, :]
        w = w[:, p:-p, p:-p, :]

    # du/dx: central differences along Nx (dim=2)
    du_dx = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dx)
    # dv/dy: central differences along Ny (dim=1)
    dv_dy = (v[:, 2:, :, :] - v[:, :-2, :, :]) / (2.0 * dx)

    # dw/dz: non-uniform spacing along Nz (dim=3)
    z_levels = torch.from_numpy(_Z_LEVELS).to(pred.device)
    dz = z_levels[2:] - z_levels[:-2]  # (Nz-2,)
    dw_dz = (w[:, :, :, 2:] - w[:, :, :, :-2]) / dz[None, None, None, :]

    # Align dimensions (trim to common inner size)
    ny = min(du_dx.shape[1], dv_dy.shape[1], dw_dz.shape[1])
    nx = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
    nz = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])

    div = (
        du_dx[:, :ny, :nx, :nz]
        + dv_dy[:, :ny, :nx, :nz]
        + dw_dz[:, :ny, :nx, :nz]
    )
    return div.pow(2).mean()


def terrain_bc_loss(
    pred: torch.Tensor, inp: torch.Tensor, inner_pad: int,
) -> torch.Tensor:
    """Penalize vertical velocity inconsistent with terrain slope at ground level.

    At z_agl ≈ 5m (lowest level), w should satisfy:
        w ≈ u * dz_terrain/dx + v * dz_terrain/dy

    pred: (B, 5, Ny, Nx, Nz)
    inp:  (B, C, Ny, Nx, Nz) — channel 0 is terrain/TERRAIN_SCALE
    """
    dx = 31.25  # m
    TERRAIN_SCALE = 500.0

    # Terrain elevation: (B, Ny, Nx)
    terrain = inp[:, 0, :, :, 0] * TERRAIN_SCALE  # any z-slice, terrain is constant

    # Terrain gradients via central differences
    dz_dx = (terrain[:, :, 2:] - terrain[:, :, :-2]) / (2.0 * dx)  # (B, Ny, Nx-2)
    dz_dy = (terrain[:, 2:, :] - terrain[:, :-2, :]) / (2.0 * dx)  # (B, Ny-2, Nx)

    # Predicted velocities at lowest level (iz=0), denormalized
    if inner_pad > 0:
        u0 = pred[:, 0, inner_pad:-inner_pad, inner_pad:-inner_pad, 0] * 20.0
        v0 = pred[:, 1, inner_pad:-inner_pad, inner_pad:-inner_pad, 0] * 20.0
        w0 = pred[:, 2, inner_pad:-inner_pad, inner_pad:-inner_pad, 0] * 20.0
        dz_dx = dz_dx[:, inner_pad:-inner_pad, inner_pad-1:-(inner_pad+1)]
        dz_dy = dz_dy[:, inner_pad-1:-(inner_pad+1), inner_pad:-inner_pad]
    else:
        u0 = pred[:, 0, :, :, 0] * 20.0
        v0 = pred[:, 1, :, :, 0] * 20.0
        w0 = pred[:, 2, :, :, 0] * 20.0

    # Align shapes
    ny = min(u0.shape[1], dz_dy.shape[1])
    nx = min(u0.shape[2], dz_dx.shape[2])

    w_expected = (
        u0[:, :ny, :nx] * dz_dx[:, :ny, :nx]
        + v0[:, :ny, :nx] * dz_dy[:, :ny, :nx]
    )
    w_actual = w0[:, :ny, :nx]

    return (w_actual - w_expected).pow(2).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, inner_pad: int) -> torch.Tensor:
    """MSE loss computed only on the inner prediction zone.

    pred/target: (B, C, Ny, Nx, Nz)
    inner_pad: number of context pixels on each side to exclude
    """
    if inner_pad > 0:
        pred = pred[:, :, inner_pad:-inner_pad, inner_pad:-inner_pad, :]
        target = target[:, :, inner_pad:-inner_pad, inner_pad:-inner_pad, :]
    return nn.functional.mse_loss(pred, target)


def per_variable_metrics(
    pred: torch.Tensor, target: torch.Tensor, inner_pad: int
) -> dict[str, float]:
    """Compute RMSE per output variable on the inner zone.

    pred/target: (B, 5, Ny, Nx, Nz) — channels: u, v, w, T, q
    """
    if inner_pad > 0:
        pred = pred[:, :, inner_pad:-inner_pad, inner_pad:-inner_pad, :]
        target = target[:, :, inner_pad:-inner_pad, inner_pad:-inner_pad, :]

    var_names = ["u", "v", "w", "T", "q"]
    metrics = {}
    for i, name in enumerate(var_names):
        mse = (pred[:, i] - target[:, i]).pow(2).mean().item()
        metrics[f"rmse_{name}"] = mse**0.5
    return metrics


def train_unet(args):
    from src.dataset_sf import SFGridDataset
    from src.model_unet3d import UNet3D, ProfileEncoder, GlobalEncoder

    device = get_device()
    variant = getattr(args, "variant", "volume")
    use_residual = True
    logger.info("Training U-Net 3D (%s) on %s", variant, device)

    train_ds = SFGridDataset(
        args.data_dir, args.dataset, split="train",
        variant=variant, use_residual=use_residual,
    )
    val_ds = SFGridDataset(
        args.data_dir, args.dataset, split="val",
        variant=variant, use_residual=use_residual,
    )

    n_workers = getattr(args, "num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=n_workers, pin_memory=True, prefetch_factor=2,
    )

    # Model setup depends on variant
    n_global_scalars = 4  # u_hub, dir_sin, dir_cos, Ri_b
    cond_dim = 128
    base_features = getattr(args, "base_features", 32)
    model = UNet3D(
        in_channels=train_ds.n_input_channels,
        out_channels=train_ds.n_output_channels,
        base_features=base_features,
        cond_dim=cond_dim,
    ).to(device)

    profile_encoder = None
    global_encoder = None
    if variant == "terrain_only":
        profile_encoder = ProfileEncoder(
            n_vars=5, n_levels=32, cond_dim=cond_dim, n_scalars=n_global_scalars,
        ).to(device)
        params = list(model.parameters()) + list(profile_encoder.parameters())
    else:
        # Volume variant: use GlobalEncoder for Ri_b/stability conditioning
        global_encoder = GlobalEncoder(
            n_scalars=n_global_scalars, cond_dim=cond_dim,
        ).to(device)
        params = list(model.parameters()) + list(global_encoder.parameters())

    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    if profile_encoder:
        n_params += sum(p.numel() for p in profile_encoder.parameters())
    if global_encoder:
        n_params += sum(p.numel() for p in global_encoder.parameters())
    logger.info("Model: %d parameters", n_params)

    # Inner pad: context pixels to exclude from loss (grid_size - grid_size//2) // 2
    # For 128×128 grid with 2×2 km prediction in 4×4 km: pad = 32
    inner_pad = getattr(args, "inner_pad", 32)

    # Physics loss config
    use_physics = getattr(args, "physics", False)
    lambda_div = getattr(args, "lambda_div", 0.1)
    lambda_bc = getattr(args, "lambda_bc", 0.05)
    if use_physics:
        logger.info("PINN mode: lambda_div=%.3f, lambda_bc=%.3f", lambda_div, lambda_bc)
        suffix = f"{variant}_pinn"
    else:
        suffix = variant

    best_val_loss = float("inf")
    output_dir = Path(args.output) / f"unet_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        if profile_encoder:
            profile_encoder.train()
        if global_encoder:
            global_encoder.train()

        train_loss = 0.0
        train_loss_phys = 0.0
        n_batches = 0
        for batch in train_loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            scalars = batch["global_scalars"].to(device)

            cond = None
            if profile_encoder and "era5_profile" in batch:
                cond = profile_encoder(batch["era5_profile"].to(device), scalars)
            elif global_encoder:
                cond = global_encoder(scalars)

            pred = model(inp, cond=cond)
            loss_data = masked_mse(pred, target, inner_pad)

            if use_physics:
                loss_div = divergence_loss(pred, inner_pad)
                loss_bc = terrain_bc_loss(pred, inp, inner_pad)
                loss = loss_data + lambda_div * loss_div + lambda_bc * loss_bc
                train_loss_phys += (lambda_div * loss_div + lambda_bc * loss_bc).item()
            else:
                loss = loss_data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss_data.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validate
        model.eval()
        if profile_encoder:
            profile_encoder.eval()
        if global_encoder:
            global_encoder.eval()

        val_loss = 0.0
        val_metrics_sum = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input"].to(device)
                target = batch["target"].to(device)
                scalars = batch["global_scalars"].to(device)

                cond = None
                if profile_encoder and "era5_profile" in batch:
                    cond = profile_encoder(batch["era5_profile"].to(device), scalars)
                elif global_encoder:
                    cond = global_encoder(scalars)

                pred = model(inp, cond=cond)
                val_loss += masked_mse(pred, target, inner_pad).item()

                metrics = per_variable_metrics(pred, target, inner_pad)
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
                n_val += 1

        val_loss /= max(n_val, 1)
        val_metrics = {k: v / n_val for k, v in val_metrics_sum.items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            phys_str = ""
            if use_physics:
                phys_str = f"  phys={train_loss_phys / max(n_batches, 1):.6f}"
            logger.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f%s  %s",
                epoch + 1, args.epochs, train_loss, val_loss, phys_str, metrics_str,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {"model": model.state_dict()}
            if profile_encoder:
                state["profile_encoder"] = profile_encoder.state_dict()
            if global_encoder:
                state["global_encoder"] = global_encoder.state_dict()
            torch.save(state, output_dir / "best_model.pt")

    state = {"model": model.state_dict()}
    if profile_encoder:
        state["profile_encoder"] = profile_encoder.state_dict()
    if global_encoder:
        state["global_encoder"] = global_encoder.state_dict()
    torch.save(state, output_dir / "final_model.pt")
    logger.info("U-Net training done. Best val_loss=%.6f", best_val_loss)
    return best_val_loss


# ---------------------------------------------------------------------------
# Generic grid model training (FNO, Factored)
# ---------------------------------------------------------------------------

def train_grid_model(args, model_type: str = "fno"):
    """Train FNO or Factored model — same data pipeline as U-Net."""
    from src.dataset_sf import SFGridDataset

    device = get_device()
    logger.info("Training %s on %s", model_type.upper(), device)

    train_ds = SFGridDataset(
        args.data_dir, args.dataset, split="train",
        variant="volume", use_residual=True,
    )
    val_ds = SFGridDataset(
        args.data_dir, args.dataset, split="val",
        variant="volume", use_residual=True,
    )

    n_workers = getattr(args, "num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=n_workers, pin_memory=True, prefetch_factor=2,
    )

    # Build model
    if model_type == "fno":
        from src.model_fno3d import FNO3D
        modes = tuple(getattr(args, "fno_modes", [16, 16, 8]))
        model = FNO3D(
            in_channels=train_ds.n_input_channels,
            out_channels=train_ds.n_output_channels,
            width=getattr(args, "fno_width", 32),
            modes=modes,
            n_layers=getattr(args, "fno_layers", 4),
        ).to(device)
    elif model_type == "factored":
        from src.model_factored import FactoredUNet
        model = FactoredUNet(
            in_channels=train_ds.n_input_channels,
            out_channels=train_ds.n_output_channels,
            base_features=getattr(args, "base_features", 32),
            vertical_layers=4,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_params)

    inner_pad = getattr(args, "inner_pad", 32)

    # Physics losses
    use_physics = getattr(args, "physics", False)
    lambda_div = getattr(args, "lambda_div", 0.1)
    lambda_bc = getattr(args, "lambda_bc", 0.05)
    if use_physics:
        logger.info("PINN mode: lambda_div=%.3f, lambda_bc=%.3f", lambda_div, lambda_bc)

    best_val_loss = float("inf")
    output_dir = Path(args.output) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)

            pred = model(inp)
            loss_data = masked_mse(pred, target, inner_pad)

            if use_physics:
                loss = loss_data + lambda_div * divergence_loss(pred, inner_pad) + lambda_bc * terrain_bc_loss(pred, inp, inner_pad)
            else:
                loss = loss_data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss_data.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_metrics_sum = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input"].to(device)
                target = batch["target"].to(device)
                pred = model(inp)
                val_loss += masked_mse(pred, target, inner_pad).item()
                metrics = per_variable_metrics(pred, target, inner_pad)
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
                n_val += 1

        val_loss /= max(n_val, 1)
        val_metrics = {k: v / n_val for k, v in val_metrics_sum.items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logger.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  %s",
                epoch + 1, args.epochs, train_loss, val_loss, metrics_str,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict()}, output_dir / "best_model.pt")

    torch.save({"model": model.state_dict()}, output_dir / "final_model.pt")
    logger.info("%s training done. Best val_loss=%.6f", model_type.upper(), best_val_loss)
    return best_val_loss


# ---------------------------------------------------------------------------
# GNN training loop
# ---------------------------------------------------------------------------

def train_gnn(args):
    from src.dataset_sf import SFUnstructuredDataset
    from src.model_gnn import GNNSpeedup

    device = get_device()
    logger.info("Training GNN on %s", device)

    train_ds = SFUnstructuredDataset(args.data_dir, args.dataset, split="train")
    val_ds = SFUnstructuredDataset(args.data_dir, args.dataset, split="val")

    model = GNNSpeedup().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    best_val_loss = float("inf")
    output_dir = Path(args.output) / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for i in range(len(train_ds)):
            sample = train_ds[i]
            node_feat = sample["node_features"].to(device)
            global_feat = sample["global_features"].to(device)
            target = sample["target"].to(device)

            # Position for k-NN: use (x, y, z_agl) from node features (first 3 columns)
            pos = node_feat[:, :3]

            pred = model(node_feat, global_feat, pos=pos)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_ds)):
                sample = val_ds[i]
                node_feat = sample["node_features"].to(device)
                pos = node_feat[:, :3]
                pred = model(
                    node_feat, sample["global_features"].to(device), pos=pos,
                )
                val_loss += criterion(pred, sample["target"].to(device)).item()
        val_loss /= max(len(val_ds), 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch + 1, args.epochs, train_loss, val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info("GNN training done. Best val_loss=%.6f", best_val_loss)
    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train wind downscaling model")
    parser.add_argument("--model", required=True, choices=["mlp", "unet", "gnn", "fno", "factored"])
    parser.add_argument("--data-dir", required=True, help="Exported CFD data directory")
    parser.add_argument("--dataset", required=True, help="dataset.yaml from assembler")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (U-Net only)")
    parser.add_argument(
        "--variant", default="volume", choices=["volume", "terrain_only"],
        help="U-Net input variant: volume (7ch) or terrain_only (2ch + FiLM)",
    )
    parser.add_argument(
        "--inner-pad", type=int, default=32,
        help="Context pixels to exclude from loss (default: 32 for 128→64 inner zone)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers for parallel I/O (default: 4)",
    )
    parser.add_argument(
        "--physics", action="store_true", default=False,
        help="Enable PINN physics losses (divergence-free + terrain BC)",
    )
    parser.add_argument(
        "--lambda-div", type=float, default=0.1,
        help="Weight for divergence loss (default: 0.1)",
    )
    parser.add_argument(
        "--lambda-bc", type=float, default=0.05,
        help="Weight for terrain BC loss (default: 0.05)",
    )
    parser.add_argument(
        "--base-features", type=int, default=32,
        help="Base feature width for U-Net/Factored (default: 32)",
    )
    parser.add_argument(
        "--fno-width", type=int, default=32,
        help="FNO hidden width (default: 32)",
    )
    parser.add_argument(
        "--fno-modes", type=int, nargs=3, default=[16, 16, 8],
        help="FNO Fourier modes (ny nx nz, default: 16 16 8)",
    )
    parser.add_argument(
        "--fno-layers", type=int, default=4,
        help="FNO number of spectral layers (default: 4)",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.model == "mlp":
        best_loss = train_mlp(args)
    elif args.model == "unet":
        best_loss = train_unet(args)
    elif args.model == "gnn":
        best_loss = train_gnn(args)
    elif args.model == "fno":
        best_loss = train_grid_model(args, model_type="fno")
    elif args.model == "factored":
        best_loss = train_grid_model(args, model_type="factored")

    elapsed = time.time() - t0
    logger.info(
        "Training complete: model=%s, best_val_loss=%.6f, time=%.0fs",
        args.model, best_loss, elapsed,
    )


if __name__ == "__main__":
    main()
