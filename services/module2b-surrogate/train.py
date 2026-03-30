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
    from src.model_unet3d import UNet3D, ProfileEncoder

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
    cond_dim = 128 if variant == "terrain_only" else 0
    model = UNet3D(
        in_channels=train_ds.n_input_channels,
        out_channels=train_ds.n_output_channels,
        base_features=32,
        cond_dim=cond_dim,
    ).to(device)

    profile_encoder = None
    if variant == "terrain_only":
        profile_encoder = ProfileEncoder(n_vars=5, n_levels=32, cond_dim=cond_dim).to(device)
        params = list(model.parameters()) + list(profile_encoder.parameters())
    else:
        params = model.parameters()

    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    if profile_encoder:
        n_params += sum(p.numel() for p in profile_encoder.parameters())
    logger.info("Model: %d parameters", n_params)

    # Inner pad: context pixels to exclude from loss (grid_size - grid_size//2) // 2
    # For 128×128 grid with 2×2 km prediction in 4×4 km: pad = 32
    inner_pad = getattr(args, "inner_pad", 32)

    best_val_loss = float("inf")
    output_dir = Path(args.output) / f"unet_{variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        if profile_encoder:
            profile_encoder.train()

        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)

            cond = None
            if profile_encoder and "era5_profile" in batch:
                cond = profile_encoder(batch["era5_profile"].to(device))

            pred = model(inp, cond=cond)
            loss = masked_mse(pred, target, inner_pad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validate
        model.eval()
        if profile_encoder:
            profile_encoder.eval()

        val_loss = 0.0
        val_metrics_sum = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input"].to(device)
                target = batch["target"].to(device)

                cond = None
                if profile_encoder and "era5_profile" in batch:
                    cond = profile_encoder(batch["era5_profile"].to(device))

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
            logger.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  %s",
                epoch + 1, args.epochs, train_loss, val_loss, metrics_str,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {"model": model.state_dict()}
            if profile_encoder:
                state["profile_encoder"] = profile_encoder.state_dict()
            torch.save(state, output_dir / "best_model.pt")

    state = {"model": model.state_dict()}
    if profile_encoder:
        state["profile_encoder"] = profile_encoder.state_dict()
    torch.save(state, output_dir / "final_model.pt")
    logger.info("U-Net training done. Best val_loss=%.6f", best_val_loss)
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
    parser.add_argument("--model", required=True, choices=["mlp", "unet", "gnn"])
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
    args = parser.parse_args()

    t0 = time.time()

    if args.model == "mlp":
        best_loss = train_mlp(args)
    elif args.model == "unet":
        best_loss = train_unet(args)
    elif args.model == "gnn":
        best_loss = train_gnn(args)

    elapsed = time.time() - t0
    logger.info(
        "Training complete: model=%s, best_val_loss=%.6f, time=%.0fs",
        args.model, best_loss, elapsed,
    )


if __name__ == "__main__":
    main()
