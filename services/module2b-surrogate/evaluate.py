"""
evaluate.py — Evaluate trained surrogate on the test split.

Supports: UNet, FNO, Factored, ViT (terrain-aware).

Computes:
  - Per-variable RMSE/MAE in physical units (global + per height level)
  - Skill scores vs ERA5 baseline
  - Height-resolved metrics (for FuXi-CFD Fig.4a comparison)
  - Per-case breakdown

Usage
-----
    # UNet / FNO
    python evaluate.py \
        --model-type unet \
        --weights  best_model.pt \
        --data-dir /path/to/training_9k \
        --dataset  /path/to/dataset.yaml \
        --output   /path/to/metrics/

    # ViT
    python evaluate.py \
        --model-type vit \
        --vit-preset base \
        --weights  best_model.pt \
        --data-dir /path/to/training_9k \
        --dataset  /path/to/dataset.yaml \
        --output   /path/to/metrics/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Denormalization scales (must match dataset_sf.py / dataset_wind9k.py)
WIND_SCALE = 20.0   # m/s (u, v for UNet/FNO)
WIND_W_SCALE_VIT = 5.0  # m/s (w for ViT dataset)
T_SCALE = 30.0       # K
Q_SCALE = 0.01       # kg/kg

# Default z-levels (from export_sf_dataset.py / geomspace)
Z_LEVELS = np.geomspace(5, 5000, 32).astype(np.float32)


# ── Height-resolved metrics ─────────────────────────────────────────────────

def compute_height_metrics(
    all_pred_by_z: dict[str, list[np.ndarray]],
    all_true_by_z: dict[str, list[np.ndarray]],
    z_levels: np.ndarray,
) -> dict:
    """Compute MAE and RMSE per variable per height level.

    Returns dict with keys like "height_metrics" containing arrays
    comparable to FuXi-CFD Fig. 4(a).
    """
    var_names = list(all_pred_by_z.keys())
    nz = len(z_levels)
    result = {"z_levels_m": z_levels.tolist()}

    for vn in var_names:
        mae_by_z = []
        rmse_by_z = []
        for iz in range(nz):
            p = np.concatenate([arr[:, :, iz].ravel()
                                for arr in all_pred_by_z[vn]])
            t = np.concatenate([arr[:, :, iz].ravel()
                                for arr in all_true_by_z[vn]])
            mae_by_z.append(float(np.mean(np.abs(p - t))))
            rmse_by_z.append(float(np.sqrt(np.mean((p - t) ** 2))))
        result[f"mae_{vn}"] = mae_by_z
        result[f"rmse_{vn}"] = rmse_by_z

    return result


# ── Grid model evaluation (UNet, FNO, Factored) ────────────────────────────

def evaluate_grid_model(args):
    """Evaluate UNet/FNO/Factored on SFGridDataset."""
    from src.dataset_sf import SFGridDataset
    from src.model_unet3d import UNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating %s on %s", args.model_type, device)

    test_ds = SFGridDataset(
        args.data_dir, args.dataset, split="test",
        variant="volume", use_residual=True,
    )
    logger.info("Test set: %d cases", len(test_ds))

    base_features = getattr(args, "base_features", 32)

    if args.model_type == "fno":
        from src.model_fno3d import FNO3D
        model = FNO3D(
            in_channels=test_ds.n_input_channels,
            out_channels=test_ds.n_output_channels,
        ).to(device)
    elif args.model_type == "factored":
        from src.model_factored import FactoredUNet
        model = FactoredUNet(
            in_channels=test_ds.n_input_channels,
            out_channels=test_ds.n_output_channels,
            base_features=base_features,
        ).to(device)
    else:
        model = UNet3D(
            in_channels=test_ds.n_input_channels,
            out_channels=test_ds.n_output_channels,
            base_features=base_features,
            cond_dim=0,
        ).to(device)

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_params)

    inner_pad = getattr(args, "inner_pad", 32)
    var_names = ["u", "v", "w", "T", "q"]
    scales = [WIND_SCALE, WIND_SCALE, WIND_SCALE, T_SCALE, Q_SCALE]

    case_metrics = []
    all_pred = {v: [] for v in var_names}
    all_true = {v: [] for v in var_names}
    # Height-resolved accumulators: list of (ny, nx, nz) arrays
    all_pred_by_z = {v: [] for v in var_names}
    all_true_by_z = {v: [] for v in var_names}

    t0 = time.time()
    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            inp = sample["input"].unsqueeze(0).to(device)
            target = sample["target"]

            pred = model(inp).squeeze(0).cpu()

            if inner_pad > 0:
                pred = pred[:, inner_pad:-inner_pad, inner_pad:-inner_pad, :]
                target = target[:, inner_pad:-inner_pad, inner_pad:-inner_pad, :]

            case_result = {"case_id": sample["case_id"]}
            for ci, (vn, sc) in enumerate(zip(var_names, scales)):
                p = pred[ci].numpy() * sc
                t = target[ci].numpy() * sc
                case_result[f"rmse_{vn}"] = float(np.sqrt(np.mean((p - t) ** 2)))
                case_result[f"mae_{vn}"] = float(np.mean(np.abs(p - t)))
                all_pred[vn].append(p.ravel())
                all_true[vn].append(t.ravel())
                all_pred_by_z[vn].append(p)
                all_true_by_z[vn].append(t)

            case_metrics.append(case_result)
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d cases", i + 1, len(test_ds))

    elapsed = time.time() - t0
    logger.info("Evaluated %d cases in %.1fs", len(test_ds), elapsed)

    summary = _compute_summary(var_names, all_pred, all_true, args.model_type,
                               len(test_ds), inner_pad)
    height = compute_height_metrics(all_pred_by_z, all_true_by_z, Z_LEVELS)
    summary["height_metrics"] = height

    return summary, case_metrics


# ── ViT evaluation ──────────────────────────────────────────────────────────

def evaluate_vit(args):
    """Evaluate TerrainViT on Wind9kDataset."""
    from src.dataset_wind9k import (Wind9kDataset, WIND_UV_SCALE,
                                     WIND_W_SCALE, T_SCALE as DSW_T_SCALE,
                                     Q_SCALE as DSW_Q_SCALE)
    from src.model_vit import build_vit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_variant = getattr(args, "vit_variant", "cross")
    logger.info("Evaluating ViT-%s-%s on %s", vit_variant, args.vit_preset, device)

    dataset_yaml = Path(args.data_dir) / "dataset.yaml"
    test_ds = Wind9kDataset(args.data_dir, dataset_yaml, split="test",
                            use_residual=True, augment=False)
    logger.info("Test set: %d cases", len(test_ds))

    # Auto-detect variant from checkpoint if available
    checkpoint_peek = torch.load(args.weights, map_location="cpu", weights_only=False)  # noqa: S614
    if "variant" in checkpoint_peek:
        vit_variant = checkpoint_peek["variant"]
        logger.info("Detected variant from checkpoint: %s", vit_variant)

    model = build_vit(variant=vit_variant, preset=args.vit_preset,
                      nz=32, img_size=128, patch_size=8)
    model = model.to(device)

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_params)

    var_names = ["u", "v", "w", "T", "q"]
    scales = [WIND_UV_SCALE, WIND_UV_SCALE, WIND_W_SCALE,
              DSW_T_SCALE, DSW_Q_SCALE]

    case_metrics = []
    all_pred = {v: [] for v in var_names}
    all_true = {v: [] for v in var_names}
    all_pred_by_z = {v: [] for v in var_names}
    all_true_by_z = {v: [] for v in var_names}

    t0 = time.time()
    with torch.no_grad():
        for i in range(len(test_ds)):
            terrain, era5, target = test_ds[i]
            terrain = terrain.unsqueeze(0).to(device)
            era5 = era5.unsqueeze(0).to(device)

            pred = model(terrain, era5).squeeze(0).cpu()  # (5, ny, nx, nz)

            case_result = {"case_idx": i}
            for ci, (vn, sc) in enumerate(zip(var_names, scales)):
                p = pred[ci].numpy() * sc
                t = target[ci].numpy() * sc
                rmse = float(np.sqrt(np.mean((p - t) ** 2)))
                mae = float(np.mean(np.abs(p - t)))
                case_result[f"rmse_{vn}"] = rmse
                case_result[f"mae_{vn}"] = mae
                all_pred[vn].append(p.ravel())
                all_true[vn].append(t.ravel())
                all_pred_by_z[vn].append(p)
                all_true_by_z[vn].append(t)

            case_metrics.append(case_result)
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d cases", i + 1, len(test_ds))

    elapsed = time.time() - t0
    logger.info("Evaluated %d cases in %.1fs", len(test_ds), elapsed)

    summary = _compute_summary(var_names, all_pred, all_true,
                               f"vit_{args.vit_preset}", len(test_ds), 0)
    height = compute_height_metrics(all_pred_by_z, all_true_by_z, Z_LEVELS)
    summary["height_metrics"] = height

    # Log height-resolved metrics (FuXi-CFD comparison)
    logger.info("")
    logger.info("=== Height-resolved metrics (FuXi-CFD Fig.4a comparison) ===")
    for vn in var_names:
        # Report at 10m, 50m, 100m (nearest z-level indices)
        for target_h in [10, 50, 100]:
            iz = int(np.argmin(np.abs(Z_LEVELS - target_h)))
            actual_h = Z_LEVELS[iz]
            mae_val = height[f"mae_{vn}"][iz]
            rmse_val = height[f"rmse_{vn}"][iz]
            logger.info("  %s @ %.0fm: MAE=%.4f, RMSE=%.4f",
                        vn, actual_h, mae_val, rmse_val)

    return summary, case_metrics


# ── Shared summary computation ──────────────────────────────────────────────

def _compute_summary(var_names, all_pred, all_true, model_name,
                     n_cases, inner_pad):
    summary = {"model": model_name, "n_test_cases": n_cases,
               "inner_pad": inner_pad}

    for vn in var_names:
        p = np.concatenate(all_pred[vn])
        t = np.concatenate(all_true[vn])
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        bias = float(np.mean(p - t))
        corr = float(np.corrcoef(p, t)[0, 1]) if np.std(t) > 1e-8 else 0.0
        rmse_baseline = float(np.sqrt(np.mean(t ** 2)))
        skill = 1.0 - rmse / max(rmse_baseline, 1e-8)

        summary[f"rmse_{vn}"] = rmse
        summary[f"mae_{vn}"] = mae
        summary[f"bias_{vn}"] = bias
        summary[f"corr_{vn}"] = corr
        summary[f"skill_{vn}"] = skill

        logger.info(
            "  %s: RMSE=%.4f, MAE=%.4f, bias=%.4f, corr=%.4f, skill=%.3f",
            vn, rmse, mae, bias, corr, skill,
        )

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate wind downscaling surrogate")
    parser.add_argument("--weights", required=True, help="Path to best_model.pt")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default=None,
                        help="dataset.yaml path (required for grid models)")
    parser.add_argument("--output", required=True, help="Output directory for metrics")
    parser.add_argument("--inner-pad", type=int, default=32)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument("--model-type", default="unet",
                        choices=["unet", "fno", "factored", "vit"],
                        help="Architecture type")
    parser.add_argument("--vit-preset", default="base",
                        choices=["small", "base"],
                        help="ViT preset (only for --model-type vit)")
    parser.add_argument("--vit-variant", default="cross",
                        choices=["film", "factor", "cross"],
                        help="ViT architecture variant")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "vit":
        summary, case_metrics = evaluate_vit(args)
        out_name = f"metrics_vit_{args.vit_preset}.json"
    else:
        if args.dataset is None:
            args.dataset = str(Path(args.data_dir) / "dataset.yaml")
        summary, case_metrics = evaluate_grid_model(args)
        out_name = f"metrics_{args.model_type}.json"

    with open(output_dir / out_name, "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "metrics_per_case.json", "w") as f:
        json.dump(case_metrics, f, indent=2)

    logger.info("Metrics saved to %s", output_dir)


if __name__ == "__main__":
    main()
