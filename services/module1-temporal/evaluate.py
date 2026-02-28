"""
evaluate.py — Évaluation du module 1 sur le split test (IOP Perdigão 2017).

Usage :
    python evaluate.py \\
        --zarr-6h  ../../data/raw/era5_perdigao.zarr \\
        --zarr-1h  ../../data/raw/era5_hourly_perdigao.zarr \\
        --model    ../../data/models/module1/best_model.pt \\
        --norm-stats ../../data/models/module1/norm_stats.json

Sorties :
  - RMSE par variable et par niveau (tableau)
  - Amélioration relative vs baseline linéaire (%)
  - Wind speed RMSE et direction MAE à 850/700/500 hPa
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset import ERA5TemporalDataset
from src.model import AdvectionResidualInterpolator
from src.normalization import NormStats, VARIABLE_ORDER


PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]


def _load_model(model_path: str, device: torch.device) -> AdvectionResidualInterpolator:
    checkpoint = torch.load(model_path, map_location=device)
    model = AdvectionResidualInterpolator(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _linear_interp(
    S0: torch.Tensor,
    S1: torch.Tensor,
    n_intermediate: int,
) -> torch.Tensor:
    """Baseline : interpolation linéaire entre S0 et S1."""
    taus = torch.tensor(
        [(k + 1) / (n_intermediate + 1) for k in range(n_intermediate)],
        dtype=S0.dtype, device=S0.device,
    )
    # (B, T, V, L, H, W)
    return (
        (1.0 - taus).view(1, -1, 1, 1, 1, 1) * S0.unsqueeze(1)
        + taus.view(1, -1, 1, 1, 1, 1) * S1.unsqueeze(1)
    )


@click.command()
@click.option("--zarr-6h",    required=True)
@click.option("--zarr-1h",    required=True)
@click.option("--model",      "model_path", required=True, help="Chemin vers best_model.pt")
@click.option("--norm-stats", required=True, help="Chemin vers norm_stats.json")
@click.option("--grid-size",  default=None, type=int)
@click.option("--split",      default="test",
              type=click.Choice(["train", "val", "test"]))
@click.option("--batch-size", default=16, type=int)
@click.option("--device",     default="auto")
def main(zarr_6h, zarr_1h, model_path, norm_stats, grid_size, split, batch_size, device):
    """Évalue le modèle et compare à la baseline linéaire."""

    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else
        "mps"  if (device == "auto" and torch.backends.mps.is_available()) else
        device if device != "auto" else "cpu"
    )
    print(f"Device : {dev}")

    ns = NormStats.load(norm_stats)
    model = _load_model(model_path, dev)
    n_interm = model.n_intermediate

    ds = ERA5TemporalDataset(
        zarr_6h, zarr_1h, ns, split=split, grid_size=grid_size,
        n_intermediate=n_interm,
    )
    print(ds.info())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    n_vars   = len(VARIABLE_ORDER)
    n_levels = len(PRESSURE_LEVELS)

    # Accumulateurs RMSE² par variable et niveau
    model_sq  = np.zeros((n_vars, n_levels), dtype=np.float64)
    linear_sq = np.zeros((n_vars, n_levels), dtype=np.float64)
    count_n   = np.zeros((n_vars, n_levels), dtype=np.int64)

    # Pour le vent : vitesse et direction
    ws_model_sq  = np.zeros(n_levels, dtype=np.float64)
    ws_linear_sq = np.zeros(n_levels, dtype=np.float64)
    wd_model_ae  = np.zeros(n_levels, dtype=np.float64)
    wd_linear_ae = np.zeros(n_levels, dtype=np.float64)
    wind_n       = np.zeros(n_levels, dtype=np.int64)

    with torch.no_grad():
        for S0, S1, targets in loader:
            S0      = S0.to(dev)
            S1      = S1.to(dev)
            targets = targets.to(dev)   # (B, T, V, L, H, W)

            preds_model  = model(S0, S1)
            preds_linear = _linear_interp(S0, S1, n_interm)

            # Dénormaliser pour avoir les unités physiques
            B, T, V, L, H, W = preds_model.shape

            preds_model_np  = preds_model.cpu().numpy()
            preds_linear_np = preds_linear.cpu().numpy()
            targets_np      = targets.cpu().numpy()

            for vi, var in enumerate(VARIABLE_ORDER):
                mu  = ns.mean[var]   # (L,)
                sig = ns.std[var]    # (L,)

                # Dénormaliser
                pm = preds_model_np[:, :, vi]   * (sig + 1e-8)[None, None, :, None, None] + mu[None, None, :, None, None]
                pl = preds_linear_np[:, :, vi]  * (sig + 1e-8)[None, None, :, None, None] + mu[None, None, :, None, None]
                tg = targets_np[:, :, vi]       * (sig + 1e-8)[None, None, :, None, None] + mu[None, None, :, None, None]

                for li in range(L):
                    err_m = pm[:, :, li] - tg[:, :, li]  # (B, T, H, W)
                    err_l = pl[:, :, li] - tg[:, :, li]
                    n = err_m.size
                    model_sq[vi, li]  += (err_m ** 2).sum()
                    linear_sq[vi, li] += (err_l ** 2).sum()
                    count_n[vi, li]   += n

            # Vitesse et direction de vent (u=0, v=1)
            u_mu, v_mu   = ns.mean["u"], ns.mean["v"]
            u_sig, v_sig = ns.std["u"],  ns.std["v"]

            for li in range(n_levels):
                u_pm = preds_model_np[:, :, 0, li]  * (u_sig[li] + 1e-8) + u_mu[li]
                v_pm = preds_model_np[:, :, 1, li]  * (v_sig[li] + 1e-8) + v_mu[li]
                u_pl = preds_linear_np[:, :, 0, li] * (u_sig[li] + 1e-8) + u_mu[li]
                v_pl = preds_linear_np[:, :, 1, li] * (v_sig[li] + 1e-8) + v_mu[li]
                u_tg = targets_np[:, :, 0, li]      * (u_sig[li] + 1e-8) + u_mu[li]
                v_tg = targets_np[:, :, 1, li]      * (v_sig[li] + 1e-8) + v_mu[li]

                ws_pm = np.sqrt(u_pm ** 2 + v_pm ** 2)
                ws_pl = np.sqrt(u_pl ** 2 + v_pl ** 2)
                ws_tg = np.sqrt(u_tg ** 2 + v_tg ** 2)

                wd_pm = np.arctan2(u_pm, v_pm)
                wd_pl = np.arctan2(u_pl, v_pl)
                wd_tg = np.arctan2(u_tg, v_tg)

                # Erreur angulaire circulaire [0, π]
                def _angle_err(a, b):
                    d = np.abs(a - b)
                    return np.minimum(d, 2 * np.pi - d)

                n = ws_tg.size
                ws_model_sq[li]  += ((ws_pm - ws_tg) ** 2).sum()
                ws_linear_sq[li] += ((ws_pl - ws_tg) ** 2).sum()
                wd_model_ae[li]  += _angle_err(wd_pm, wd_tg).sum()
                wd_linear_ae[li] += _angle_err(wd_pl, wd_tg).sum()
                wind_n[li] += n

    # ── Affichage des résultats ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Résultats — split={split!r}, n_windows={len(ds)}")
    print(f"{'='*72}\n")

    print("RMSE par variable et niveau de pression (unités physiques) :")
    print(f"{'Variable':>10} {'Niveau':>8} {'RMSE modèle':>14} {'RMSE linéaire':>16} {'Gain %':>8}")
    print("-" * 60)
    for vi, var in enumerate(VARIABLE_ORDER):
        for li, hpa in enumerate(PRESSURE_LEVELS):
            n = max(count_n[vi, li], 1)
            rmse_m = float(np.sqrt(model_sq[vi, li] / n))
            rmse_l = float(np.sqrt(linear_sq[vi, li] / n))
            gain   = 100 * (rmse_l - rmse_m) / (rmse_l + 1e-10)
            print(f"{var:>10} {hpa:>6} hPa {rmse_m:>12.4f}  {rmse_l:>14.4f}  {gain:>+7.1f}%")
        print()

    print("\nVitesse de vent (m/s) et direction (°) aux niveaux principaux :")
    print(f"{'Niveau':>8} {'WS modèle':>12} {'WS linéaire':>14} {'WS gain%':>10} "
          f"{'WD modèle°':>12} {'WD linéaire°':>14}")
    print("-" * 72)
    highlight = {850, 700, 500}
    for li, hpa in enumerate(PRESSURE_LEVELS):
        if hpa not in highlight:
            continue
        n = max(wind_n[li], 1)
        ws_rm = float(np.sqrt(ws_model_sq[li] / n))
        ws_rl = float(np.sqrt(ws_linear_sq[li] / n))
        wd_rm = float(np.degrees(wd_model_ae[li] / n))
        wd_rl = float(np.degrees(wd_linear_ae[li] / n))
        ws_gain = 100 * (ws_rl - ws_rm) / (ws_rl + 1e-10)
        print(f"{hpa:>6} hPa {ws_rm:>12.3f} {ws_rl:>14.3f} {ws_gain:>+9.1f}% "
              f"{wd_rm:>12.2f}° {wd_rl:>12.2f}°")

    # Score global sur u et v
    n_uv = max(count_n[0].sum() + count_n[1].sum(), 1)
    rmse_uv_m = float(np.sqrt((model_sq[0].sum() + model_sq[1].sum()) / n_uv))
    rmse_uv_l = float(np.sqrt((linear_sq[0].sum() + linear_sq[1].sum()) / n_uv))
    gain_uv   = 100 * (rmse_uv_l - rmse_uv_m) / (rmse_uv_l + 1e-10)
    print(f"\nRMSE global u+v : modèle={rmse_uv_m:.4f} m/s, "
          f"linéaire={rmse_uv_l:.4f} m/s, gain={gain_uv:+.1f}%")


if __name__ == "__main__":
    main()
