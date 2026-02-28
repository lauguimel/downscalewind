"""
train.py — Boucle d'entraînement du module 1 (temporal downscaling 6h → 1h).

Usage :
    python train.py \\
        --zarr-6h  ../../data/raw/era5_perdigao.zarr \\
        --zarr-1h  ../../data/raw/era5_hourly_perdigao.zarr \\
        --output   ../../data/models/module1 \\
        --grid-size 7

Hyperparamètres :
    Configurables via configs/default.yaml ou flags CLI.
    Priorité : CLI > yaml > valeurs par défaut.

MLflow :
    Toutes les runs sont loguées dans data/mlruns/.
    UI : mlflow ui --port 5000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
import zarr
from torch.utils.data import DataLoader

# Accès aux modules partagés et src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset import ERA5TemporalDataset
from src.model import AdvectionResidualInterpolator
from src.normalization import NormStats, build_loss_weight_tensor, VARIABLE_ORDER


# ── Constantes physiques ──────────────────────────────────────────────────────

# Espacement physique à 40°N pour la grille ERA5 0.25°
DX_M_PHYSICAL = 21_480.0   # 0.25° × 111320 m/° × cos(40°)
DY_M_PHYSICAL = 27_830.0   # 0.25° × 111320 m/°
DT_S          = 6 * 3600.0  # 6 heures en secondes


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        default = Path(__file__).parent / "configs" / "default.yaml"
        config_path = str(default)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _val_rmse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    var_idx: int = 0,   # u par défaut (cible principale)
) -> float:
    """
    Calcule le RMSE moyen sur un DataLoader (données dénormalisées).

    Retourne le RMSE sur la variable var_idx, moyenné sur les niveaux,
    le temps et l'espace — en unités normalisées (σ ≈ 1 après normalisation).
    """
    model.eval()
    total_sq = 0.0
    total_n = 0
    with torch.no_grad():
        for S0, S1, targets in loader:
            S0      = S0.to(device)
            S1      = S1.to(device)
            targets = targets.to(device)   # (B, T, V, L, H, W)

            preds = model(S0, S1)           # (B, T, V, L, H, W)
            err = preds[:, :, var_idx] - targets[:, :, var_idx]  # (B, T, L, H, W)
            total_sq += err.pow(2).sum().item()
            total_n  += err.numel()

    model.train()
    return float(np.sqrt(total_sq / max(total_n, 1)))


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

@click.command()
@click.option("--zarr-6h",    required=True,  help="Store Zarr ERA5 6h (entrées)")
@click.option("--zarr-1h",    required=True,  help="Store Zarr ERA5 1h (cibles)")
@click.option("--output",     required=True,  help="Dossier de sortie (modèles + stats)")
@click.option("--config",     default=None,   help="Chemin vers configs/default.yaml")
@click.option("--norm-stats", default=None,   help="JSON NormStats (défaut: calculé)")
@click.option("--grid-size",  default=None,   type=int,
              help="Taille de la fenêtre spatiale (ex: 7). Défaut: tout le domaine.")
@click.option("--n-hidden",   default=None,   type=int, help="Canaux cachés du CNN")
@click.option("--batch-size", default=None,   type=int)
@click.option("--lr",         default=None,   type=float)
@click.option("--max-epochs", default=None,   type=int)
@click.option("--patience",   default=None,   type=int)
@click.option("--device",     default="auto", help="'cpu', 'cuda', 'mps', ou 'auto'")
@click.option("--experiment", default="module1-temporal",
              help="Nom de l'expérience MLflow")
@click.option("--run-name",   default=None,   help="Nom de la run MLflow")
def main(
    zarr_6h, zarr_1h, output, config, norm_stats,
    grid_size, n_hidden, batch_size, lr, max_epochs, patience,
    device, experiment, run_name,
):
    """Entraîne l'AdvectionResidualInterpolator sur ERA5 et logue dans MLflow."""

    # ── Configuration ─────────────────────────────────────────────────────────
    cfg = _load_config(config)

    # Les flags CLI écrasent le yaml
    grid_size  = grid_size  or cfg["training"].get("grid_size")
    n_hidden   = n_hidden   or cfg["model"].get("n_hidden",   48)
    batch_size = batch_size or cfg["training"].get("batch_size", 32)
    lr         = lr         or cfg["training"].get("lr",        3e-4)
    max_epochs = max_epochs or cfg["training"].get("max_epochs", 300)
    patience   = patience   or cfg["training"].get("patience",   30)

    dev = _device(device)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device : {dev}")
    print(f"Config : grid_size={grid_size}, n_hidden={n_hidden}, "
          f"batch={batch_size}, lr={lr}, epochs={max_epochs}")

    # ── NormStats ─────────────────────────────────────────────────────────────
    norm_path = output_path / "norm_stats.json"
    if norm_stats is not None:
        print(f"Chargement NormStats depuis {norm_stats}")
        ns = NormStats.load(norm_stats)
    elif norm_path.exists():
        print(f"Chargement NormStats depuis {norm_path}")
        ns = NormStats.load(norm_path)
    else:
        print("Calcul des NormStats sur le split train (peut prendre quelques minutes)…")
        # Approx. train : 4 pas/jour × 304 jours (2016-01 → 2016-10) = 1216 pas
        n_train_approx = 4 * 304
        n_times_6h = zarr.open_group(zarr_6h, mode="r")["pressure/u"].shape[0]
        ns = NormStats.compute_from_zarr(
            zarr_path=zarr_6h,
            time_slice=slice(0, min(n_times_6h, n_train_approx)),
        )
        ns.save(norm_path)
        print(f"NormStats sauvegardés dans {norm_path}")

    # Correction std pour l'advection physique
    std_u, std_v = ns.get_wind_std()
    dx_m_eff = DX_M_PHYSICAL / (std_u + 1e-8)
    dy_m_eff = DY_M_PHYSICAL / (std_v + 1e-8)
    print(f"std_u={std_u:.2f} m/s, std_v={std_v:.2f} m/s")
    print(f"dx_eff={dx_m_eff:.0f} m/σ, dy_eff={dy_m_eff:.0f} m/σ")

    # ── Datasets et DataLoaders ───────────────────────────────────────────────
    ds_train = ERA5TemporalDataset(zarr_6h, zarr_1h, ns, split="train",
                                   grid_size=grid_size)
    ds_val   = ERA5TemporalDataset(zarr_6h, zarr_1h, ns, split="val",
                                   grid_size=grid_size)

    print(ds_train.info())
    print(ds_val.info())

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(dev.type == "cuda"),
        prefetch_factor=2,
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(dev.type == "cuda"),
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    n_vars   = cfg["model"].get("n_vars",          5)
    n_levels = cfg["model"].get("n_levels",        10)
    n_interm = cfg["model"].get("n_intermediate",  5)

    model = AdvectionResidualInterpolator(
        n_vars=n_vars,
        n_levels=n_levels,
        n_hidden=n_hidden,
        n_intermediate=n_interm,
        dx_m=dx_m_eff,
        dy_m=dy_m_eff,
        dt_s=DT_S,
    ).to(dev)

    print(f"Modèle : {model.count_parameters():,} paramètres entraînables")

    # ── Poids de la loss ──────────────────────────────────────────────────────
    loss_weights = build_loss_weight_tensor(n_levels=n_levels, device=dev)
    # Shape : (1, V, 1, 1, 1) — broadcastable sur (B, T, V, L, H, W)
    loss_weights = loss_weights.unsqueeze(1)  # (1, 1, V, 1, 1, 1)

    # ── Optimiseur et scheduler ───────────────────────────────────────────────
    weight_decay = cfg["training"].get("weight_decay", 1e-5)
    grad_clip    = cfg["training"].get("grad_clip",    1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(Path(output).parent / "mlruns"))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "grid_size":    grid_size,
            "n_hidden":     n_hidden,
            "n_vars":       n_vars,
            "n_levels":     n_levels,
            "n_intermediate": n_interm,
            "batch_size":   batch_size,
            "lr":           lr,
            "weight_decay": weight_decay,
            "grad_clip":    grad_clip,
            "max_epochs":   max_epochs,
            "patience":     patience,
            "dx_m_eff":     round(dx_m_eff, 1),
            "dy_m_eff":     round(dy_m_eff, 1),
            "std_u":        round(std_u, 3),
            "std_v":        round(std_v, 3),
            "n_params":     model.count_parameters(),
            "n_train":      len(ds_train),
            "n_val":        len(ds_val),
        })

        # ── Boucle d'entraînement ─────────────────────────────────────────────
        best_val_rmse = float("inf")
        best_epoch    = 0
        no_improve    = 0
        best_model_path = output_path / "best_model.pt"

        model.train()
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss_sum = 0.0
            n_batches = 0

            for S0, S1, targets in loader_train:
                S0      = S0.to(dev)
                S1      = S1.to(dev)
                targets = targets.to(dev)   # (B, T, V, L, H, W)

                optimizer.zero_grad()
                preds = model(S0, S1)       # (B, T, V, L, H, W)

                # Weighted MSE : shape loss_weights est (1, 1, V, 1, 1, 1)
                loss = (loss_weights * (preds - targets).pow(2)).mean()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            scheduler.step()

            train_loss = train_loss_sum / max(n_batches, 1)
            val_rmse   = _val_rmse(model, loader_val, dev, var_idx=0)  # sur u
            elapsed    = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"loss={train_loss:.4f} | val_rmse_u={val_rmse:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss":  train_loss,
                "val_rmse_u":  val_rmse,
                "lr":          optimizer.param_groups[0]["lr"],
            }, step=epoch)

            # Early stopping et sauvegarde du meilleur modèle
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch    = epoch
                no_improve    = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_rmse_u":  val_rmse,
                    "model_kwargs": {
                        "n_vars":         n_vars,
                        "n_levels":       n_levels,
                        "n_hidden":       n_hidden,
                        "n_intermediate": n_interm,
                        "dx_m":           dx_m_eff,
                        "dy_m":           dy_m_eff,
                        "dt_s":           DT_S,
                    },
                }, best_model_path)
                print(f"  → Nouveau meilleur modèle (val_rmse_u={val_rmse:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping à l'époque {epoch} "
                          f"(pas d'amélioration depuis {patience} époques).")
                    break

        # ── Artefacts MLflow ──────────────────────────────────────────────────
        mlflow.log_metric("best_val_rmse_u", best_val_rmse)
        mlflow.log_metric("best_epoch",      best_epoch)
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(norm_path))

        # Log du modèle PyTorch
        checkpoint = torch.load(best_model_path, map_location="cpu")
        best_model = AdvectionResidualInterpolator(**checkpoint["model_kwargs"])
        best_model.load_state_dict(checkpoint["model_state"])
        mlflow.pytorch.log_model(best_model, "model")

        print(f"\nEntraînement terminé.")
        print(f"Meilleur modèle : époque {best_epoch}, val_rmse_u={best_val_rmse:.4f}")
        print(f"Sauvegardé dans : {best_model_path}")


if __name__ == "__main__":
    main()
