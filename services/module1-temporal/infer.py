"""
infer.py — Inférence du module 1 : ERA5 6h → ERA5 1h interpolé.

Lit un store Zarr 6h et produit un store Zarr 1h contenant les états
interpolés à chaque heure entre les pas de temps 6h connus.
Les pas de temps 6h d'origine sont recopiés à l'identique.

Usage :
    python infer.py \\
        --zarr-6h     ../../data/raw/era5_perdigao.zarr \\
        --zarr-out    ../../data/interim/era5_1h_perdigao.zarr \\
        --model       ../../data/models/module1/best_model.pt \\
        --norm-stats  ../../data/models/module1/norm_stats.json

Schéma de sortie : identique au schéma Zarr DownscaleWind (data_io.py),
time_step_hours=1, tous les pas de temps de la plage demandée.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.data_io import create_empty_store, open_store, PRESSURE_CHUNKS
from src.model import AdvectionResidualInterpolator
from src.normalization import NormStats, VARIABLE_ORDER


PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]


def _load_model(
    model_path: str,
    device: torch.device,
) -> AdvectionResidualInterpolator:
    checkpoint = torch.load(model_path, map_location=device)
    model = AdvectionResidualInterpolator(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@click.command()
@click.option("--zarr-6h",    required=True, help="Store Zarr ERA5 6h (entrées)")
@click.option("--zarr-out",   required=True, help="Store Zarr de sortie (1h)")
@click.option("--model",      "model_path", required=True,
              help="Chemin vers best_model.pt")
@click.option("--norm-stats", required=True, help="Chemin vers norm_stats.json")
@click.option("--start",      default=None,
              help="Début de la plage à inférer (YYYY-MM-DD, défaut: tout)")
@click.option("--end",        default=None,
              help="Fin de la plage (YYYY-MM-DD, défaut: tout)")
@click.option("--device",     default="auto")
@click.option("--overwrite",  is_flag=True,
              help="Écrase le store de sortie s'il existe déjà")
def main(zarr_6h, zarr_out, model_path, norm_stats, start, end, device, overwrite):
    """Interpole un store ERA5 6h vers 1h et écrit le résultat dans zarr_out."""

    dev = _select_device(device)
    print(f"Device : {dev}")

    ns    = NormStats.load(norm_stats)
    model = _load_model(model_path, dev)
    n_interm = model.n_intermediate   # 5 pas intermédiaires par défaut

    # ── Lire les timestamps du store 6h ──────────────────────────────────────
    store_in = zarr.open_group(zarr_6h, mode="r")
    _raw_time = store_in["coords/time"][:]
    # Normaliser vers int64 (zarr v2 renvoie int64, zarr v3 peut renvoyer datetime64[ns])
    if np.asarray(_raw_time).dtype.kind == "M":
        times_ns = np.asarray(_raw_time).astype("datetime64[ns]").view(np.int64)
    else:
        times_ns = np.asarray(_raw_time, dtype=np.int64)
    times_h = times_ns.view("datetime64[ns]").astype("datetime64[h]")

    lats   = store_in["coords/lat"][:]
    lons   = store_in["coords/lon"][:]
    levels = store_in["coords/level"][:]

    n_6h = len(times_h)

    # Filtrer sur la plage demandée
    t_lo = np.datetime64(start, "h") if start else times_h[0]
    t_hi = np.datetime64(end, "h")   if end   else times_h[-1] + np.timedelta64(1, "h")

    mask_in = (times_h >= t_lo) & (times_h < t_hi)
    idxs_in = np.where(mask_in)[0]

    if len(idxs_in) == 0:
        raise RuntimeError(f"Aucun pas de temps 6h dans la plage [{start}, {end}]")

    dt_6h = np.timedelta64(6, "h")
    dt_1h = np.timedelta64(1, "h")

    # Construire la liste des fenêtres : (i_s0, i_s1) pour chaque paire consécutive
    windows = []
    for k in range(len(idxs_in) - 1):
        i0 = idxs_in[k]
        i1 = idxs_in[k + 1]
        if times_h[i1] == times_h[i0] + dt_6h:
            windows.append((i0, i1))

    if not windows:
        raise RuntimeError("Aucune paire 6h consécutive trouvée dans la plage demandée.")

    # ── Calculer le nombre total de pas de temps 1h en sortie ─────────────────
    # Pour chaque fenêtre de 6h : 1 pas S0 + 5 intermédiaires = 6 pas
    # Plus le dernier S1 du dernier fenêtre
    t_out_list: list[np.datetime64] = []
    for i_s0, _ in windows:
        t0 = times_h[i_s0]
        t_out_list.append(t0)
        for k in range(1, n_interm + 1):
            t_out_list.append(t0 + k * dt_1h)
    # Ajouter le dernier S1
    i_last_s1 = windows[-1][1]
    t_out_list.append(times_h[i_last_s1])

    n_times_out = len(t_out_list)
    print(f"Fenêtres 6h : {len(windows)} | Pas horaires en sortie : {n_times_out}")

    # ── Créer le store de sortie ───────────────────────────────────────────────
    out_path = Path(zarr_out)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{zarr_out} existe déjà. Utilisez --overwrite pour écraser."
        )

    site = store_in.attrs.get("site", "unknown")
    store_out = create_empty_store(
        path=out_path,
        n_times=n_times_out,
        levels_hpa=[int(l) for l in levels],
        lats=lats.tolist(),
        lons=lons.tolist(),
        source="era5_interp_module1",
        site=site,
        time_step_hours=1,
        overwrite=overwrite,
    )

    # ── Inférence et écriture ─────────────────────────────────────────────────
    out_idx = 0

    def _write_snapshot(t_ns: int, snap_norm: np.ndarray) -> None:
        """Dénormalise et écrit un snapshot (V, L, H, W) dans le store de sortie."""
        nonlocal out_idx
        snap_phys = ns.denormalize_snapshot(snap_norm)  # (V, L, H, W)
        store_out["coords/time"][out_idx] = t_ns
        for vi, var in enumerate(VARIABLE_ORDER):
            store_out[f"pressure/{var}"][out_idx] = snap_phys[vi]
        out_idx += 1

    with torch.no_grad():
        for win_k, (i_s0, i_s1) in enumerate(windows):
            t0_h  = times_h[i_s0]
            t0_ns = times_ns[i_s0]

            # Charger S0 et S1
            snap0 = np.stack(
                [store_in[f"pressure/{var}"][i_s0].astype(np.float32) for var in VARIABLE_ORDER],
                axis=0,
            )  # (V, L, H, W)
            snap1 = np.stack(
                [store_in[f"pressure/{var}"][i_s1].astype(np.float32) for var in VARIABLE_ORDER],
                axis=0,
            )

            snap0_norm = ns.normalize_snapshot(snap0)
            snap1_norm = ns.normalize_snapshot(snap1)

            S0 = torch.from_numpy(snap0_norm).unsqueeze(0).to(dev)  # (1, V, L, H, W)
            S1 = torch.from_numpy(snap1_norm).unsqueeze(0).to(dev)

            # Inférence → (1, T, V, L, H, W)
            preds = model(S0, S1)
            preds_np = preds[0].cpu().numpy()  # (T, V, L, H, W)

            # Écrire S0
            _write_snapshot(int(times_ns[i_s0]), snap0_norm)

            # Écrire les T pas intermédiaires
            for k in range(n_interm):
                t_k_ns = int(
                    (t0_h + (k + 1) * dt_1h).astype("datetime64[ns]").astype(np.int64)
                )
                _write_snapshot(t_k_ns, preds_np[k])

            if (win_k + 1) % 50 == 0 or win_k == len(windows) - 1:
                print(f"  [{win_k + 1}/{len(windows)}] t0={t0_h}")

        # Écrire le dernier S1
        i_last_s1 = windows[-1][1]
        snap_last = np.stack(
            [store_in[f"pressure/{var}"][i_last_s1].astype(np.float32) for var in VARIABLE_ORDER],
            axis=0,
        )
        snap_last_norm = ns.normalize_snapshot(snap_last)
        _write_snapshot(int(times_ns[i_last_s1]), snap_last_norm)

    print(f"\nInférence terminée. {out_idx} pas de temps écrits dans {zarr_out}")


if __name__ == "__main__":
    main()
