"""
eval_devine_style.py — Phase H' M_H'0: Evaluate the DEVINE-style ANN correction
against the raw frozen surrogate on a validation parquet.

Outputs:
    - <out_dir>/eval_summary.json     overall MAE/RMSE/bias raw vs corrected
    - <out_dir>/eval_pairings.parquet per-pairing (station_id, ts, speed_obs,
                                                    speed_pred_raw, speed_pred_corr)

Usage:
    python eval_devine_style.py --config configs/training/devine_style_smoke.yaml \\
        --ann-checkpoint <out_dir>/best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from src.ann_correction import ANNCorrection  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import (  # noqa: E402
    ObsCenteredDataset,
    collate_obs_centered,
    watertight_station_split,
)
from train_v2_devine_style import (  # noqa: E402
    _build_era5_layout,
    _era5_baseline_uv_at_center,
    _denorm_uv_at_center,
    _load_norm_overrides,
    build_frozen_surrogate,
    resolve_pairings,
)

logger = logging.getLogger("eval_devine")


def _eval_loader(
    ann: torch.nn.Module | None,
    surrogate: torch.nn.Module,
    loader: DataLoader,
    norm: dict,
    era5_layout: dict,
    device: str,
) -> dict:
    use_ann = ann is not None
    obs_l: list[float] = []
    pred_l: list[float] = []
    meta_l: list[dict] = []
    if use_ann:
        ann.eval()
    surrogate.eval()
    with torch.no_grad():
        for batch in loader:
            terrain, era5, geo, topo, speed_obs, k_obs, meta = batch
            terrain = terrain.to(device, non_blocking=True)
            era5 = era5.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            topo = topo.to(device, non_blocking=True)
            speed_obs = speed_obs.to(device, non_blocking=True)
            k_obs = k_obs.to(device, non_blocking=True)
            era5_in = ann(era5, topo, terrain=terrain) if use_ann else era5
            pred = surrogate(terrain, era5_in, geo)
            u_res, v_res = _denorm_uv_at_center(pred, norm, k_obs)
            u10_b, v10_b = _era5_baseline_uv_at_center(era5_in, norm, era5_layout)
            u_pred = u_res + u10_b
            v_pred = v_res + v10_b
            speed_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2 + 1e-8)
            obs_l.extend(speed_obs.cpu().numpy().tolist())
            pred_l.extend(speed_pred.cpu().numpy().tolist())
            meta_l.extend(meta)
    obs = np.asarray(obs_l, dtype=np.float32)
    pred = np.asarray(pred_l, dtype=np.float32)
    err = pred - obs
    return {
        "n": int(obs.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "obs": obs,
        "pred": pred,
        "meta": meta_l,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ann-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda")
    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(target_agl_levels.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = era5_layout["total_dim"]

    # Val split (same as train — including the M_I7 merged multi-pop parquet)
    pairings = resolve_pairings(cfg, out_dir)
    _, val_sids = watertight_station_split(
        pairings, val_frac=float(cfg.get("val_frac", 0.2)),
        seed=int(cfg.get("seed", 42)),
        exclude_substrings=tuple(cfg.get("exclude_substrings", ["perdigao"])),
    )
    n_smoke_stations = cfg.get("max_stations")
    if n_smoke_stations is not None:
        n_train = max(1, int(n_smoke_stations * (1.0 - cfg.get("val_frac", 0.2))))
        n_val = max(1, int(n_smoke_stations - n_train))
        val_sids = val_sids[:n_val]

    val_ds = ObsCenteredDataset(
        pairings,
        station_filter=val_sids,
        max_pairings=cfg.get("max_val_pairings"),
        era5_store=Path(cfg["era5_store"]),
        dem=Path(cfg["dem"]),
        worldcover=Path(cfg["worldcover"]) if cfg.get("worldcover") else None,
        cache_dir=Path(cfg["cache_dir"]),
        norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)),
        n_workers=int(cfg.get("n_prep_workers", 4)),
        enable_phys_features=bool(cfg.get("enable_phys_features", False)),
    )
    bs = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate_obs_centered, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Surrogate v2 frozen
    surrogate = build_frozen_surrogate(
        Path(cfg["surrogate_checkpoint"]),
        era5_dim=era5_dim, nz=nz,
        terrain_in_channels=4, geo_channels=2,
        preset=cfg.get("surrogate_preset", "base"),
        device=device,
    )

    # ANN
    ann = ANNCorrection(
        era5_dim=era5_dim, topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)),
        zero_init_output=True,
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", False)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
        use_calm_gate=bool(cfg.get("use_calm_gate", False)),
        gate_v0_init=float(cfg.get("gate_v0_init", 2.5)),
        gate_s_init=float(cfg.get("gate_s_init", 1.0)),
        gate_norm=norm,
    ).to(device)
    ck = torch.load(str(args.ann_checkpoint), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"])
    logger.info("Loaded ANN from %s (epoch=%d)", args.ann_checkpoint, ck.get("epoch", -1))

    # Eval RAW (no ANN)
    t0 = time.time()
    res_raw = _eval_loader(None, surrogate, val_loader, norm, era5_layout, device)
    # Eval CORRECTED
    res_corr = _eval_loader(ann, surrogate, val_loader, norm, era5_layout, device)
    wall = time.time() - t0

    summary = {
        "n_val_pairings": res_raw["n"],
        "wall_s": wall,
        "raw":  {"mae": res_raw["mae"],  "rmse": res_raw["rmse"],  "bias": res_raw["bias"]},
        "corr": {"mae": res_corr["mae"], "rmse": res_corr["rmse"], "bias": res_corr["bias"]},
        "delta_mae":  res_corr["mae"]  - res_raw["mae"],
        "delta_rmse": res_corr["rmse"] - res_raw["rmse"],
        "delta_bias": res_corr["bias"] - res_raw["bias"],
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for m, o, p_raw, p_corr in zip(
        res_raw["meta"], res_raw["obs"], res_raw["pred"], res_corr["pred"]
    ):
        rows.append({
            "station_id": m["station_id"],
            "timestamp_iso": m["timestamp_iso"],
            "source": m["source"],
            "height_obs": m["height_obs"],
            "speed_obs": float(o),
            "speed_pred_raw": float(p_raw),
            "speed_pred_corr": float(p_corr),
        })
    pd.DataFrame(rows).to_parquet(out_dir / "eval_pairings.parquet", index=False)
    logger.info("eval done: %s", summary)


if __name__ == "__main__":
    main()
