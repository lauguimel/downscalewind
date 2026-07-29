"""
eval_devine_steep54_windstrat.py — M_I5a: wind-stratified eval of the M_I3
ENCODER correction on the 54 held-out STEEP val stations.

Reuses the exact eval machinery of eval_devine_style.py but:
  - restricts station_filter to the 54 val stations whose pop == "steep"
    (watertight split seed=42, val_frac=0.20, exclude perdigao);
  - reads only from the pre-materialised shared cache (require_cached=True,
    never re-materialises);
  - stratifies per-pairing (speed_obs, speed_corr, speed_raw) by obs wind speed
    into bins <1,1-2,2-3,3-5,5-7,>7: MAE + bias, corrected vs raw, + n;
  - reports the high-wind (>6 m/s) corrected-vs-raw bias closure.

Usage:
    python eval_devine_steep54_windstrat.py \
        --config configs/training/devine_style_M_I3_encoder.yaml \
        --ann-checkpoint data/models/surrogate_v2_devine_M_I3_encoder/best.pt
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
)
from eval_devine_style import _eval_loader  # noqa: E402

logger = logging.getLogger("eval_steep54")

BINS = [(-np.inf, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 3.0, "2-3"),
        (3.0, 5.0, "3-5"), (5.0, 7.0, "5-7"), (7.0, np.inf, ">7")]


def _steep_val_station_ids(pairings_parquet: Path, val_frac: float,
                           seed: int, exclude_substrings) -> list[str]:
    _, val_sids = watertight_station_split(
        pairings_parquet, val_frac=val_frac, seed=seed,
        exclude_substrings=tuple(exclude_substrings),
    )
    val_set = set(map(str, val_sids))
    df = pd.read_parquet(pairings_parquet, columns=["station_id", "pop"])
    df["station_id"] = df["station_id"].astype(str)
    pop_by_sid = df.groupby("station_id")["pop"].agg(
        lambda s: tuple(sorted(set(s))))
    steep = {sid for sid, pops in pop_by_sid.items() if pops == ("steep",)}
    return sorted(val_set & steep)


def _strat_table(obs: np.ndarray, raw: np.ndarray, corr: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi, label in BINS:
        m = (obs >= lo) & (obs < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": label, "n": 0, "mae_raw": None, "mae_corr": None,
                         "bias_raw": None, "bias_corr": None})
            continue
        er, ec = raw[m] - obs[m], corr[m] - obs[m]
        rows.append({
            "bin": label, "n": n,
            "mae_raw": float(np.abs(er).mean()),
            "mae_corr": float(np.abs(ec).mean()),
            "bias_raw": float(er.mean()),
            "bias_corr": float(ec.mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ann-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or (Path(cfg["output_dir"]) / "steep54_windstrat")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda")
    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(target_agl_levels.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = era5_layout["total_dim"]

    pairings = Path(cfg["pairings_parquet"])
    steep_val_sids = _steep_val_station_ids(
        pairings,
        val_frac=float(cfg.get("val_frac", 0.2)),
        seed=int(cfg.get("seed", 42)),
        exclude_substrings=cfg.get("exclude_substrings", ["perdigao"]),
    )
    logger.info("steep-54 held-out val stations: n=%d", len(steep_val_sids))

    val_ds = ObsCenteredDataset(
        pairings,
        station_filter=steep_val_sids,
        era5_store=Path(cfg["era5_store"]),
        dem=Path(cfg["dem"]),
        worldcover=Path(cfg["worldcover"]) if cfg.get("worldcover") else None,
        cache_dir=Path(cfg["cache_dir"]),
        norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)),
        n_workers=int(cfg.get("n_prep_workers", 4)),
        overwrite_cache=False,
        require_cached=True,  # NEVER re-materialise; read pre-built cache only
        enable_phys_features=bool(cfg.get("enable_phys_features", False)),
    )
    bs = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate_obs_centered, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    surrogate = build_frozen_surrogate(
        Path(cfg["surrogate_checkpoint"]),
        era5_dim=era5_dim, nz=nz,
        terrain_in_channels=4, geo_channels=2,
        preset=cfg.get("surrogate_preset", "base"),
        device=device,
    )

    ann = ANNCorrection(
        era5_dim=era5_dim, topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)),
        zero_init_output=True,
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", False)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
    ).to(device)
    ck = torch.load(str(args.ann_checkpoint), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"])
    logger.info("Loaded ANN from %s (epoch=%d)", args.ann_checkpoint, ck.get("epoch", -1))

    t0 = time.time()
    res_raw = _eval_loader(None, surrogate, val_loader, norm, era5_layout, device)
    res_corr = _eval_loader(ann, surrogate, val_loader, norm, era5_layout, device)
    wall = time.time() - t0

    obs = res_raw["obs"]
    raw = res_raw["pred"]
    corr = res_corr["pred"]

    # Per-pairing parquet
    rows = []
    for m, o, p_raw, p_corr in zip(res_raw["meta"], obs, raw, corr):
        rows.append({
            "station_id": m["station_id"],
            "timestamp_iso": m["timestamp_iso"],
            "source": m["source"],
            "height_obs": m["height_obs"],
            "speed_obs": float(o),
            "speed_pred_raw": float(p_raw),
            "speed_pred_corr": float(p_corr),
        })
    df_pair = pd.DataFrame(rows)
    df_pair.to_parquet(out_dir / "eval_pairings_steep54.parquet", index=False)

    strat = _strat_table(obs, raw, corr)

    # High-wind closure (>6 m/s)
    hw = obs > 6.0
    hw_n = int(hw.sum())
    hw_block = {
        "n": hw_n,
        "bias_raw": float((raw[hw] - obs[hw]).mean()) if hw_n else None,
        "bias_corr": float((corr[hw] - obs[hw]).mean()) if hw_n else None,
        "mae_raw": float(np.abs(raw[hw] - obs[hw]).mean()) if hw_n else None,
        "mae_corr": float(np.abs(corr[hw] - obs[hw]).mean()) if hw_n else None,
    }

    summary = {
        "n_steep54_stations": len(steep_val_sids),
        "n_pairings": int(obs.size),
        "wall_s": wall,
        "overall": {
            "raw":  {"mae": res_raw["mae"], "bias": res_raw["bias"]},
            "corr": {"mae": res_corr["mae"], "bias": res_corr["bias"]},
        },
        "wind_strat": strat,
        "highwind_gt6": hw_block,
    }
    (out_dir / "steep54_windstrat_summary.json").write_text(json.dumps(summary, indent=2))

    # Pretty print
    print("\n=== STEEP-54 held-out val | M_I3 encoder | wind-stratified ===")
    print(f"n_stations={len(steep_val_sids)}  n_pairings={obs.size}  wall={wall:.1f}s")
    print(f"OVERALL  raw MAE={res_raw['mae']:.3f} bias={res_raw['bias']:+.3f}  | "
          f"corr MAE={res_corr['mae']:.3f} bias={res_corr['bias']:+.3f}")
    print(f"{'bin':>5} {'n':>7} | {'MAE_raw':>8} {'MAE_corr':>9} | "
          f"{'bias_raw':>9} {'bias_corr':>10}")
    for r in strat:
        if r["n"] == 0:
            print(f"{r['bin']:>5} {0:>7} |  (empty)")
            continue
        print(f"{r['bin']:>5} {r['n']:>7} | {r['mae_raw']:>8.3f} {r['mae_corr']:>9.3f} | "
              f"{r['bias_raw']:>+9.3f} {r['bias_corr']:>+10.3f}")
    print(f"\nHIGH-WIND >6 m/s (n={hw_n}): "
          f"bias raw={hw_block['bias_raw']:+.3f} -> corr={hw_block['bias_corr']:+.3f}  | "
          f"MAE raw={hw_block['mae_raw']:.3f} -> corr={hw_block['mae_corr']:.3f}")
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
