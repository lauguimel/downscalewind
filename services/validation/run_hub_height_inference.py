"""Hub-height out-of-sample inference: RAW surrogate v3 vs M_I8-calibrated at masts/turbines.

Derived from `data/validation/crest_deficit/run_masts_M_K2.py` (same forward pass, same
denormalisation) with three changes needed for sites never seen in training:
  * pairings are taken as-is (any height, any hour) from a multiheight-style parquet;
  * grid.zarr inputs are materialised on the fly (require_cached=False) from a local ERA5
    store + DEM tiles + a per-site WorldCover GeoTIFF, into a dedicated cache dir;
  * `--station-filter` / `--hour-stride` bound the cost on a laptop (MPS) or a single GPU.

Config = the M_I8 training YAML (architecture keys) + local path overrides in a small YAML
(see configs/validation/hub_height_*.yaml). Output parquet columns match M_K2 so the same
scoring code works: station_id, timestamp_iso, height_obs, speed_obs, speed_raw, speed_corr,
speed_era5_baseline_patch, u10/v10_era5_baseline_patch.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_PROJECT = Path(__file__).resolve().parents[2]
_SURR = _PROJECT / "services" / "module2b-surrogate"
if str(_SURR) not in sys.path:
    sys.path.insert(0, str(_SURR))

from src.ann_correction import ANNCorrection  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import ObsCenteredDataset, collate_obs_centered  # noqa: E402
from train_v2_devine_style import (  # noqa: E402
    _build_era5_layout, _era5_baseline_uv_at_center, _denorm_uv_at_center,
    _load_norm_overrides, build_frozen_surrogate,
)

logger = logging.getLogger("hub_height")

PAIRING_COLS = ["station_id", "timestamp", "lat", "lon", "elev", "height_obs",
                "speed_obs", "u_obs", "v_obs"]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pairings(parquet: Path, out: Path, station_filter: list[str] | None,
                   hour_stride: int, max_rows: int | None) -> Path:
    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df.timestamp.dt.minute == 0]
    if station_filter:
        df = df[df.station_id.isin(station_filter)]
    if hour_stride > 1:
        df = df[(df.timestamp.dt.hour % hour_stride) == 0]
    df = df.dropna(subset=["speed_obs"])
    df = df[PAIRING_COLS].sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)
    df.to_parquet(out, index=False)
    logger.info("pairings: %d rows | %d stations | %d hours | heights=%s", len(df),
                df.station_id.nunique(), df.timestamp.nunique(),
                sorted(df.height_obs.unique().tolist()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", type=Path, required=True,
                    help="M_I8 training YAML (architecture + norm keys)")
    ap.add_argument("--site-config", type=Path, required=True,
                    help="YAML with local paths: pairings_parquet, era5_store, dem, worldcover, "
                         "cache_dir, surrogate_checkpoint, ann_checkpoint, norm_yaml, out_dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-prep-workers", type=int, default=4)
    ap.add_argument("--station-filter", default=None, help="comma-separated station_ids")
    ap.add_argument("--hour-stride", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: cap pairings")
    ap.add_argument("--profile-heights", default=None,
                    help="comma-separated AGL heights (m): also write speed_raw_h<H>/speed_corr_h<H> "
                         "at the nearest model level, to inspect the predicted vertical profile")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = yaml.safe_load(args.train_config.read_text())
    site = yaml.safe_load(args.site_config.read_text())
    out_dir = Path(site["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(site["norm_yaml"]))}
    agl_name = cfg.get("target_agl_levels", "agl_0_200_32")
    nz = int(parse_agl_levels(agl_name).size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = era5_layout["total_dim"]

    levels = parse_agl_levels(agl_name)
    prof = [float(h) for h in args.profile_heights.split(",")] if args.profile_heights else []
    prof_k = {h: int(abs(levels - h).argmin()) for h in prof}
    if prof:
        logger.info("profile levels: %s", {h: float(levels[k]) for h, k in prof_k.items()})
    stations = args.station_filter.split(",") if args.station_filter else None
    pair_path = build_pairings(Path(site["pairings_parquet"]), out_dir / "pairings.parquet",
                               stations, args.hour_stride, args.max_rows)

    ds = ObsCenteredDataset(
        pair_path,
        era5_store=Path(site["era5_store"]),
        dem=Path(site["dem"]),
        worldcover=Path(site["worldcover"]) if site.get("worldcover") else None,
        cache_dir=Path(site["cache_dir"]),
        norm=norm,
        target_agl_levels=agl_name,
        max_era5_delta_h=float(site.get("max_era5_delta_h", 0.6)),  # hourly ERA5: exact slot
        seed=int(cfg.get("seed", 42)),
        n_workers=args.n_prep_workers,
        require_cached=False,
        enable_phys_features=bool(cfg.get("enable_phys_features", True)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_obs_centered)

    surrogate = build_frozen_surrogate(
        Path(site["surrogate_checkpoint"]), era5_dim=era5_dim, nz=nz,
        terrain_in_channels=4, geo_channels=2,
        preset=cfg.get("surrogate_preset", "base"), device=device,
    )
    ann = ANNCorrection(
        era5_dim=era5_dim, topo_dim=int(cfg.get("topo_dim", 12)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)), zero_init_output=True,
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", True)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
        use_calm_gate=bool(cfg.get("use_calm_gate", False)),
        gate_v0_init=float(cfg.get("gate_v0_init", 2.5)),
        gate_s_init=float(cfg.get("gate_s_init", 1.0)),
        gate_norm=norm,
    ).to(device)
    ck = torch.load(str(site["ann_checkpoint"]), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"]); ann.eval(); surrogate.eval()
    logger.info("device=%s | ANN epoch=%s | nz=%d era5_dim=%d", device, ck.get("epoch", "?"),
                nz, era5_dim)

    rows: list[dict] = []
    t0 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            terrain, era5, geo, topo, speed_obs, k_obs, meta = batch
            terrain = terrain.to(device); era5 = era5.to(device)
            geo = geo.to(device); topo = topo.to(device); k_obs = k_obs.to(device)
            pred_raw = surrogate(terrain, era5, geo)
            era5_corr = ann(era5, topo, terrain=terrain)
            pred_corr = surrogate(terrain, era5_corr, geo)
            ur, vr = _denorm_uv_at_center(pred_raw, norm, k_obs)
            uc, vc = _denorm_uv_at_center(pred_corr, norm, k_obs)
            u0r, v0r = _era5_baseline_uv_at_center(era5, norm, era5_layout)
            u0c, v0c = _era5_baseline_uv_at_center(era5_corr, norm, era5_layout)
            sp_raw = torch.sqrt((ur + u0r) ** 2 + (vr + v0r) ** 2 + 1e-8)
            sp_corr = torch.sqrt((uc + u0c) ** 2 + (vc + v0c) ** 2 + 1e-8)
            sp_base = torch.sqrt(u0r ** 2 + v0r ** 2 + 1e-8)
            prof_out: dict[str, torch.Tensor] = {}
            for h, k in prof_k.items():
                kk = torch.full_like(k_obs, k)
                a, b = _denorm_uv_at_center(pred_raw, norm, kk)
                c, d = _denorm_uv_at_center(pred_corr, norm, kk)
                prof_out[f"speed_raw_h{int(h)}"] = torch.sqrt((a + u0r) ** 2 + (b + v0r) ** 2 + 1e-8)
                prof_out[f"speed_corr_h{int(h)}"] = torch.sqrt((c + u0c) ** 2 + (d + v0c) ** 2 + 1e-8)
            for i, m in enumerate(meta):
                rows.append({
                    "station_id": str(m["station_id"]),
                    "timestamp_iso": str(m["timestamp_iso"]),
                    "height_obs": float(m["height_obs"]),
                    "speed_obs": float(speed_obs[i]),
                    "speed_raw": float(sp_raw[i]),
                    "speed_corr": float(sp_corr[i]),
                    "speed_era5_baseline_patch": float(sp_base[i]),
                    "u10_era5_baseline_patch": float(u0r[i]),
                    "v10_era5_baseline_patch": float(v0r[i]),
                    **{name: float(t[i]) for name, t in prof_out.items()},
                })
            if bi % 50 == 0:
                logger.info("batch %d | rows=%d | %.1fs", bi, len(rows), time.time() - t0)
                pd.DataFrame(rows).to_parquet(out_dir / "predictions_partial.parquet", index=False)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "predictions.parquet", index=False)
    err_raw = df.speed_raw - df.speed_obs
    err_corr = df.speed_corr - df.speed_obs
    err_base = df.speed_era5_baseline_patch - df.speed_obs
    summary = {
        "n_pairings": int(len(df)), "n_stations": int(df.station_id.nunique()),
        "n_hours": int(df.timestamp_iso.nunique()), "wall_s": time.time() - t0,
        "device": device, "ann_checkpoint": str(site["ann_checkpoint"]),
        "ann_epoch": int(ck.get("epoch", -1)),
        "surrogate_checkpoint": str(site["surrogate_checkpoint"]),
        "target_agl_levels": agl_name,
        "raw": {"mae": float(err_raw.abs().mean()), "bias": float(err_raw.mean())},
        "corr": {"mae": float(err_corr.abs().mean()), "bias": float(err_corr.mean())},
        "era5_10m_baseline": {"mae": float(err_base.abs().mean()), "bias": float(err_base.mean())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("DONE %s", json.dumps(summary["raw"]) + " | corr " + json.dumps(summary["corr"]))


if __name__ == "__main__":
    main()
