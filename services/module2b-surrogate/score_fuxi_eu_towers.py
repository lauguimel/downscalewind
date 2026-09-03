"""FuXi-CFD EU-towers benchmark: FuXi vs our ANN+surrogate v2 vs ERA5 at the
ICOS tall towers used in the FuXi-CFD paper EU validation (OPE, IPR; JJA 2020).

Purpose: neutralise the OOD critique of the main head-to-head (our val stations
are "away" for FuXi, trained on SE China) by also scoring on THEIR EU
validation sites. Torfhaus (their 3rd EU site) has no publicly ingestable obs
and is not covered.

Per-height like-for-like:
  - FuXi: output level round((h-10)/5)   (levels 10..140 m step 5).
  - Ours: height_obs -> k_obs on the AGL grid (0..100 m, 24 levels).
  - ERA5: native levels only — u10/v10 at 10 m, u100/v100 at 100 m; NaN
    elsewhere (no log-profile extrapolation).

Reuses score_fuxi_vs_ours.score_ours and fuxicfd_infer_at_stations verbatim.

Usage (Aqua, env fuxicfd — see configs/hpc/eval_fuxi_eu_towers.pbs).
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
import yaml
import zarr

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

import fuxicfd_infer_at_stations as FX  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM  # noqa: E402
from train_v2_devine_style import _load_norm_overrides  # noqa: E402
from score_fuxi_vs_ours import metrics_block, sample_pairings, score_ours  # noqa: E402

logger = logging.getLogger("score_fuxi_eu_towers")

FUXI_LEVEL_STEP = 5.0
FUXI_LEVEL0_H = 10.0


def fuxi_level_for_height(h: float) -> int:
    idx = int(round((h - FUXI_LEVEL0_H) / FUXI_LEVEL_STEP))
    if not 0 <= idx <= 26:
        raise ValueError(f"height {h} m outside FuXi output range 10..140 m")
    return idx


def score_fuxi_towers(sample_df: pd.DataFrame, *, onnx: Path, scaler_dir: Path,
                      dem: Path, worldcover: Path, store_100m: Path,
                      device: str, batch_size: int, max_delta_h: float,
                      out_dir: Path) -> pd.DataFrame:
    runner = FX.FuxiRunner(onnx, scaler_dir, device=device)
    terr_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    n_skip = 0

    partial_path = out_dir / "fuxi_partial.parquet"
    done: set[tuple[str, str]] = set()
    if partial_path.exists():
        prev = pd.read_parquet(partial_path)
        rows = prev.to_dict("records")
        done = {(str(r["station_id"]), str(r["timestamp_iso"])) for r in rows}
        logger.info("FuXi resume: %d pairings already scored", len(done))

    df = sample_df.reset_index(drop=True)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        xs, metas, levels = [], [], []
        for _, row in chunk.iterrows():
            sid = str(row["station_id"])
            ts = pd.Timestamp(row["timestamp"])
            if (sid, ts.isoformat()) in done:
                continue
            lat, lon = float(row["lat"]), float(row["lon"])
            try:
                # station_id keys the terrain cache: same coords for all heights
                site = sid.split("_h")[0]
                if site in terr_cache:
                    dem301, rough301 = terr_cache[site]
                else:
                    dem301 = FX.extract_fuxi_dem(dem, lat, lon)
                    rough301 = FX.extract_fuxi_roughness(worldcover, lat, lon)
                    terr_cache[site] = (dem301, rough301)
                ts_ns = int(np.datetime64(ts.to_datetime64(), "ns").astype("int64"))
                u9, v9, dh, _ = FX.uv100_from_era5_native(store_100m, lat, lon, ts_ns,
                                                          max_delta_h=max_delta_h)
                xs.append(runner.build_input(u9, v9, dem301, rough301))
                levels.append(fuxi_level_for_height(float(row["height_obs"])))
                metas.append({"sid": sid, "ts": ts, "dh": dh})
            except Exception as exc:
                n_skip += 1
                logger.debug("FuXi skip %s @ %s: %s", sid, ts, exc)
        if not xs:
            continue
        speeds = runner.forward_speed_centre(np.concatenate(xs, axis=0), levels)
        for meta, sp in zip(metas, speeds):
            rows.append({"station_id": meta["sid"],
                         "timestamp_iso": meta["ts"].isoformat(),
                         "speed_fuxi": float(sp),
                         "uv100_delta_h": float(meta["dh"])})
        pd.DataFrame(rows).to_parquet(partial_path, index=False)
        logger.info("FuXi towers: %d scored (%d skipped)", len(rows), n_skip)
    return pd.DataFrame(rows)


def era5_native_speed(store: Path, uvar: str, vvar: str, rows: pd.DataFrame,
                      max_delta_h: float) -> pd.Series:
    """Bilinear u/v at native level for `rows`; nearest time <= max_delta_h."""
    g = zarr.open_group(str(store), mode="r")
    times = np.array(g["coords/time"]).astype("datetime64[ns]")
    lats = np.array(g["coords/lat"], dtype=np.float64)
    lons = np.array(g["coords/lon"], dtype=np.float64)
    u = np.array(g[f"surface/{uvar}"]); v = np.array(g[f"surface/{vvar}"])
    order = np.argsort(lats)
    la = lats[order]
    out = []
    for _, r in rows.iterrows():
        ts = np.datetime64(pd.Timestamp(r["timestamp"]).to_datetime64(), "ns")
        k = int(np.argmin(np.abs(times - ts)))
        if abs((times[k] - ts) / np.timedelta64(1, "h")) > max_delta_h:
            out.append(np.nan); continue
        lat, lon = float(r["lat"]), float(r["lon"])
        i = int(np.clip(np.searchsorted(la, lat) - 1, 0, len(la) - 2))
        j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons) - 2))
        ty = float(np.clip((lat - la[i]) / (la[i + 1] - la[i]), 0, 1))
        tx = float(np.clip((lon - lons[j]) / (lons[j + 1] - lons[j]), 0, 1))
        vals = []
        for f in (u[k][..., order, :], v[k][..., order, :]):
            vals.append((1 - ty) * ((1 - tx) * f[i, j] + tx * f[i, j + 1])
                        + ty * ((1 - tx) * f[i + 1, j] + tx * f[i + 1, j + 1]))
        out.append(float(np.hypot(vals[0], vals[1])))
    return pd.Series(out, index=rows.index)


def build_towers_table(uni: pd.DataFrame) -> dict:
    table = {"overall": metrics_block(uni), "per_series": {}, "wind_class": {}}
    for sid in sorted(uni["station_id"].unique()):
        table["per_series"][sid] = metrics_block(uni[uni["station_id"] == sid])
    for name, mask in [("lt3", uni["speed_obs"] < 3.0),
                       ("3to6", (uni["speed_obs"] >= 3.0) & (uni["speed_obs"] <= 6.0)),
                       ("gt6", uni["speed_obs"] > 6.0)]:
        table["wind_class"][name] = metrics_block(uni[mask])
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ann-checkpoint", type=Path, default=None)
    ap.add_argument("--pairings", type=Path, required=True,
                    help="icos_eu_towers.parquet (build_icos_eu_pairings.py)")
    ap.add_argument("--era5-store", type=Path, required=True,
                    help="era5_hourly_jja2020_eu.zarr (pressure+surface, ours)")
    ap.add_argument("--store-100m", type=Path, required=True,
                    help="era5_100m_jja2020eu.zarr (FuXi input + 100 m baseline)")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--scaler-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-station", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fuxi-device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fuxi-batch-size", type=int, default=16)
    ap.add_argument("--max-delta-h", type=float, default=1.5)
    ap.add_argument("--n-prep-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("rasterio", "rasterio._env", "rasterio.env", "fiona", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = yaml.safe_load(args.config.read_text())
    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    ann_ckpt = args.ann_checkpoint or (Path(cfg["output_dir"]) / "best.pt")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_parquet(args.pairings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sample = sample_pairings(df, args.n_per_station, args.seed)
    sample.to_parquet(args.out_dir / "sample_pairings.parquet", index=False)
    logger.info("SAMPLE: %d pairings, %d series", len(sample),
                sample["station_id"].nunique())

    fuxi_df = score_fuxi_towers(
        sample, onnx=args.onnx, scaler_dir=args.scaler_dir,
        dem=Path(cfg["dem"]), worldcover=Path(cfg["worldcover"]),
        store_100m=args.store_100m, device=args.fuxi_device,
        batch_size=args.fuxi_batch_size, max_delta_h=args.max_delta_h,
        out_dir=args.out_dir)

    ours_df = score_ours(
        cfg, norm, sample, device=args.device, ann_ckpt=Path(ann_ckpt),
        batch_size=args.batch_size, n_prep=args.n_prep_workers,
        cache_dir=args.cache_dir, era5_store=args.era5_store,
        require_cached=False, tag="eu_towers")
    ours_df.to_parquet(args.out_dir / "ours_scores.parquet", index=False)

    sample["timestamp_iso"] = sample["timestamp"].map(lambda x: pd.Timestamp(x).isoformat())
    sample["speed_era5"] = np.nan
    m10 = sample["height_obs"] == 10.0
    m100 = sample["height_obs"] == 100.0
    if m10.any():
        sample.loc[m10, "speed_era5"] = era5_native_speed(
            args.era5_store, "u10", "v10", sample[m10], args.max_delta_h)
    if m100.any():
        sample.loc[m100, "speed_era5"] = era5_native_speed(
            args.store_100m, "u100", "v100", sample[m100], args.max_delta_h)

    uni = sample.merge(fuxi_df, on=["station_id", "timestamp_iso"], how="left")
    uni = uni.merge(ours_df, on=["station_id", "timestamp_iso"], how="left")
    cols = ["station_id", "timestamp_iso", "lat", "lon", "elev", "height_obs",
            "speed_obs", "speed_fuxi", "speed_ours", "speed_era5", "uv100_delta_h"]
    uni = uni[[c for c in cols if c in uni.columns]]
    uni.to_parquet(args.out_dir / "unified_scores.parquet", index=False)

    common = uni[uni["speed_fuxi"].notna() & uni["speed_ours"].notna()
                 & uni["speed_obs"].notna()].copy()
    report = {
        "wall_s": round(time.time() - t0, 1),
        "n_sample": int(len(sample)),
        "n_fuxi": int(uni["speed_fuxi"].notna().sum()),
        "n_ours": int(uni["speed_ours"].notna().sum()),
        "n_era5": int(uni["speed_era5"].notna().sum()),
        "era5_native_levels_only": "10 m (u10/v10) and 100 m (u100/v100)",
        "table_common_subset": build_towers_table(common),
        "table_full_coverage": build_towers_table(uni),
    }
    (args.out_dir / "eu_towers_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["table_common_subset"]["per_series"], indent=2))
    print(f"\nwrote {args.out_dir}/unified_scores.parquet + eu_towers_report.json",
          flush=True)


if __name__ == "__main__":
    main()
