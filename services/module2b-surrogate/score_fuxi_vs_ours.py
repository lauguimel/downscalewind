"""
score_fuxi_vs_ours.py — Head-to-head: OFFICIAL FuXi-CFD vs OUR M_I5 DEVINE-style
surrogate vs ERA5, ALL scored against the SAME real station obs.

This is the deliverable scorer. It:
  1. Reproduces the reference held-out VAL split (106 stations incl. 54 steep)
     from combined_steep_plain_v2.parquet via watertight_station_split(seed=42,
     exclude perdigao) + adds Perdigao IOP pairings (perdigao_obs.zarr).
  2. Samples ~N timestamps/station stratified across the available seasons
     (+ a Perdigao IOP sample). The SAME pairings feed FuXi, our model, ERA5
     and obs (apples-to-apples).
  3. Scores FuXi at each pairing's centre 10 m pixel with --uv100-source
     era5_native, routing the 100 m ERA5 store BY MONTH (winter2223=DJF spillover,
     mam/jja/son 2023, era5_100m_perdigao2017 for Perdigao). Terrain 300x300 is
     reprojected ONCE PER STATION (timestamp-independent) and cached.
  4. Scores OUR M_I5 regime model (ANN + frozen surrogate) on the SAME sample
     using the pre-materialised grid.zarr cache (no re-materialise).
  5. Joins ERA5 baseline + obs (already in the combined parquet for val; computed
     for Perdigao) and writes a unified parquet:
       station_id, timestamp, lat, lon, elev, pop, season, speed_obs,
       speed_fuxi, speed_ours, speed_era5.
  6. Prints the MAE/bias table vs obs (overall, by wind class, steep, Perdigao).

NOT a training script; does not modify the surrogate/ANN code. Read-only on all
models. Run on Aqua (conda fuxicfd), GPU for our model, CPU/GPU for FuXi ONNX.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

import fuxicfd_infer_at_stations as FX  # noqa: E402
from src.ann_correction import ANNCorrection  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import (  # noqa: E402
    ObsCenteredDataset,
    collate_obs_centered,
    watertight_station_split,
)
from train_v2_devine_style import (  # noqa: E402
    _build_era5_layout,
    _denorm_uv_at_center,
    _era5_baseline_uv_at_center,
    _load_norm_overrides,
    build_frozen_surrogate,
)

logger = logging.getLogger("score_fuxi_vs_ours")


# ─── 100 m ERA5 store routing by MONTH (not by season label) ─────────────────
# The combined parquet `season` label has edge spillover months; the native
# 100 m stores have tight time ranges, so route by the actual pairing month.
def store_for_month(stores_root: Path, year: int, month: int) -> Path | None:
    if month in (3, 4, 5):
        tag = "mam2023"
    elif month in (6, 7, 8):
        tag = "jja2023"
    elif month in (9, 10, 11):
        tag = "son2023"
    elif month in (12, 1, 2):
        tag = "winter2223"
    else:
        return None
    return stores_root / f"era5_100m_{tag}.zarr"


# ─── Sampling: ~N timestamps/station stratified across seasons ───────────────
def sample_pairings(df: pd.DataFrame, n_per_station: int, seed: int,
                    season_col: str = "season") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for sid, sub in df.groupby("station_id", sort=True):
        if len(sub) <= n_per_station:
            out.append(sub)
            continue
        # Stratify across seasons present for this station.
        if season_col in sub.columns and sub[season_col].nunique() > 1:
            seasons = sorted(sub[season_col].unique())
            per = max(1, n_per_station // len(seasons))
            picks = []
            for s in seasons:
                ss = sub[sub[season_col] == s]
                k = min(per, len(ss))
                picks.append(ss.sample(n=k, random_state=int(rng.integers(1 << 31))))
            picked = pd.concat(picks)
            if len(picked) < n_per_station:
                remain = sub.drop(picked.index)
                extra = remain.sample(n=min(n_per_station - len(picked), len(remain)),
                                      random_state=int(rng.integers(1 << 31)))
                picked = pd.concat([picked, extra])
            out.append(picked.head(n_per_station))
        else:
            out.append(sub.sample(n=n_per_station, random_state=int(rng.integers(1 << 31))))
    return pd.concat(out).reset_index(drop=True)


# ─── Perdigao IOP pairings (reuse audit logic, 10 m) ─────────────────────────
def build_perdigao_sample(obs_zarr: Path, n_per_station: int, seed: int,
                          height_target: float = 10.0) -> pd.DataFrame:
    g = zarr.open_group(str(obs_zarr), mode="r")

    def _dec(v):
        return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)

    sids = [_dec(x) for x in g["coords/site_id"][:]]
    lats = np.asarray(g["coords/lat"][:], dtype=np.float64)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float64)
    elevs = np.asarray(g["coords/altitude_m"][:], dtype=np.float64)
    heights = np.asarray(g["coords/height_m"][:], dtype=np.float64)
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    h_idx = int(np.argmin(np.abs(heights - height_target)))
    u = np.asarray(g["sites/u"][:], dtype=np.float32)
    v = np.asarray(g["sites/v"][:], dtype=np.float32)
    # infer axes (time, site, height)
    shape = u.shape
    axes = None
    for ta in range(3):
        for sa in range(3):
            for ha in range(3):
                if len({ta, sa, ha}) != 3:
                    continue
                if shape[ta] == len(times) and shape[sa] == len(sids) and shape[ha] == len(heights):
                    axes = (ta, sa, ha)
    u = np.moveaxis(u, axes, (0, 1, 2))
    v = np.moveaxis(v, axes, (0, 1, 2))
    speed = np.hypot(u[:, :, h_idx], v[:, :, h_idx])
    ts = pd.to_datetime(np.array(times).astype("datetime64[ns]"))
    period = (ts >= pd.Timestamp("2017-05-01")) & (ts <= pd.Timestamp("2017-06-30 23:59:59"))
    rng = np.random.default_rng(seed)
    rows = []
    for s_idx in range(len(sids)):
        valid = np.flatnonzero(period & np.isfinite(speed[:, s_idx]) & (speed[:, s_idx] > 0.0))
        if valid.size == 0:
            continue
        if valid.size > n_per_station:
            stride = max(1, valid.size // n_per_station)
            valid = valid[::stride][:n_per_station]
        for t_idx in valid:
            rows.append({
                "station_id": f"perdigao_{sids[s_idx]}"[:24],
                "timestamp": pd.Timestamp(ts[t_idx]),
                "lat": float(lats[s_idx]), "lon": float(lons[s_idx]),
                "elev": float(elevs[s_idx]), "height_obs": float(heights[h_idx]),
                "speed_obs": float(speed[t_idx, s_idx]),
                "u_obs": float(u[t_idx, s_idx, h_idx]), "v_obs": float(v[t_idx, s_idx, h_idx]),
                "season": "perdigao_iop", "pop": "perdigao",
            })
    return pd.DataFrame(rows)


# ─── OUR M_I5 model on cached pairings ───────────────────────────────────────
def score_ours(cfg: dict, norm: dict, sample_df: pd.DataFrame, *, device: str,
               ann_ckpt: Path, batch_size: int, n_prep: int,
               cache_dir: Path | None = None, era5_store: Path | None = None,
               require_cached: bool = True, tag: str = "val") -> pd.DataFrame:
    target_agl = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(target_agl.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = int(era5_layout["total_dim"])

    # Write a pairings parquet with iso timestamps for the dataset.
    tmp = Path("tmp"); tmp.mkdir(exist_ok=True)
    pq = tmp / f"score_ours_pairings_{tag}.parquet"
    sdf = sample_df.copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"]).map(lambda x: x.isoformat())
    sdf.to_parquet(pq, index=False)

    cache_dir = Path(cache_dir) if cache_dir is not None else Path(cfg["cache_dir"])
    era5_store = Path(era5_store) if era5_store is not None else Path(cfg["era5_store"])
    ds = ObsCenteredDataset(
        pq,
        era5_store=era5_store, dem=Path(cfg["dem"]),
        worldcover=Path(cfg["worldcover"]) if cfg.get("worldcover") else None,
        cache_dir=cache_dir, norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)), n_workers=n_prep,
        overwrite_cache=False, require_cached=require_cached,
        enable_phys_features=bool(cfg.get("enable_phys_features", False)),
    )
    logger.info("OURS[%s] dataset: %d pairings (of %d sampled)", tag, len(ds), len(sdf))
    if len(ds) == 0:
        return pd.DataFrame()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_obs_centered)
    surrogate = build_frozen_surrogate(
        Path(cfg["surrogate_checkpoint"]), era5_dim=era5_dim, nz=nz,
        terrain_in_channels=4, geo_channels=int(cfg.get("geo_channels", 2)),
        preset=cfg.get("surrogate_preset", "base"), device=device)
    ann = ANNCorrection(
        era5_dim=era5_dim, topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)), zero_init_output=True,
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", False)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
    ).to(device)
    ck = torch.load(str(ann_ckpt), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"]); ann.eval()
    logger.info("OURS ANN loaded %s (epoch=%s)", ann_ckpt, ck.get("epoch", "?"))

    rows = []
    with torch.inference_mode():
        for batch in loader:
            terrain, era5, geo, topo, speed_obs, k_obs, meta = batch
            terrain = terrain.to(device); era5 = era5.to(device); geo = geo.to(device)
            topo = topo.to(device); k_obs = k_obs.to(device)
            era5_corr = ann(era5, topo, terrain=terrain)
            pred_corr = surrogate(terrain, era5_corr, geo)
            u_res, v_res = _denorm_uv_at_center(pred_corr, norm, k_obs)
            u10, v10 = _era5_baseline_uv_at_center(era5_corr, norm, era5_layout)
            u = u_res + u10; v = v_res + v10
            speed = torch.sqrt(u * u + v * v + 1e-8).detach().cpu().numpy()
            for i, m in enumerate(meta):
                rows.append({
                    "station_id": str(m["station_id"]),
                    "timestamp_iso": str(m["timestamp_iso"]),
                    "speed_ours": float(speed[i]),
                })
    return pd.DataFrame(rows)


# ─── FuXi on the SAME sample, month-routed 100 m store ───────────────────────
def score_fuxi(sample_df: pd.DataFrame, *, onnx: Path, scaler_dir: Path,
               dem: Path, worldcover: Path, stores_root: Path, device: str,
               batch_size: int, max_delta_h: float,
               out_dir: Path | None = None,
               checkpoint_every: int = 200) -> pd.DataFrame:
    runner = FX.FuxiRunner(onnx, scaler_dir, device=device)
    # per-STATION terrain cache (timestamp-independent): key = station_id.
    terr_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    n_ok = n_skip = 0
    t0 = time.time()
    df = sample_df.reset_index(drop=True)

    # Incremental checkpointing: flush accumulated rows every `checkpoint_every`
    # processed pairings to <out_dir>/pairings_partial.parquet (overwrite).
    partial_path = (out_dir / "pairings_partial.parquet") if out_dir is not None else None
    # Restart: skip pairings already present in pairings_partial.parquet.
    done_keys: set[tuple[str, str]] = set()
    if partial_path is not None and partial_path.exists():
        try:
            prev = pd.read_parquet(partial_path)
            if {"station_id", "timestamp_iso"}.issubset(prev.columns):
                rows = prev.to_dict("records")
                done_keys = {(str(r["station_id"]), str(r["timestamp_iso"])) for r in rows}
                n_ok = len(rows)
                logger.info("FuXi resume: %d pairings already in %s",
                            len(done_keys), partial_path)
        except Exception as exc:
            logger.warning("FuXi could not read partial %s: %s", partial_path, exc)
    if done_keys:
        keys = list(zip(df["station_id"].astype(str),
                        df["timestamp"].map(lambda x: pd.Timestamp(x).isoformat())))
        mask = [k not in done_keys for k in keys]
        df = df[mask].reset_index(drop=True)
        logger.info("FuXi resume: %d pairings remaining to score", len(df))

    n_processed = 0
    last_flush = 0

    def _flush():
        if partial_path is not None and rows:
            pd.DataFrame(rows).to_parquet(partial_path, index=False)

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        xs, metas = [], []
        for _, row in chunk.iterrows():
            sid = str(row["station_id"]); lat = float(row["lat"]); lon = float(row["lon"])
            ts = pd.Timestamp(row["timestamp"])
            try:
                if sid in terr_cache:
                    dem301, rough301 = terr_cache[sid]
                else:
                    dem301 = FX.extract_fuxi_dem(dem, lat, lon)
                    rough301 = FX.extract_fuxi_roughness(worldcover, lat, lon)
                    terr_cache[sid] = (dem301, rough301)
                # route 100 m store
                if str(row.get("pop", "")) == "perdigao" or sid.startswith("perdigao"):
                    store = stores_root / "era5_100m_perdigao2017.zarr"
                else:
                    store = store_for_month(stores_root, ts.year, ts.month)
                if store is None or not store.exists():
                    raise FileNotFoundError(f"no 100m store for {ts}")
                ts_ns = int(np.datetime64(ts.to_datetime64(), "ns").astype("int64"))
                u9, v9, dh, lab = FX.uv100_from_era5_native(store, lat, lon, ts_ns,
                                                            max_delta_h=max_delta_h)
                xs.append(runner.build_input(u9, v9, dem301, rough301))
                metas.append({"row": row, "dh": dh})
            except Exception as exc:
                n_skip += 1
                logger.debug("FuXi skip %s @ %s: %s", sid, row["timestamp"], exc)
        if not xs:
            continue
        x_batch = np.concatenate(xs, axis=0)
        speeds = runner.forward_speed10_centre(x_batch)
        for meta, sp in zip(metas, speeds):
            row = meta["row"]
            rows.append({
                "station_id": str(row["station_id"]),
                "timestamp_iso": pd.Timestamp(row["timestamp"]).isoformat(),
                "speed_fuxi": float(sp),
                "uv100_delta_h": float(meta["dh"]),
            })
            n_ok += 1
        n_processed += len(chunk)
        # Flush partial results every `checkpoint_every` processed pairings.
        if n_processed - last_flush >= checkpoint_every:
            _flush()
            last_flush = n_processed
            logger.info("FuXi checkpoint: flushed %d rows to %s", len(rows), partial_path)
        if (start // batch_size) % 20 == 0:
            logger.info("FuXi [%d/%d] ok=%d skip=%d stations_cached=%d elapsed=%.0fs",
                        min(start + batch_size, len(df)), len(df), n_ok, n_skip,
                        len(terr_cache), time.time() - t0)
    _flush()  # final flush of accumulated rows
    logger.info("FuXi DONE ok=%d skip=%d stations=%d elapsed=%.0fs",
                n_ok, n_skip, len(terr_cache), time.time() - t0)
    return pd.DataFrame(rows)


# ─── Metrics table ───────────────────────────────────────────────────────────
def _mae(p, o):
    p = np.asarray(p, float); o = np.asarray(o, float)
    m = np.isfinite(p) & np.isfinite(o)
    return float(np.abs(p[m] - o[m]).mean()) if m.any() else float("nan")


def _bias(p, o):
    p = np.asarray(p, float); o = np.asarray(o, float)
    m = np.isfinite(p) & np.isfinite(o)
    return float((p[m] - o[m]).mean()) if m.any() else float("nan")


def metrics_block(sub: pd.DataFrame) -> dict:
    o = sub["speed_obs"]
    out = {"n": int(len(sub))}
    for name, col in [("fuxi", "speed_fuxi"), ("ours", "speed_ours"), ("era5", "speed_era5")]:
        if col in sub.columns:
            out[f"mae_{name}"] = _mae(sub[col], o)
            out[f"bias_{name}"] = _bias(sub[col], o)
    return out


def build_table(uni: pd.DataFrame) -> dict:
    val = uni[uni["pop"] != "perdigao"]
    perd = uni[uni["pop"] == "perdigao"]
    table = {
        "overall_val": metrics_block(val),
        "wind_class_val": {},
        "steep_val": metrics_block(val[val["pop"] == "steep"]),
        "plain_val": metrics_block(val[val["pop"] == "plain"]),
        "perdigao": metrics_block(perd),
        "overall_all": metrics_block(uni),
    }
    for name, mask in [("lt3", val["speed_obs"] < 3.0),
                       ("3to6", (val["speed_obs"] >= 3.0) & (val["speed_obs"] <= 6.0)),
                       ("gt6", val["speed_obs"] > 6.0)]:
        table["wind_class_val"][name] = metrics_block(val[mask])
    # Perdigao by wind class too
    table["perdigao_wind_class"] = {}
    for name, mask in [("lt3", perd["speed_obs"] < 3.0),
                       ("3to6", (perd["speed_obs"] >= 3.0) & (perd["speed_obs"] <= 6.0)),
                       ("gt6", perd["speed_obs"] > 6.0)]:
        table["perdigao_wind_class"][name] = metrics_block(perd[mask])
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="M_I5 eval config (devine_style_M_I5_encoder.yaml)")
    ap.add_argument("--ann-checkpoint", type=Path, default=None)
    ap.add_argument("--combined-parquet", type=Path, required=True)
    ap.add_argument("--perdigao-obs", type=Path, required=True)
    ap.add_argument("--stores-root", type=Path, required=True,
                    help="dir containing era5_100m_*.zarr")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--scaler-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--perdigao-era5-store", type=Path, default=None,
                    help="ERA5 3D store for Perdigao grid.zarr materialise "
                         "(era5_europe_spring2017_v2.zarr)")
    ap.add_argument("--perdigao-cache-dir", type=Path, default=None,
                    help="grid.zarr cache dir for Perdigao (materialised on the fly)")
    ap.add_argument("--n-per-station", type=int, default=30)
    ap.add_argument("--perdigao-n-per-station", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fuxi-device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fuxi-batch-size", type=int, default=8)
    ap.add_argument("--max-delta-h", type=float, default=1.5)
    ap.add_argument("--n-prep-workers", type=int, default=4)
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

    # 1) Reference val split (106 stations) + sample.
    df = pd.read_parquet(args.combined_parquet)
    df["station_id"] = df["station_id"].astype(str)
    _, val_sids = watertight_station_split(
        args.combined_parquet, val_frac=float(cfg.get("val_frac", 0.2)),
        seed=int(cfg.get("seed", 42)),
        exclude_substrings=tuple(cfg.get("exclude_substrings", ["perdigao"])))
    val = df[df["station_id"].isin(set(val_sids))].copy()
    val = val.dropna(subset=["speed_obs", "lat", "lon", "height_obs"])
    val = val[val["speed_obs"] > 0.0].reset_index(drop=True)
    val_sample = sample_pairings(val, args.n_per_station, args.seed)
    logger.info("VAL sample: %d pairings, %d stations (target %d/station)",
                len(val_sample), val_sample["station_id"].nunique(), args.n_per_station)

    # 2) Perdigao sample.
    perd_sample = build_perdigao_sample(args.perdigao_obs, args.perdigao_n_per_station, args.seed)
    logger.info("PERDIGAO sample: %d pairings, %d stations",
                len(perd_sample), perd_sample["station_id"].nunique())

    # Unify columns for sampling/scoring.
    keep = ["station_id", "timestamp", "lat", "lon", "elev", "height_obs",
            "speed_obs", "u_obs", "v_obs", "season", "pop",
            "u10_era5_baseline", "v10_era5_baseline", "speed_era5_baseline"]
    for c in keep:
        if c not in perd_sample.columns:
            perd_sample[c] = np.nan
    val_sample = val_sample[[c for c in keep if c in val_sample.columns]].copy()
    perd_sample = perd_sample[[c for c in keep if c in perd_sample.columns]].copy()
    sample = pd.concat([val_sample, perd_sample], ignore_index=True)
    sample["timestamp"] = pd.to_datetime(sample["timestamp"])
    sample.to_parquet(args.out_dir / "sample_pairings.parquet", index=False)
    logger.info("TOTAL sample: %d pairings", len(sample))

    # 3) FuXi on the sample.
    fuxi_df = score_fuxi(
        sample, onnx=args.onnx, scaler_dir=args.scaler_dir,
        dem=Path(cfg["dem"]), worldcover=Path(cfg["worldcover"]),
        stores_root=args.stores_root, device=args.fuxi_device,
        batch_size=args.fuxi_batch_size, max_delta_h=args.max_delta_h,
        out_dir=args.out_dir, checkpoint_every=200)
    fuxi_df.to_parquet(args.out_dir / "fuxi_scores.parquet", index=False)

    # 4) OUR model on the sample.
    #    VAL: combined cache, require_cached (no re-materialise).
    #    PERDIGAO: own cache + spring2017 store, materialise on the fly (cheap, 2017
    #    timestamps differ from any prior subsample so we cannot require_cached).
    ours_parts = []
    val_only = sample[sample["pop"] != "perdigao"]
    if len(val_only):
        ours_parts.append(score_ours(
            cfg, norm, val_only, device=args.device, ann_ckpt=Path(ann_ckpt),
            batch_size=args.batch_size, n_prep=args.n_prep_workers,
            require_cached=True, tag="val"))
    perd_only = sample[sample["pop"] == "perdigao"]
    if len(perd_only) and args.perdigao_era5_store is not None:
        ours_parts.append(score_ours(
            cfg, norm, perd_only, device=args.device, ann_ckpt=Path(ann_ckpt),
            batch_size=args.batch_size, n_prep=args.n_prep_workers,
            cache_dir=args.perdigao_cache_dir, era5_store=args.perdigao_era5_store,
            require_cached=False, tag="perdigao"))
    ours_df = (pd.concat(ours_parts, ignore_index=True)
               if ours_parts else pd.DataFrame(columns=["station_id", "timestamp_iso", "speed_ours"]))
    ours_df.to_parquet(args.out_dir / "ours_scores.parquet", index=False)

    # 5) Build unified parquet (inner-ish: keep rows scored by at least obs+fuxi or obs+ours).
    sample["timestamp_iso"] = sample["timestamp"].map(lambda x: pd.Timestamp(x).isoformat())
    # ERA5 baseline: from parquet for val; for perdigao none → NaN.
    sample["speed_era5"] = sample["speed_era5_baseline"]
    uni = sample.merge(fuxi_df, on=["station_id", "timestamp_iso"], how="left")
    uni = uni.merge(ours_df, on=["station_id", "timestamp_iso"], how="left")
    cols = ["station_id", "timestamp_iso", "lat", "lon", "elev", "pop", "season",
            "speed_obs", "speed_fuxi", "speed_ours", "speed_era5", "uv100_delta_h"]
    uni = uni[[c for c in cols if c in uni.columns]]
    uni.to_parquet(args.out_dir / "unified_scores.parquet", index=False)

    # Coverage report.
    cov = {
        "n_sample": int(len(sample)),
        "n_fuxi_scored": int(uni["speed_fuxi"].notna().sum()),
        "n_ours_scored": int(uni["speed_ours"].notna().sum()),
        "n_era5": int(uni["speed_era5"].notna().sum()),
        "n_all_three": int((uni["speed_fuxi"].notna() & uni["speed_ours"].notna()
                            & uni["speed_obs"].notna()).sum()),
        "stations_total": int(uni["station_id"].nunique()),
        "stations_fuxi": int(uni.loc[uni["speed_fuxi"].notna(), "station_id"].nunique()),
        "stations_ours": int(uni.loc[uni["speed_ours"].notna(), "station_id"].nunique()),
    }
    # Score on the common subset where all of obs+fuxi+ours exist (apples-to-apples).
    common = uni[uni["speed_fuxi"].notna() & uni["speed_ours"].notna() & uni["speed_obs"].notna()].copy()
    table_common = build_table(common)
    table_full = build_table(uni)  # each model on its own coverage
    report = {
        "wall_s": round(time.time() - t0, 1),
        "coverage": cov,
        "n_per_station_val": args.n_per_station,
        "perdigao_n_per_station": args.perdigao_n_per_station,
        "max_delta_h_100m": args.max_delta_h,
        "terrain_cache": "per-station (station_id key), reprojected once/station",
        "table_common_subset": table_common,
        "table_full_coverage": table_full,
    }
    (args.out_dir / "score_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n===== COVERAGE =====", flush=True)
    print(json.dumps(cov, indent=2), flush=True)
    print("\n===== TABLE (common subset: obs+fuxi+ours all present) =====", flush=True)
    print(json.dumps(table_common, indent=2), flush=True)
    logger.info("Wrote unified parquet + report to %s", args.out_dir)


if __name__ == "__main__":
    main()
