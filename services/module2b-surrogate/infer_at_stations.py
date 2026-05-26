"""
infer_at_stations.py — Phase G M_G7 batched surrogate v2 inference at OBS stations.

Pipeline:
  1. Read OBS multi-source Zarrs (`obs_unified_*.zarr`) and concatenate stations
     via pandas. For each `(station_id, timestamp)` pairing, sample u_obs/v_obs
     /speed_obs at the height of observation (10 m for SYNOP/AEMET/IPMA/ISD,
     multi-height for Perdigão/ICOS).
  2. For each pairing, call M_G6 `build_one()` to materialise a grid.zarr/input
     at the station coords + timestamp (DEM + WorldCover + ERA5 3×3).
  3. Normalise via the same logic as `WindV2DatasetViT.__getitem__`, stack on a
     batch axis, forward through the best surrogate v2 checkpoint
     (`vit_base_resid_s4_geo_agl100_k24_surface`, val_mse=0.121).
  4. Denormalise (incl. residual + ERA5-surface baseline reconstruction) and
     extract u/v/w at the central column (i=NI/2, j=NJ/2) interpolated to the
     observation height in AGL.
  5. Write `data/inference/surrogate_at_stations.parquet` with columns
     `[station_id, timestamp, source, lat, lon, elev, height_obs, u_obs,
       v_obs, speed_obs, u_pred, v_pred, w_pred, speed_pred,
       era5_time_delta_minutes]`.

Smoke (CPU, ≤10 pairings):
  conda run -n downscalewind python services/module2b-surrogate/infer_at_stations.py \\
    --obs-zarrs data/raw/obs_unified_perdigao.zarr \\
    --era5-store data/raw/era5_europe_spring2017_v2.zarr \\
    --checkpoint /Users/guillaume/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt \\
    --output data/inference/smoke_surrogate_at_stations.parquet \\
    --smoke --max-pairings 10 --device cpu

Production (Aqua H100, stratified):
  python infer_at_stations.py \\
    --obs-zarrs data/raw/obs_unified_{perdigao,icos,noaa_isd,aemet_es,ipma_pt}.zarr \\
    --era5-store data/raw/era5_europe_hourly.zarr \\
    --checkpoint ... --output ... --stratify-timestamps \\
    --batch-size 32 --device cuda

Note on ERA5 cadence:
  M_G6 looks up the nearest ERA5 time in the store. When the store is Δt=6h
  (e.g. `era5_europe_spring2017_v2.zarr`), the per-pairing delta is reported in
  `era5_time_delta_minutes`. For hourly OBS one wants an hourly ERA5 store; the
  pipeline does not interpolate ERA5 in time.

Note on ERA5 d2m:
  The surrogate v2 surface input includes d2m. Stores without d2m
  (`era5_europe.zarr` legacy) cannot be used as-is — pick a v2-flavoured store
  that exposes d2m (e.g. `era5_europe_spring2017_v2.zarr`).
"""
from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import zarr

# Local imports — allow running as a standalone script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extract_v2_input_at_coords import build_one  # noqa: E402
from src.dataset_v2 import (  # noqa: E402
    DEFAULT_NORM,
    build_era5_baseline_tensor,
    parse_agl_levels,
)
from src.model_vit_v2 import build_vit_v2  # noqa: E402
from utils.inference_batch import (  # noqa: E402
    NI,
    NJ,
    build_features,
    denorm_fields,
    value_at_height,
)
from evaluate_v2_physical import load_norm_overrides  # noqa: E402

logger = logging.getLogger("infer_at_stations")


# ─── OBS pairings extraction ────────────────────────────────────────────────

def _decode_bytes(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8").rstrip("\x00"))
        else:
            out.append(str(v).rstrip("\x00"))
    return out


def load_obs_pairings(
    obs_zarrs: list[Path],
    *,
    height_target: float = 10.0,
    max_pairings: int | None = None,
    stratify: bool = False,
    era5_time_window_ns: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build a long-form DataFrame of `(station_id, timestamp, source, ...)`
    pairings across the listed obs Zarrs.

    Columns: station_id, source, lat, lon, elev, height_obs, timestamp_ns,
    u_obs, v_obs, speed_obs.

    Stations missing `u/v` at `height_target` are skipped. Each station is read
    independently from each zarr (no merge needed) — output is concatenated.
    """
    frames: list[pd.DataFrame] = []
    for zpath in obs_zarrs:
        if not zpath.exists():
            logger.warning("OBS zarr missing: %s — skipped", zpath)
            continue
        g = zarr.open_group(str(zpath), mode="r")
        sids = _decode_bytes(g["stations/station_id"][:])
        srcs = _decode_bytes(g["stations/source"][:])
        lats = np.asarray(g["stations/lat"][:], dtype=np.float32)
        lons = np.asarray(g["stations/lon"][:], dtype=np.float32)
        elevs = np.asarray(g["stations/elev"][:], dtype=np.float32)
        times_ns = np.asarray(g["coords/time"][:], dtype=np.int64)
        heights = np.asarray(g["heights/height_m"][:], dtype=np.float32)
        # closest height index to `height_target`
        h_idx = int(np.argmin(np.abs(heights - height_target)))
        height_used = float(heights[h_idx])

        u_all = np.asarray(g["data/u"][:, :, h_idx], dtype=np.float32)         # (T, S)
        v_all = np.asarray(g["data/v"][:, :, h_idx], dtype=np.float32)
        ws_all = np.asarray(g["data/wind_speed"][:, :, h_idx], dtype=np.float32)

        for s_idx, sid in enumerate(sids):
            u = u_all[:, s_idx]
            v = v_all[:, s_idx]
            ws = ws_all[:, s_idx]
            # require u or v valid (speed alone is direction-less)
            valid = np.isfinite(u) & np.isfinite(v)
            if not valid.any():
                continue
            ts_ns = times_ns[valid]
            frames.append(pd.DataFrame({
                "station_id": sid,
                "source": srcs[s_idx],
                "lat": float(lats[s_idx]),
                "lon": float(lons[s_idx]),
                "elev": float(elevs[s_idx]),
                "height_obs": height_used,
                "timestamp_ns": ts_ns,
                "u_obs": u[valid].astype(np.float32),
                "v_obs": v[valid].astype(np.float32),
                "speed_obs": ws[valid].astype(np.float32),
                "obs_zarr": str(zpath),
            }))
            logger.debug("  %s/%s: %d valid pairings at h=%.0f m",
                         srcs[s_idx], sid, int(valid.sum()), height_used)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if era5_time_window_ns is not None:
        n_before = len(df)
        t_lo, t_hi = era5_time_window_ns
        df = df[(df["timestamp_ns"] >= t_lo) & (df["timestamp_ns"] <= t_hi)].reset_index(drop=True)
        logger.info("ERA5 time-window pre-filter: %d → %d pairings (%.1f%% kept)",
                    n_before, len(df), 100.0 * len(df) / max(1, n_before))
    if stratify:
        df = _stratify_pairings(df)
    if max_pairings is not None and len(df) > max_pairings:
        df = df.head(max_pairings).reset_index(drop=True)
    logger.info("Loaded %d pairings across %d obs Zarrs (%d unique stations)",
                len(df), len(obs_zarrs), df["station_id"].nunique())
    return df


def _stratify_pairings(df: pd.DataFrame) -> pd.DataFrame:
    """48 cells × 30 ts/cell/station: 4 seasons × 3 wind_class × 4 synoptic h.

    Wind class: low (<3), mid (3-7), high (>7) — uses `speed_obs`.
    Synoptic hours: 00, 06, 12, 18 UTC ± 1h (so any hour falls in exactly one).
    """
    rng = np.random.default_rng(seed=42)
    t = pd.to_datetime(df["timestamp_ns"], utc=True)
    season = ((t.dt.month % 12) // 3).to_numpy()      # 0..3
    hour = t.dt.hour.to_numpy()
    syn_h = ((hour + 3) // 6) % 4                      # 4 buckets
    ws = df["speed_obs"].to_numpy()
    wc = np.where(ws < 3.0, 0, np.where(ws < 7.0, 1, 2))
    cell = season * 12 + wc * 4 + syn_h                # 48 cells
    df = df.copy()
    df["_strat_cell"] = cell

    out: list[pd.DataFrame] = []
    for (sid, c), grp in df.groupby(["station_id", "_strat_cell"], sort=False):
        n = min(len(grp), 30)
        idx = rng.choice(len(grp), size=n, replace=False)
        out.append(grp.iloc[idx])
    df_out = pd.concat(out, ignore_index=True).drop(columns="_strat_cell")
    logger.info("Stratification: kept %d / %d pairings (48 cells × 30 ts cap)",
                len(df_out), len(df))
    return df_out


# ─── Per-pairing grid.zarr (M_G6 reuse) ─────────────────────────────────────

def materialise_grid_zarr(
    *, station_id: str, lat: float, lon: float, elev: float,
    timestamp_ns: int, era5_store: Path, dem: Path, worldcover: Path | None,
    workdir: Path, max_era5_delta_h: float,
) -> Path:
    """Wrap M_G6 build_one() with a per-pairing tmp directory."""
    ts_iso = str(np.array(int(timestamp_ns)).astype("datetime64[ns]"))
    tag = ts_iso.replace(":", "").replace("-", "")[:13]
    out = workdir / f"{station_id}_{tag}" / "grid.zarr"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    return build_one(
        site_id=station_id,
        lat=lat, lon=lon,
        timestamp_iso=ts_iso,
        era5_store=era5_store,
        dem=dem,
        worldcover=worldcover,
        output=out,
        overwrite=True,
        extra_meta={"station_elev": float(elev)},
        max_era5_delta_h=max_era5_delta_h,
    )


# ─── Model build (lazy on first sample) ─────────────────────────────────────

def build_surrogate(
    ck: dict, sample_shapes: tuple[int, int, int], device: torch.device,
) -> torch.nn.Module:
    cfg = ck.get("config", {})
    terrain_channels, era5_dim, nz = sample_shapes
    model = build_vit_v2(
        preset=cfg.get("preset", "base"),
        era5_input_dim=era5_dim,
        nz=nz,
        terrain_in_channels=terrain_channels,
        geo_channels=2 if bool(cfg.get("use_geo", False)) else 0,
    )
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model


# ─── Batched forward ────────────────────────────────────────────────────────

def infer_batch(
    *, model: torch.nn.Module, stores: list, norm: dict, cfg: dict,
    target_agl_levels: np.ndarray | None, device: torch.device, amp: bool,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Run the surrogate on a list of grid.zarr stores, returning denormalised
    fields per pairing and the AGL levels (1D)."""
    use_geo = bool(cfg.get("use_geo", False))
    use_resid = bool(cfg.get("use_residual", False))
    resid_mode = str(cfg.get("residual_baseline_mode", "pressure_index"))

    terrain_l: list[np.ndarray] = []
    era5_l: list[np.ndarray] = []
    geo_l: list[np.ndarray] = []
    base_l: list[np.ndarray] = []
    levels: np.ndarray | None = None

    for st in stores:
        terrain_2d, era5_flat, geo, lv = build_features(st, norm, cfg, target_agl_levels)
        terrain_l.append(terrain_2d)
        era5_l.append(era5_flat)
        geo_l.append(geo)
        if use_resid:
            base_l.append(build_era5_baseline_tensor(st, norm, lv.size, mode=resid_mode))
        if levels is None:
            levels = lv
    terrain_t = torch.from_numpy(np.stack(terrain_l, axis=0)).to(device)
    era5_t = torch.from_numpy(np.stack(era5_l, axis=0)).to(device)
    geo_t = torch.from_numpy(np.stack(geo_l, axis=0)).to(device) if use_geo else None

    with torch.no_grad(), torch.autocast(device_type=device.type,
                                          enabled=amp and device.type == "cuda"):
        pred = model(terrain_t, era5_t, geo_t).detach().float().cpu().numpy()

    if use_resid:
        baseline = np.stack(base_l, axis=0)                # (B, 5, NI, NJ, nz)
        pred = pred + baseline

    fields_per_sample = [denorm_fields(pred[b], norm) for b in range(pred.shape[0])]
    return fields_per_sample, np.asarray(levels, dtype=np.float32)


# ─── Per-pairing extraction at station column ──────────────────────────────

def extract_at_station(
    fields: dict[str, np.ndarray], levels: np.ndarray, height_obs: float,
) -> dict[str, float]:
    """Return u/v/w at the central column (NI/2, NJ/2), interpolated to `height_obs`.
    """
    iy, ix = NI // 2, NJ // 2
    u = value_at_height(fields["u"][iy, ix, :], levels, height_obs)
    v = value_at_height(fields["v"][iy, ix, :], levels, height_obs)
    w = value_at_height(fields["w"][iy, ix, :], levels, height_obs)
    return {
        "u_pred": float(u),
        "v_pred": float(v),
        "w_pred": float(w),
        "speed_pred": float(math.hypot(u, v)),
    }


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run_inference(
    *, df: pd.DataFrame, era5_store: Path, dem: Path, worldcover: Path | None,
    checkpoint: Path, norm_yaml: Path | None, output: Path,
    device_name: str, batch_size: int, amp: bool, max_era5_delta_h: float,
    workdir: Path, keep_grids: bool,
) -> Path:
    device = torch.device(
        device_name if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("device=%s | n_pairings=%d | batch=%d", device, len(df), batch_size)

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model" not in ck:
        raise KeyError(f"{checkpoint} missing 'model' state_dict")
    cfg = ck.get("config", {}) or {}
    norm = {**DEFAULT_NORM, **load_norm_overrides(norm_yaml)} if norm_yaml else {**DEFAULT_NORM}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels"))
    logger.info("checkpoint epoch=%s val_mse=%s preset=%s use_geo=%s use_residual=%s "
                "residual_baseline_mode=%s include_slopes=%s nz=%d",
                ck.get("epoch"), ck.get("val_mse"), cfg.get("preset", "base"),
                cfg.get("use_geo"), cfg.get("use_residual"),
                cfg.get("residual_baseline_mode"), cfg.get("include_slopes"),
                target_agl_levels.size if target_agl_levels is not None else -1)

    workdir.mkdir(parents=True, exist_ok=True)
    model: torch.nn.Module | None = None
    out_rows: list[dict] = []
    n_built = n_skipped = 0
    t_start = time.time()

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        stores = []
        chunk_meta = []
        for _, row in chunk.iterrows():
            try:
                gz = materialise_grid_zarr(
                    station_id=str(row["station_id"]),
                    lat=float(row["lat"]), lon=float(row["lon"]),
                    elev=float(row["elev"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                    era5_store=era5_store,
                    dem=dem, worldcover=worldcover,
                    workdir=workdir,
                    max_era5_delta_h=max_era5_delta_h,
                )
                st = zarr.open_group(str(gz), mode="r")
                stores.append(st)
                chunk_meta.append((row, gz))
                n_built += 1
            except Exception as exc:
                logger.warning("skipped %s @ %s: %s",
                               row.get("station_id"), row.get("timestamp_ns"), exc)
                n_skipped += 1

        if not stores:
            continue

        # lazy build model from the first batch's first store
        if model is None:
            terrain_2d, era5_flat, geo, lv = build_features(stores[0], norm, cfg, target_agl_levels)
            model = build_surrogate(
                ck,
                (terrain_2d.shape[0], era5_flat.shape[0], lv.size),
                device,
            )
            logger.info("model built | terrain_ch=%d era5_dim=%d nz=%d",
                        terrain_2d.shape[0], era5_flat.shape[0], lv.size)

        fields_l, levels = infer_batch(
            model=model, stores=stores, norm=norm, cfg=cfg,
            target_agl_levels=target_agl_levels, device=device, amp=amp,
        )
        for (row, gz), fields in zip(chunk_meta, fields_l):
            vals = extract_at_station(fields, levels, float(row["height_obs"]))
            ts_iso = str(np.array(int(row["timestamp_ns"])).astype("datetime64[ns]"))
            gz_root = zarr.open_group(str(gz), mode="r")
            era5_attrs = dict(gz_root["input/inflow_meta"].attrs)
            era5_delta_s = float(era5_attrs.get("era5_time_delta_s", 0.0))
            try:
                u10_era5 = float(gz_root["input/era5_surface/u10"][1, 1])
                v10_era5 = float(gz_root["input/era5_surface/v10"][1, 1])
                speed_era5_baseline = float(math.hypot(u10_era5, v10_era5))
            except (KeyError, IndexError):
                u10_era5 = v10_era5 = speed_era5_baseline = float("nan")
            out_rows.append({
                "station_id": row["station_id"],
                "timestamp": ts_iso,
                "source": row["source"],
                "lat": row["lat"], "lon": row["lon"], "elev": row["elev"],
                "height_obs": row["height_obs"],
                "u_obs": row["u_obs"], "v_obs": row["v_obs"],
                "speed_obs": row["speed_obs"],
                "u_pred": vals["u_pred"], "v_pred": vals["v_pred"],
                "w_pred": vals["w_pred"], "speed_pred": vals["speed_pred"],
                "u10_era5_baseline": u10_era5,
                "v10_era5_baseline": v10_era5,
                "speed_era5_baseline": speed_era5_baseline,
                "era5_time_delta_minutes": era5_delta_s / 60.0,
                "obs_zarr": row.get("obs_zarr", ""),
            })
            if not keep_grids:
                shutil.rmtree(gz.parent, ignore_errors=True)
        elapsed = time.time() - t_start
        logger.info("[%d/%d] built=%d skipped=%d elapsed=%.1fs",
                    min(start + batch_size, len(df)), len(df),
                    n_built, n_skipped, elapsed)
        # Periodic parquet checkpoint to survive walltime kills.
        # Flush every ~1000 new rows (overwrite all-rows-so-far).
        if out_rows and len(out_rows) >= 1000 and len(out_rows) // 1000 != getattr(run_inference, "_last_flush", -1):
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(out_rows).to_parquet(output, index=False)
            run_inference._last_flush = len(out_rows) // 1000
            logger.info("checkpoint: wrote %d rows to %s", len(out_rows), output)

    if not out_rows:
        logger.error("No pairings produced output — aborting parquet write.")
        sys.exit(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(out_rows)
    df_out.to_parquet(output, index=False)
    logger.info("Wrote %s (%d rows, finite speed_pred: %d / %d, finite speed_obs: %d / %d)",
                output, len(df_out),
                int(np.isfinite(df_out["speed_pred"]).sum()), len(df_out),
                int(np.isfinite(df_out["speed_obs"]).sum()), len(df_out))
    return output


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command(context_settings={"show_default": True})
@click.option("--obs-zarrs", required=True,
              help="Comma-separated paths to obs_unified_*.zarr stores")
@click.option("--era5-store", type=click.Path(exists=True, path_type=Path),
              required=True,
              help="ERA5 Zarr (must include surface/d2m for the residual baseline)")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path),
              required=True, help="best.pt of the surrogate v2 to use")
@click.option("--norm-yaml", type=click.Path(exists=False, path_type=Path),
              default=None,
              help="dataset_v2_norm.yaml (Welford stats); falls back to DEFAULT_NORM if absent")
@click.option("--dem", type=click.Path(exists=True, path_type=Path),
              default="data/raw/srtm_perdigao_30m.tif",
              help="DEM GeoTIFF (Copernicus GLO-30 / SRTM)")
@click.option("--worldcover", type=click.Path(exists=False, path_type=Path),
              default=None,
              help="ESA WorldCover 2021 GeoTIFF for z0_eff (optional; pipeline "
                   "uses fallback z0 if absent)")
@click.option("--output", type=click.Path(path_type=Path), required=True,
              help="Output parquet path")
@click.option("--device", default="auto", help="cpu, cuda or auto")
@click.option("--batch-size", type=int, default=32, help="Batched forward size")
@click.option("--amp", is_flag=True, default=False,
              help="Mixed precision (CUDA only)")
@click.option("--height-target", type=float, default=10.0,
              help="OBS height (m AGL) used as ground truth")
@click.option("--max-era5-delta-h", type=float, default=6.5,
              help="Allowed gap between OBS ts and nearest ERA5 ts (hours)")
@click.option("--smoke", is_flag=True, default=False,
              help="Smoke mode: log per-batch + keep last grid.zarr for debug")
@click.option("--max-pairings", type=int, default=None,
              help="Cap total pairings (smoke mode default = 10)")
@click.option("--stratify-timestamps", is_flag=True, default=False,
              help="4 seasons × 3 wind class × 4 synoptic h = 48 cells × 30 ts cap")
@click.option("--workdir", type=click.Path(path_type=Path), default=None,
              help="Where to put temporary grid.zarr (default = tmp directory)")
@click.option("--keep-grids", is_flag=True, default=False,
              help="Do not delete per-pairing grid.zarr after forward")
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(obs_zarrs, era5_store, checkpoint, norm_yaml, dem, worldcover, output,
        device, batch_size, amp, height_target, max_era5_delta_h, smoke,
        max_pairings, stratify_timestamps, workdir, keep_grids, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    obs_paths = [Path(p.strip()) for p in obs_zarrs.split(",") if p.strip()]
    if smoke and max_pairings is None:
        max_pairings = 10
    # Pre-filter OBS pairings on ERA5 time window (avoid 96% skip rate at
    # materialise_grid_zarr stage — saw 478k → 9k built on job 21901136
    # walltime kill, root cause: ERA5 store only covers 4 months).
    era5_window = None
    try:
        era5_root = zarr.open_group(str(era5_store), mode="r")
        era5_times_ns = np.asarray(era5_root["coords/time"][:], dtype=np.int64)
        margin_ns = int(max_era5_delta_h * 3600 * 1_000_000_000)
        era5_window = (int(era5_times_ns[0]) - margin_ns,
                       int(era5_times_ns[-1]) + margin_ns)
        logger.info("ERA5 time window: [%s, %s] ± %.1f h",
                    str(np.array(era5_times_ns[0]).astype("datetime64[ns]")),
                    str(np.array(era5_times_ns[-1]).astype("datetime64[ns]")),
                    max_era5_delta_h)
    except Exception as exc:
        logger.warning("Could not pre-filter ERA5 time window: %s", exc)
    df = load_obs_pairings(
        obs_paths,
        height_target=height_target,
        max_pairings=max_pairings,
        stratify=stratify_timestamps,
        era5_time_window_ns=era5_window,
    )
    if df.empty:
        logger.error("No usable OBS pairings found — aborting.")
        sys.exit(2)

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="infer_at_stations_"))
    else:
        workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info("workdir=%s", workdir)

    run_inference(
        df=df,
        era5_store=Path(era5_store),
        dem=Path(dem),
        worldcover=Path(worldcover) if worldcover else None,
        checkpoint=Path(checkpoint),
        norm_yaml=Path(norm_yaml) if norm_yaml else None,
        output=Path(output),
        device_name=device,
        batch_size=batch_size,
        amp=amp,
        max_era5_delta_h=max_era5_delta_h,
        workdir=workdir,
        keep_grids=keep_grids or smoke,
    )


if __name__ == "__main__":
    cli()
