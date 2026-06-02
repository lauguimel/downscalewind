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
import zarr
from torch.utils.data import DataLoader

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from src.ann_correction import ANNCorrection  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import ObsCenteredDataset, collate_obs_centered, watertight_station_split  # noqa: E402
from train_v2_devine_style import (  # noqa: E402
    _build_era5_layout,
    _denorm_uv_at_center,
    _era5_baseline_uv_at_center,
    _load_norm_overrides,
    build_frozen_surrogate,
)

logger = logging.getLogger("eval_devine_loso")

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
WIND_DIR_CONVENTION = (
    "Meteorological wind-from direction in degrees clockwise from north: "
    "dir_from = (270 - degrees(atan2(v_northward, u_eastward))) % 360"
)


def _timestamp_iso(value) -> str:
    try:
        return pd.Timestamp(str(value)).isoformat()
    except Exception:
        return str(value)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def _err(pred: np.ndarray, obs: np.ndarray, absolute: bool) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not mask.any():
        return float("nan")
    diff = pred[mask] - obs[mask]
    return float(np.abs(diff).mean() if absolute else diff.mean())


def _mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return _err(pred, obs, True)


def _bias(pred: np.ndarray, obs: np.ndarray) -> float:
    return _err(pred, obs, False)


def _dir_from_uv(u, v):
    # Meteorological "from" direction for vector components u eastward, v northward.
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def _angular_diff_deg(pred, obs):
    d = ((np.asarray(pred) - np.asarray(obs) + 180.0) % 360.0) - 180.0
    return np.where(d <= -180.0, d + 360.0, d)


def _circular_mean_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    rad = np.deg2rad(arr[mask])
    mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    return float(mean if mean > -180.0 else mean + 360.0)


def _mean_abs_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    return float(np.abs(arr[mask]).mean()) if mask.any() else float("nan")


def _median_abs_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    return float(np.median(np.abs(arr[mask]))) if mask.any() else float("nan")


def _direction_errors(sub: pd.DataFrame, pred_col: str) -> np.ndarray:
    mask = np.isfinite(sub["wind_dir_obs"]) & np.isfinite(sub[pred_col])
    mask &= sub["speed_obs"] >= 1.0
    return _angular_diff_deg(sub.loc[mask, pred_col], sub.loc[mask, "wind_dir_obs"])


def _sector_from_dir(direction_deg: np.ndarray) -> np.ndarray:
    idx = np.floor(((direction_deg + 22.5) % 360.0) / 45.0).astype(int) % 8
    return np.asarray(SECTORS, dtype=object)[idx]


def _selected_pairings(pairings: Path, station_ids: list[str], *, seed: int, max_pairings: int | None) -> pd.DataFrame:
    df = pd.read_parquet(pairings)
    required = {"station_id", "timestamp", "lat", "lon", "height_obs", "speed_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pairings parquet missing columns: {sorted(missing)}")
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(set(station_ids))].reset_index(drop=True)
    df = df.dropna(subset=["speed_obs", "lat", "lon", "height_obs"])
    df = df[df["speed_obs"] > 0.0].reset_index(drop=True)
    if max_pairings is not None and len(df) > max_pairings:
        df = df.sample(n=max_pairings, random_state=seed).reset_index(drop=True)
    df["timestamp_iso"] = df["timestamp"].map(_timestamp_iso)
    return df


def filter_to_cached_pairings(
    df: pd.DataFrame,
    cache_dir: Path,
    split_by_sid: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    if not cache_dir.exists():
        raise RuntimeError(
            f"BLOCKED: grid.zarr cache dir is missing: {cache_dir}. "
            "Do not re-materialise; use a subsample plan instead."
        )
    keep_mask: list[bool] = []
    dropped_months: dict[str, int] = {}
    total_by_val_sid: dict[str, int] = {}
    kept_by_val_sid: dict[str, int] = {}
    kept_per_split = {"train": 0, "val": 0}

    for row in df.itertuples(index=False):
        sid = str(row.station_id)
        split = split_by_sid.get(sid, "unknown")
        if split == "val":
            total_by_val_sid[sid] = total_by_val_sid.get(sid, 0) + 1
        p = ObsCenteredDataset._cache_path(cache_dir, sid, str(row.timestamp_iso))
        hit = p.exists()
        keep_mask.append(hit)
        if hit:
            if split in kept_per_split:
                kept_per_split[split] += 1
            if split == "val":
                kept_by_val_sid[sid] = kept_by_val_sid.get(sid, 0) + 1
        else:
            try:
                month = pd.Timestamp(str(row.timestamp_iso)).strftime("%Y-%m")
            except Exception:
                month = "unknown"
            dropped_months[month] = dropped_months.get(month, 0) + 1

    kept_df = df.loc[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
    val_zero = sorted(sid for sid in total_by_val_sid if kept_by_val_sid.get(sid, 0) == 0)
    total = int(len(df))
    kept = int(len(kept_df))
    coverage = {
        "total": total,
        "kept": kept,
        "dropped": int(total - kept),
        "kept_pct": float(kept / total * 100.0) if total else 0.0,
        "dropped_by_month": dict(sorted(dropped_months.items())),
        "kept_per_split": kept_per_split,
        "val_stations_with_zero_pairings": val_zero,
    }
    logger.info("Cache coverage: %s", json.dumps(_json_ready(coverage), sort_keys=True))
    if val_zero:
        raise RuntimeError(f"BLOCKED: val stations with zero cached pairings: {val_zero}")
    if kept < 1000:
        raise RuntimeError(f"BLOCKED: only {kept} cached pairings kept; refusing evaluation")
    return kept_df, coverage


def _build_dataset(cfg: dict, norm: dict, station_ids: list[str], *, pairings_parquet: Path,
                   max_pairings: int | None,
                   n_prep_workers: int) -> ObsCenteredDataset:
    return ObsCenteredDataset(
        Path(pairings_parquet),
        station_filter=station_ids,
        max_pairings=max_pairings,
        era5_store=Path(cfg["era5_store"]),
        dem=Path(cfg["dem"]),
        worldcover=Path(cfg["worldcover"]) if cfg.get("worldcover") else None,
        cache_dir=Path(cfg["cache_dir"]),
        norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)),
        n_workers=n_prep_workers,
        overwrite_cache=False,
        require_cached=True,
    )


def _load_ann(cfg: dict, checkpoint: Path, era5_dim: int, device: str) -> ANNCorrection:
    ann = ANNCorrection(
        era5_dim=era5_dim,
        topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)),
        zero_init_output=True,
    ).to(device)
    ck = torch.load(str(checkpoint), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"])
    ann.eval()
    logger.info("Loaded ANN %s (epoch=%s)", checkpoint, ck.get("epoch", "?"))
    return ann


def _forward_rows(ann: torch.nn.Module, surrogate: torch.nn.Module, loader: DataLoader, norm: dict,
                  era5_layout: dict, split_by_sid: dict[str, str], device: str, *,
                  limit_batches: int | None, dry_run: bool) -> list[dict]:
    rows: list[dict] = []
    ann.eval()
    surrogate.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            terrain, era5, geo, topo, speed_obs, k_obs, meta = batch
            terrain = terrain.to(device, non_blocking=True)
            era5 = era5.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            topo = topo.to(device, non_blocking=True)
            speed_obs = speed_obs.to(device, non_blocking=True)
            k_obs = k_obs.to(device, non_blocking=True)

            era5_corr = ann(era5, topo)
            pred_corr = surrogate(terrain, era5_corr, geo)
            u_res_c, v_res_c = _denorm_uv_at_center(pred_corr, norm, k_obs)
            u10_c, v10_c = _era5_baseline_uv_at_center(era5_corr, norm, era5_layout)
            u_c = u_res_c + u10_c
            v_c = v_res_c + v10_c
            speed_c = torch.sqrt(u_c * u_c + v_c * v_c + 1e-8)

            pred_raw = surrogate(terrain, era5, geo)
            u_res_r, v_res_r = _denorm_uv_at_center(pred_raw, norm, k_obs)
            u10_r, v10_r = _era5_baseline_uv_at_center(era5, norm, era5_layout)
            u_r = u_res_r + u10_r
            v_r = v_res_r + v10_r
            speed_r = torch.sqrt(u_r * u_r + v_r * v_r + 1e-8)

            if dry_run:
                print(f"terrain.shape={tuple(terrain.shape)}", flush=True)
                print(f"speed_pred_corr.shape={tuple(speed_c.shape)}", flush=True)
                print(f"speed_pred_raw.shape={tuple(speed_r.shape)}", flush=True)
                print(f"u_pred_corr.shape={tuple(u_c.shape)} v_pred_corr.shape={tuple(v_c.shape)}", flush=True)
                print(f"u_pred_raw.shape={tuple(u_r.shape)} v_pred_raw.shape={tuple(v_r.shape)}", flush=True)

            arrays = {
                "speed_obs": speed_obs.detach().cpu().numpy(),
                "speed_pred_corr": speed_c.detach().cpu().numpy(),
                "speed_pred_raw": speed_r.detach().cpu().numpy(),
                "u_pred_corr": u_c.detach().cpu().numpy(),
                "v_pred_corr": v_c.detach().cpu().numpy(),
                "u_pred_raw": u_r.detach().cpu().numpy(),
                "v_pred_raw": v_r.detach().cpu().numpy(),
                "era5_u10_raw": u10_r.detach().cpu().numpy(),
                "era5_v10_raw": v10_r.detach().cpu().numpy(),
            }
            era5_dir = _dir_from_uv(arrays["era5_u10_raw"], arrays["era5_v10_raw"])
            sectors = _sector_from_dir(era5_dir)
            for i, m in enumerate(meta):
                sid = str(m["station_id"])
                rows.append({
                    "station_id": sid,
                    "timestamp_iso": str(m["timestamp_iso"]),
                    "source": str(m.get("source", "")),
                    "height_obs": float(m["height_obs"]),
                    "split": split_by_sid[sid],
                    "sector": str(sectors[i]),
                    **{name: float(values[i]) for name, values in arrays.items()},
                })
            if dry_run:
                print("dry-run OK", flush=True)
                break
    return rows


def _attach_pairing_columns(df: pd.DataFrame, pairings_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    station_meta = (
        pairings_df.sort_values("timestamp_iso")
        .groupby("station_id", as_index=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"), elev=("elev", "first"))
    )
    keys = ["station_id", "timestamp_iso"]
    cols = ["speed_era5_baseline", "u_obs", "v_obs", "u_pred", "v_pred",
            "u10_era5_baseline", "v10_era5_baseline"]
    missing = [c for c in keys + cols if c not in pairings_df.columns]
    if missing:
        raise ValueError(f"pairings parquet missing required direction columns: {missing}")
    meta = pairings_df[keys + cols].copy()
    if meta.duplicated(keys).any():
        dup = int(meta.duplicated(keys).sum())
        raise ValueError(f"pairings parquet has duplicate station/timestamp keys: {dup}")
    out = df.merge(meta, on=keys, how="left", validate="m:1")
    status = {"pairing_column_join": "ok"}
    status.update({f"{c}_missing": int(out[c].isna().sum()) for c in cols})
    return out, status, station_meta


def _station_table(df: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("station_id", sort=True):
        dir_corr = _direction_errors(sub, "dir_pred_corr")
        dir_raw = _direction_errors(sub, "dir_pred_raw")
        dir_era5 = _direction_errors(sub, "dir_era5")
        rows.append({
            "station_id": sid,
            "split": str(sub["split"].iloc[0]),
            "n_pairings": int(len(sub)),
            "mae_corrected": _mae(sub["speed_pred_corr"], sub["speed_obs"]),
            "mae_raw": _mae(sub["speed_pred_raw"], sub["speed_obs"]),
            "mae_era5_baseline": _mae(sub["speed_era5_baseline"], sub["speed_obs"]),
            "bias_corrected": _bias(sub["speed_pred_corr"], sub["speed_obs"]),
            "bias_raw": _bias(sub["speed_pred_raw"], sub["speed_obs"]),
            "mae_dir_corrected": _mean_abs_finite(dir_corr),
            "mae_dir_raw": _mean_abs_finite(dir_raw),
            "mae_dir_era5": _mean_abs_finite(dir_era5),
            "bias_dir_corrected": _circular_mean_deg(dir_corr),
        })
    table = pd.DataFrame(rows).merge(station_meta, on="station_id", how="left")
    cols = ["station_id", "lat", "lon", "split", "n_pairings", "mae_corrected", "mae_raw",
            "mae_era5_baseline", "bias_corrected", "bias_raw", "mae_dir_corrected",
            "mae_dir_raw", "mae_dir_era5", "bias_dir_corrected"]
    return table[cols].sort_values(["split", "mae_corrected"], ascending=[True, False])


def _sector_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector in SECTORS:
        sub = df[df["sector"] == sector]
        dir_corr = _direction_errors(sub, "dir_pred_corr")
        dir_raw = _direction_errors(sub, "dir_pred_raw")
        rows.append({"sector": sector, "n_pairings": int(len(sub)),
                     "mae_corrected": _mae(sub["speed_pred_corr"], sub["speed_obs"]),
                     "mae_raw": _mae(sub["speed_pred_raw"], sub["speed_obs"]),
                     "bias_corrected": _bias(sub["speed_pred_corr"], sub["speed_obs"]),
                     "bias_raw": _bias(sub["speed_pred_raw"], sub["speed_obs"]),
                     "mae_dir_corrected": _mean_abs_finite(dir_corr),
                     "mae_dir_raw": _mean_abs_finite(dir_raw)})
    return pd.DataFrame(rows)


def _split_summary(df: pd.DataFrame, split: str) -> dict:
    sub = df[df["split"] == split]
    mae_c = _mae(sub["speed_pred_corr"], sub["speed_obs"])
    mae_r = _mae(sub["speed_pred_raw"], sub["speed_obs"])
    return {
        "mae_corrected": mae_c,
        "mae_raw": mae_r,
        "mae_era5_baseline": _mae(sub["speed_era5_baseline"], sub["speed_obs"]),
        "delta_pct": float((mae_r - mae_c) / mae_r * 100.0) if np.isfinite(mae_r) and mae_r else float("nan"),
        "n_stations": int(sub["station_id"].nunique()),
        "n_pairings": int(len(sub)),
        "bias_corrected": _bias(sub["speed_pred_corr"], sub["speed_obs"]),
        "bias_raw": _bias(sub["speed_pred_raw"], sub["speed_obs"]),
    }


def _speed_by_wind_class(df: pd.DataFrame) -> dict:
    val = df[df["split"] == "val"]
    classes = {
        "low_lt3": val["speed_obs"] < 3.0,
        "mid_3_7": (val["speed_obs"] >= 3.0) & (val["speed_obs"] <= 7.0),
        "high_gt7": val["speed_obs"] > 7.0,
    }
    out = {}
    for name, mask in classes.items():
        sub = val[mask]
        out[name] = {
            "n_pairings": int(len(sub)),
            "mae_corrected": _mae(sub["speed_pred_corr"], sub["speed_obs"]),
            "mae_raw": _mae(sub["speed_pred_raw"], sub["speed_obs"]),
            "mae_era5_baseline": _mae(sub["speed_era5_baseline"], sub["speed_obs"]),
        }
    return out


def _direction_error_triplet(sub: pd.DataFrame, speed_mask: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    mask = np.asarray(speed_mask, dtype=bool)
    for col in ["wind_dir_obs", "dir_pred_corr", "dir_pred_raw", "dir_era5"]:
        mask &= np.isfinite(sub[col].to_numpy(dtype=np.float64))
    obs = sub.loc[mask, "wind_dir_obs"]
    return {
        "corrected": _angular_diff_deg(sub.loc[mask, "dir_pred_corr"], obs),
        "raw": _angular_diff_deg(sub.loc[mask, "dir_pred_raw"], obs),
        "era5": _angular_diff_deg(sub.loc[mask, "dir_era5"], obs),
    }, int(mask.sum())


def _direction_metric_dict(errs: dict[str, np.ndarray]) -> dict:
    return {
        "mae_dir_deg": {k: _mean_abs_finite(v) for k, v in errs.items()},
        "median_dir_deg": {k: _median_abs_finite(v) for k, v in errs.items()},
        "bias_dir_deg": {k: _circular_mean_deg(v) for k, v in errs.items()},
    }


def _direction_bucket_pct(err: np.ndarray) -> dict:
    arr = np.abs(np.asarray(err, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    pct = lambda m: float(m.sum() / n * 100.0) if n else float("nan")
    return {"0_15": pct((arr >= 0.0) & (arr < 15.0)),
            "15_30": pct((arr >= 15.0) & (arr < 30.0)),
            "30_45": pct((arr >= 30.0) & (arr < 45.0)),
            "45_90": pct((arr >= 45.0) & (arr <= 90.0)),
            "90_180": pct((arr > 90.0) & (arr <= 180.0))}


def _direction_summary(df: pd.DataFrame, split: str) -> dict:
    sub = df[df["split"] == split]
    speed = sub["speed_obs"].to_numpy(dtype=np.float64)
    errs, n = _direction_error_triplet(sub, speed >= 1.0)
    out = {"n_pairings": n, **_direction_metric_dict(errs),
           "buckets_pct": {k: _direction_bucket_pct(v) for k, v in errs.items()}}
    out.update({
        "mae_dir_deg_corrected": out["mae_dir_deg"]["corrected"],
        "mae_dir_deg_raw": out["mae_dir_deg"]["raw"],
        "mae_dir_deg_era5": out["mae_dir_deg"]["era5"],
        "bias_dir_deg_corrected": out["bias_dir_deg"]["corrected"],
        "bias_dir_deg_raw": out["bias_dir_deg"]["raw"],
        "bias_dir_deg_era5": out["bias_dir_deg"]["era5"],
    })
    classes = {
        "calm_lt1": speed < 1.0,
        "low_1_3": (speed >= 1.0) & (speed < 3.0),
        "mid_3_7": (speed >= 3.0) & (speed <= 7.0),
        "high_gt7": speed > 7.0,
    }
    by_class = {}
    for name, mask in classes.items():
        class_errs, class_n = _direction_error_triplet(sub, mask)
        entry = {"n_pairings": class_n,
                 "mae_dir_deg": {k: _mean_abs_finite(v) for k, v in class_errs.items()}}
        if name == "calm_lt1":
            entry["note"] = "physically uninformative for direction; excluded from headline mae"
        by_class[name] = entry
    out["by_wind_class"] = by_class
    return out


def _terrain_proxies(cache_dir: Path, station_id: str, timestamp_iso: str) -> tuple[float, float]:
    try:
        p = ObsCenteredDataset._cache_path(cache_dir, station_id, timestamp_iso)
        g = zarr.open_group(str(p), mode="r")
        z0_eff = float(g["input"].attrs.get("z0_eff", np.nan))
        terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
        ic, jc = terrain.shape[0] // 2, terrain.shape[1] // 2
        patch = terrain[max(0, ic - 15):ic + 15, max(0, jc - 15):jc + 15]
        return z0_eff, float(np.nanstd(patch))
    except Exception as exc:
        logger.debug("terrain proxy read failed for %s @ %s: %s", station_id, timestamp_iso, exc)
        return float("nan"), float("nan")


def _direction_deflection_table(df: pd.DataFrame, station_meta: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    meta = station_meta.set_index("station_id")
    rows = []
    for sid, sub in df.groupby("station_id", sort=True):
        diff_era5 = _angular_diff_deg(sub["dir_pred_corr"], sub["dir_era5"])
        diff_raw = _angular_diff_deg(sub["dir_pred_corr"], sub["dir_pred_raw"])
        z0_eff, std_topo = _terrain_proxies(cache_dir, str(sid), str(sub["timestamp_iso"].iloc[0]))
        elev = float(meta.loc[sid, "elev"]) if sid in meta.index and "elev" in meta.columns else float("nan")
        dir_corr = _direction_errors(sub, "dir_pred_corr")
        dir_era5 = _direction_errors(sub, "dir_era5")
        rows.append({
            "station_id": sid,
            "split": str(sub["split"].iloc[0]),
            "n_pairings": int(len(sub)),
            "elev": elev,
            "defl_corr_minus_era5_deg": _circular_mean_deg(diff_era5),
            "defl_corr_minus_raw_deg": _circular_mean_deg(diff_raw),
            "z0_eff": z0_eff,
            "std_topo": std_topo,
            "mae_dir_corrected": _mean_abs_finite(dir_corr),
            "mae_dir_era5": _mean_abs_finite(dir_era5),
        })
    cols = ["station_id", "split", "n_pairings", "elev", "defl_corr_minus_era5_deg", "defl_corr_minus_raw_deg",
            "z0_eff", "std_topo", "mae_dir_corrected", "mae_dir_era5"]
    return pd.DataFrame(rows)[cols]


def _pearson_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    xarr = np.asarray(x, dtype=np.float64)
    yarr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xarr) & np.isfinite(yarr)
    if int(mask.sum()) < 2:
        return float("nan"), int(mask.sum())
    return float(np.corrcoef(xarr[mask], yarr[mask])[0, 1]), int(mask.sum())


def _deflection_summary(deflection_df: pd.DataFrame) -> dict:
    abs_defl = np.abs(deflection_df["defl_corr_minus_era5_deg"])
    pearson_z0, n_z0 = _pearson_corr(abs_defl, deflection_df["z0_eff"])
    pearson_std, n_std = _pearson_corr(abs_defl, deflection_df["std_topo"])
    out = {"n_stations": int(len(deflection_df)), "pearson_abs_defl_vs_z0_eff": pearson_z0,
           "pearson_abs_defl_vs_std_topo": pearson_std, "n_z0_eff": n_z0, "n_std_topo": n_std}
    try:
        from scipy import stats  # type: ignore

        zmask = np.isfinite(abs_defl) & np.isfinite(deflection_df["z0_eff"])
        smask = np.isfinite(abs_defl) & np.isfinite(deflection_df["std_topo"])
        for key, col, mask in [
            ("spearman_abs_defl_vs_z0_eff", "z0_eff", zmask),
            ("spearman_abs_defl_vs_std_topo", "std_topo", smask),
        ]:
            out[key] = (float(stats.spearmanr(abs_defl[mask], deflection_df.loc[mask, col]).correlation)
                        if int(mask.sum()) >= 2 else float("nan"))
    except Exception as exc:
        out["spearman_note"] = f"scipy.stats unavailable; Spearman skipped ({exc})"
    return out


def _direction_report(df: pd.DataFrame, obs_check: dict) -> dict:
    return {
        "val": _direction_summary(df, "val"),
        "train": _direction_summary(df, "train"),
        "calm_threshold": 1.0,
        "wind_dir_convention": WIND_DIR_CONVENTION,
        "obs_check": obs_check,
    }


def _summary(df: pd.DataFrame, station_df: pd.DataFrame, deflection_df: pd.DataFrame,
             coverage: dict, direction_report: dict, join_status: dict, wall_s: float) -> dict:
    return {
        "wall_s": wall_s,
        "coverage": coverage,
        "speed": {
            "val": _split_summary(df, "val"),
            "train": _split_summary(df, "train"),
            "by_wind_class": _speed_by_wind_class(df),
        },
        "direction": {**direction_report, **direction_report["val"]},
        "deflection": _deflection_summary(deflection_df),
        "top5_hardest_speed": station_df.nlargest(5, "mae_corrected")[["station_id", "mae_corrected"]].to_dict("records"),
        "top5_worst_dir": (
            station_df[np.isfinite(station_df["mae_dir_corrected"])]
            .nlargest(5, "mae_dir_corrected")[["station_id", "mae_dir_corrected"]]
            .to_dict("records")
        ),
        "wind_dir_convention": WIND_DIR_CONVENTION,
        "pairing_join_status": join_status,
        "obs_uv_dir_check": direction_report["obs_check"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ann-checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/validation/phase_H_prime_loso"))
    ap.add_argument("--obs-store", type=Path, default=Path("/home/maitreje/dsw/data/raw/obs_unified_noaa_isd_prod.zarr"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--max-pairings", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--n-prep-workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    device = args.device or cfg.get("device", "cuda")
    ann_checkpoint = args.ann_checkpoint or (Path(cfg["output_dir"]) / "best.pt")
    t0 = time.time()

    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(target_agl_levels.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = era5_layout["total_dim"]

    train_sids, val_sids = watertight_station_split(
        Path(cfg["pairings_parquet"]),
        val_frac=float(cfg.get("val_frac", 0.2)),
        seed=int(cfg.get("seed", 42)),
        exclude_substrings=tuple(cfg.get("exclude_substrings", ["perdigao"])),
    )
    station_ids = train_sids + val_sids
    split_by_sid = {sid: "train" for sid in train_sids} | {sid: "val" for sid in val_sids}
    logger.info("Stations split: train=%d val=%d all=%d", len(train_sids), len(val_sids), len(station_ids))

    selection_max = None if args.dry_run and args.max_pairings is not None else args.max_pairings
    pairings_df = _selected_pairings(Path(cfg["pairings_parquet"]), station_ids,
                                     seed=int(cfg.get("seed", 42)), max_pairings=selection_max)
    pairings_df, coverage = filter_to_cached_pairings(pairings_df, Path(cfg["cache_dir"]), split_by_sid)
    eval_pairings_df = pairings_df
    if args.dry_run and args.max_pairings is not None and len(eval_pairings_df) > args.max_pairings:
        eval_pairings_df = (
            eval_pairings_df.sample(n=args.max_pairings, random_state=int(cfg.get("seed", 42)))
            .reset_index(drop=True)
        )
        logger.info("Dry-run sampled %d cached pairings after full coverage check", len(eval_pairings_df))

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = "phase_H_prime_loso_pairings_cached_dryrun.parquet" if args.dry_run else \
        "phase_H_prime_loso_pairings_cached.parquet"
    filtered_pairings = tmp_dir / tmp_name
    eval_pairings_df.drop(columns=["timestamp_iso"], errors="ignore").to_parquet(filtered_pairings, index=False)
    logger.info("Wrote cached-only pairings parquet: %s rows=%d", filtered_pairings, len(eval_pairings_df))

    nprep = int(args.n_prep_workers if args.n_prep_workers is not None else cfg.get("n_prep_workers", 4))
    ds = _build_dataset(cfg, norm, station_ids, pairings_parquet=filtered_pairings,
                        max_pairings=None, n_prep_workers=nprep)
    logger.info("Dataset length=%d; cached-only parquet rows=%d; coverage kept=%d",
                len(ds), len(eval_pairings_df), coverage["kept"])
    if len(ds) == 0:
        raise RuntimeError("No cached evaluation pairings survived dataset construction")
    bs = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8))
    nw = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_obs_centered,
                        pin_memory=str(device).startswith("cuda"), persistent_workers=nw > 0)

    surrogate = build_frozen_surrogate(Path(cfg["surrogate_checkpoint"]), era5_dim=era5_dim, nz=nz,
                                       terrain_in_channels=4, geo_channels=int(cfg.get("geo_channels", 2)),
                                       preset=cfg.get("surrogate_preset", "base"), device=device)
    ann = _load_ann(cfg, Path(ann_checkpoint), era5_dim, device)

    rows = _forward_rows(ann, surrogate, loader, norm, era5_layout, split_by_sid, device,
                         limit_batches=args.limit_batches, dry_run=args.dry_run)
    if args.dry_run:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No evaluation rows produced")
    df, join_status, station_meta = _attach_pairing_columns(df, eval_pairings_df)
    df["wind_dir_obs"] = _dir_from_uv(df["u_obs"], df["v_obs"])
    df["dir_pred_raw"] = _dir_from_uv(df["u_pred"], df["v_pred"])
    df["dir_era5"] = _dir_from_uv(df["u10_era5_baseline"], df["v10_era5_baseline"])
    df["dir_pred_corr"] = _dir_from_uv(df["u_pred_corr"], df["v_pred_corr"])
    obs_check = {"note": "obs direction taken from paired parquet (u_obs,v_obs); separate obs-store join removed"}

    station_df = _station_table(df, station_meta)
    sector_df = _sector_table(df)
    deflection_df = _direction_deflection_table(df, station_meta, Path(cfg["cache_dir"]))
    direction_report = _direction_report(df, obs_check)
    summary = _summary(df, station_df, deflection_df, coverage, direction_report, join_status, time.time() - t0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(args.out_dir / "per_station_loso.csv", index=False)
    sector_df.to_csv(args.out_dir / "speed_mae_by_sector.csv", index=False)
    deflection_df.to_csv(args.out_dir / "direction_deflection.csv", index=False)
    pair_cols = ["station_id", "timestamp_iso", "split", "speed_obs", "u_obs", "v_obs", "u_pred", "v_pred",
                 "u_pred_corr", "v_pred_corr", "u10_era5_baseline", "v10_era5_baseline"]
    df[pair_cols].rename(columns={"u_pred": "u_pred_raw", "v_pred": "v_pred_raw"}).to_parquet(
        args.out_dir / "pairing_dir.parquet", index=False
    )
    (args.out_dir / "loso_summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    (args.out_dir / "loso_summary_dir.json").write_text(json.dumps(_json_ready(direction_report), indent=2) + "\n")
    val = df[(df["split"] == "val") & (df["speed_obs"] >= 1.0)]
    mask = np.ones(len(val), dtype=bool)
    for col in ["u_obs", "v_obs", "u_pred", "v_pred", "u10_era5_baseline", "v10_era5_baseline"]:
        mask &= np.isfinite(val[col].to_numpy(dtype=np.float64))
    obs_dir = _dir_from_uv(val.loc[mask, "u_obs"], val.loc[mask, "v_obs"])
    raw_err = _angular_diff_deg(_dir_from_uv(val.loc[mask, "u_pred"], val.loc[mask, "v_pred"]), obs_dir)
    era5_err = _angular_diff_deg(
        _dir_from_uv(val.loc[mask, "u10_era5_baseline"], val.loc[mask, "v10_era5_baseline"]), obs_dir
    )
    print(
        "SANITY parquet-only recompute (ALL val, speed>=1): "
        f"raw mae={_mean_abs_finite(raw_err):.2f} med={_median_abs_finite(raw_err):.2f} "
        f"bias={_circular_mean_deg(raw_err):+.2f} | era5 mae={_mean_abs_finite(era5_err):.2f} "
        f"med={_median_abs_finite(era5_err):.2f} bias={_circular_mean_deg(era5_err):+.2f}",
        flush=True,
    )
    logger.info("Wrote LOSO diagnostic to %s", args.out_dir)
    logger.info("Summary: %s", json.dumps(_json_ready(summary["speed"]["val"]), sort_keys=True))


if __name__ == "__main__":
    main()
