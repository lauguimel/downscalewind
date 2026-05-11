"""
Sample IMERG and ERA5-Land daily rain at frozen FWI validation stations.

This script audits whether the selected station-days actually contain the dry
false-rain halos that motivate the IMERG fire-season correction.  It is a
pre-CNN diagnostic: it compares station rain24 with gridded IMERG/ERA5-Land at
the station point and writes halo manifests for downstream correction/inference.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Grid2D:
    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


@dataclass
class Grid3D:
    values: np.ndarray
    time: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray


def load_daily_rain(path: Path) -> Grid3D:
    with Dataset(path) as ds:
        rain = np.asarray(ds.variables["rain"][:], dtype=np.float32)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        tvar = ds.variables["time"]
        times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
        time = pd.DatetimeIndex(pd.to_datetime([str(t) for t in times]).normalize())
    return Grid3D(values=rain, time=time, lat=lat, lon=lon)


def load_static(path: Path, variable: str) -> Grid2D:
    with Dataset(path) as ds:
        values = np.asarray(ds.variables[variable][:], dtype=np.float32)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    return Grid2D(values=values, lat=lat, lon=lon)


def nearest_idx(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - float(value))))


def date_idx(times: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    target = pd.Timestamp(date).normalize()
    idx = int(np.argmin(np.abs(times - target)))
    if abs(times[idx] - target) > pd.Timedelta(hours=12):
        raise ValueError(f"nearest grid time {times[idx]} too far from {target}")
    return idx


def sample3(grid: Grid3D, date: pd.Timestamp, lat: float, lon: float) -> float:
    ti = date_idx(grid.time, date)
    yi = nearest_idx(grid.lat, lat)
    xi = nearest_idx(grid.lon, lon)
    value = grid.values[ti, yi, xi]
    return float(max(value, 0.0)) if np.isfinite(value) else float("nan")


def sample2(grid: Grid2D, lat: float, lon: float) -> float:
    yi = nearest_idx(grid.lat, lat)
    xi = nearest_idx(grid.lon, lon)
    value = grid.values[yi, xi]
    return float(value) if np.isfinite(value) else float("nan")


def sample_tpi(elevation: Grid2D, lat: float, lon: float, radius: int = 3) -> float:
    yi = nearest_idx(elevation.lat, lat)
    xi = nearest_idx(elevation.lon, lon)
    y0 = max(0, yi - radius)
    y1 = min(elevation.values.shape[0], yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(elevation.values.shape[1], xi + radius + 1)
    center = float(elevation.values[yi, xi])
    mean = float(np.nanmean(elevation.values[y0:y1, x0:x1]))
    return center - mean


def rain_metrics(df: pd.DataFrame, column: str, wet_threshold_mm: float) -> dict[str, float]:
    valid = df[["rain24_obs_mm", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {
            "product": column,
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "dry_false_wet_rate": np.nan,
            "wet_recall": np.nan,
        }
    obs = valid["rain24_obs_mm"].clip(lower=0.0).to_numpy()
    pred = valid[column].clip(lower=0.0).to_numpy()
    wet_obs = obs > wet_threshold_mm
    wet_pred = pred > wet_threshold_mm
    dry_obs = ~wet_obs
    corr = np.corrcoef(obs, pred)[0, 1] if len(valid) > 2 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    return {
        "product": column,
        "n": int(len(valid)),
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "mae": float(np.mean(np.abs(pred - obs))),
        "bias": float(np.mean(pred - obs)),
        "corr": float(corr) if np.isfinite(corr) else np.nan,
        "dry_false_wet_rate": float(wet_pred[dry_obs].mean()) if dry_obs.any() else np.nan,
        "wet_recall": float(wet_pred[wet_obs].mean()) if wet_obs.any() else np.nan,
    }


def write_frame(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def write_report(
    output_dir: Path,
    sampled: pd.DataFrame,
    metrics: pd.DataFrame,
    halo: pd.DataFrame,
    wet_threshold_mm: float,
    dry_threshold_mm: float,
) -> None:
    top_halo_cols = [
        "station_id",
        "name",
        "date",
        "rain24_obs_mm",
        "rain_imerg_center",
        "rain_era5land_center",
        "fwi_obs",
    ]
    lines = [
        "# FWI station rain audit",
        "",
        f"- Station-days: {len(sampled):,}.",
        f"- Stations: {sampled['station_id'].nunique():,}.",
        f"- Wet threshold: > {wet_threshold_mm:g} mm/day.",
        f"- Dry threshold for halo: <= {dry_threshold_mm:g} mm/day.",
        f"- IMERG or ERA5-Land dry false-rain halo days: {len(halo):,}.",
        "",
        "## Rain metrics",
        "",
    ]
    try:
        lines.append(metrics.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        lines.append("```csv")
        lines.append(metrics.to_csv(index=False))
        lines.append("```")
    lines.extend(["", "## Top halo days", ""])
    if halo.empty:
        lines.append("No halo days found.")
    else:
        top = halo.sort_values("halo_amount_max_mm", ascending=False).head(30)
        try:
            lines.append(top[top_halo_cols + ["halo_amount_max_mm"]].to_markdown(index=False, floatfmt=".3f"))
        except Exception:
            lines.append("```csv")
            lines.append(top[top_halo_cols + ["halo_amount_max_mm"]].to_csv(index=False))
            lines.append("```")
    (output_dir / "rain_audit_report.md").write_text("\n".join(lines) + "\n")


def run(
    station_days_path: Path,
    imerg_path: Path,
    era5land_path: Path,
    terrain_path: Path,
    output_dir: Path,
    wet_threshold_mm: float,
    dry_threshold_mm: float,
) -> None:
    station_days = pd.read_parquet(station_days_path) if station_days_path.suffix == ".parquet" else pd.read_csv(station_days_path)
    station_days["date"] = pd.to_datetime(station_days["date"]).dt.normalize()
    imerg = load_daily_rain(imerg_path)
    era5land = load_daily_rain(era5land_path)
    elevation = load_static(terrain_path, "elevation")
    slope = load_static(terrain_path, "slope")

    rows = []
    for row in station_days.itertuples(index=False):
        rows.append(
            {
                **row._asdict(),
                "rain_imerg_center": sample3(imerg, row.date, float(row.lat), float(row.lon)),
                "rain_era5land_center": sample3(era5land, row.date, float(row.lat), float(row.lon)),
                "terrain_elevation": sample2(elevation, float(row.lat), float(row.lon)),
                "terrain_slope": sample2(slope, float(row.lat), float(row.lon)),
                "terrain_tpi_approx": sample_tpi(elevation, float(row.lat), float(row.lon)),
            }
        )
    sampled = pd.DataFrame(rows)
    sampled["rain24_obs_mm"] = sampled["rain24_obs_mm"].clip(lower=0.0)
    sampled["imerg_dry_halo"] = (sampled["rain24_obs_mm"] <= dry_threshold_mm) & (
        sampled["rain_imerg_center"] > wet_threshold_mm
    )
    sampled["era5land_dry_halo"] = (sampled["rain24_obs_mm"] <= dry_threshold_mm) & (
        sampled["rain_era5land_center"] > wet_threshold_mm
    )
    sampled["halo_amount_max_mm"] = np.maximum(
        np.where(sampled["imerg_dry_halo"], sampled["rain_imerg_center"], 0.0),
        np.where(sampled["era5land_dry_halo"], sampled["rain_era5land_center"], 0.0),
    )

    metrics = pd.DataFrame(
        [
            rain_metrics(sampled, "rain_imerg_center", wet_threshold_mm),
            rain_metrics(sampled, "rain_era5land_center", wet_threshold_mm),
        ]
    )
    halo = sampled[sampled["imerg_dry_halo"] | sampled["era5land_dry_halo"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(sampled, output_dir / "rain_station_gridded_comparison.csv")
    write_frame(metrics, output_dir / "rain_metrics.csv")
    write_frame(halo, output_dir / "rain_halo_cases.csv")
    write_report(output_dir, sampled, metrics, halo, wet_threshold_mm, dry_threshold_mm)
    print(f"station_days={len(sampled)}")
    print(f"halo_days={len(halo)}")
    print(f"output_dir={output_dir}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-days", type=Path, required=True)
    parser.add_argument(
        "--imerg",
        type=Path,
        default=PROJECT_ROOT / "data/raw/downscalrain_grids/gee_2022/imerg_daily.nc",
    )
    parser.add_argument(
        "--era5land",
        type=Path,
        default=PROJECT_ROOT / "data/raw/downscalrain_grids/gee_2022/era5land_daily.nc",
    )
    parser.add_argument(
        "--terrain",
        type=Path,
        default=PROJECT_ROOT / "data/raw/downscalrain_grids/gee_2022/terrain_static.nc",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wet-threshold-mm", type=float, default=1.0)
    parser.add_argument("--dry-threshold-mm", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        args.station_days,
        args.imerg,
        args.era5land,
        args.terrain,
        args.output_dir,
        args.wet_threshold_mm,
        args.dry_threshold_mm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
