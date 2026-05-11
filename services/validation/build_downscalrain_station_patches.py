"""
Build DownscalRain CNN patches for frozen FWI station-days.

This avoids the xarray dependency used by the training dataset builder and reads
the small 2022 NetCDF grids directly with netCDF4.  The output directory is
compatible with services/module3-precip/predict_downscalrain.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "services/module3-precip"))

from src.patch_dataset import write_patch_dataset_metadata  # noqa: E402


CHANNELS = [
    "imerg_d0",
    "imerg_d1",
    "imerg_d2",
    "era5land_d0",
    "era5land_d1",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
]
META_COLUMNS = ["lat", "lon", "elevation", "slope", "tpi", "month_sin", "month_cos"]


@dataclass
class DynamicGrid:
    values: np.ndarray
    time: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    time_cache: dict[pd.Timestamp, int]
    point_cache: dict[tuple[float, float], tuple[int, int]]


@dataclass
class StaticGrid:
    values: dict[str, np.ndarray]
    lat: np.ndarray
    lon: np.ndarray
    point_cache: dict[tuple[float, float], tuple[int, int]]


def load_dynamic(path: Path) -> DynamicGrid:
    with Dataset(path) as ds:
        rain = np.asarray(ds.variables["rain"][:], dtype=np.float32)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        tvar = ds.variables["time"]
        times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
        time = pd.DatetimeIndex(pd.to_datetime([str(t) for t in times]).normalize())
    return DynamicGrid(rain, time, lat, lon, {}, {})


def load_static(path: Path) -> StaticGrid:
    with Dataset(path) as ds:
        values = {
            name: np.asarray(ds.variables[name][:], dtype=np.float32)
            for name in ["elevation", "slope", "aspect_sin", "aspect_cos"]
        }
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    return StaticGrid(values, lat, lon, {})


def nearest_idx(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - float(value))))


def point_idx(cache: dict[tuple[float, float], tuple[int, int]], lat_values: np.ndarray, lon_values: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    key = (round(float(lat), 6), round(float(lon), 6))
    if key not in cache:
        cache[key] = (nearest_idx(lat_values, lat), nearest_idx(lon_values, lon))
    return cache[key]


def time_idx(grid: DynamicGrid, date: pd.Timestamp) -> int:
    target = pd.Timestamp(date).normalize()
    if target not in grid.time_cache:
        idx = int(np.argmin(np.abs(grid.time - target)))
        if abs(grid.time[idx] - target) > pd.Timedelta(hours=12):
            raise ValueError(f"nearest grid time {grid.time[idx]} too far from {target}")
        grid.time_cache[target] = idx
    return grid.time_cache[target]


def patch2(values: np.ndarray, row: int, col: int, size: int, fill: float = 0.0) -> np.ndarray:
    half = size // 2
    r0 = row - half
    c0 = col - half
    r1 = r0 + size
    c1 = c0 + size
    out = np.full((size, size), fill, dtype=np.float32)
    src_r0 = max(0, r0)
    src_c0 = max(0, c0)
    src_r1 = min(values.shape[-2], r1)
    src_c1 = min(values.shape[-1], c1)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    out[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = values[
        src_r0:src_r1,
        src_c0:src_c1,
    ]
    return out


def dynamic_patch(grid: DynamicGrid, date: pd.Timestamp, offset_days: int, lat: float, lon: float, size: int) -> np.ndarray:
    row, col = point_idx(grid.point_cache, grid.lat, grid.lon, lat, lon)
    idx = time_idx(grid, pd.Timestamp(date) + pd.Timedelta(days=offset_days))
    return patch2(grid.values[idx], row, col, size)


def static_patch(grid: StaticGrid, variable: str, lat: float, lon: float, size: int) -> np.ndarray:
    row, col = point_idx(grid.point_cache, grid.lat, grid.lon, lat, lon)
    return patch2(grid.values[variable], row, col, size)


def build_meta(df: pd.DataFrame) -> np.ndarray:
    date = pd.to_datetime(df["date"])
    month = date.dt.month.astype(float)
    values = pd.DataFrame(
        {
            "lat": df["lat"].astype(float),
            "lon": df["lon"].astype(float),
            "elevation": df.get("terrain_elevation", df.get("alt_m", 0.0)).astype(float),
            "slope": df.get("terrain_slope", 0.0).astype(float),
            "tpi": df.get("terrain_tpi_approx", 0.0).astype(float),
            "month_sin": np.sin(2.0 * np.pi * month / 12.0),
            "month_cos": np.cos(2.0 * np.pi * month / 12.0),
        }
    )
    return values[META_COLUMNS].fillna(0.0).to_numpy(np.float32)


def run(
    station_days_path: Path,
    output_dir: Path,
    imerg_path: Path,
    era5land_path: Path,
    terrain_path: Path,
    patch_size: int,
) -> None:
    df = pd.read_parquet(station_days_path) if station_days_path.suffix == ".parquet" else pd.read_csv(station_days_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["station_id", "date", "lat", "lon", "rain24_obs_mm"]).reset_index(drop=True)
    imerg = load_dynamic(imerg_path)
    era5land = load_dynamic(era5land_path)
    static = load_static(terrain_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    patches = np.lib.format.open_memmap(
        output_dir / "patches.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(df), len(CHANNELS), patch_size, patch_size),
    )
    for i, row in enumerate(df.itertuples(index=False)):
        if (i + 1) % 500 == 0:
            print(f"patches {i + 1}/{len(df)}")
        lat = float(row.lat)
        lon = float(row.lon)
        date = pd.Timestamp(row.date)
        patches[i, 0] = dynamic_patch(imerg, date, 0, lat, lon, patch_size)
        patches[i, 1] = dynamic_patch(imerg, date, -1, lat, lon, patch_size)
        patches[i, 2] = dynamic_patch(imerg, date, -2, lat, lon, patch_size)
        patches[i, 3] = dynamic_patch(era5land, date, 0, lat, lon, patch_size)
        patches[i, 4] = dynamic_patch(era5land, date, -1, lat, lon, patch_size)
        patches[i, 5] = static_patch(static, "elevation", lat, lon, patch_size)
        patches[i, 6] = static_patch(static, "slope", lat, lon, patch_size)
        patches[i, 7] = static_patch(static, "aspect_sin", lat, lon, patch_size)
        patches[i, 8] = static_patch(static, "aspect_cos", lat, lon, patch_size)
    patches.flush()
    del patches

    write_patch_dataset_metadata(
        output_dir=output_dir,
        rain=df["rain24_obs_mm"].clip(lower=0.0).to_numpy(np.float32),
        meta=build_meta(df),
        station_ids=df["station_id"].astype(str).to_numpy(),
        dates=df["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        channels=CHANNELS,
        meta_columns=META_COLUMNS,
    )
    df.to_parquet(output_dir / "source_rows.parquet", index=False)
    (output_dir / "build_config.json").write_text(
        json.dumps(
            {
                "station_days": str(station_days_path),
                "imerg": str(imerg_path),
                "era5land": str(era5land_path),
                "terrain": str(terrain_path),
                "patch_size": patch_size,
                "channels": CHANNELS,
                "meta_columns": META_COLUMNS,
            },
            indent=2,
        )
    )
    print(f"patch_samples={len(df)}")
    print(f"output_dir={output_dir}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-days", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.station_days, args.output_dir, args.imerg, args.era5land, args.terrain, args.patch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
