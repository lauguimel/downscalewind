"""
Build a DownscalRain CNN patch dataset from station labels and gridded sources.

Example:
    cd services/module3-precip
    python build_patch_dataset.py --config configs/downscalrain_cnn.yaml

The output is a directory-format dataset readable by `train_downscalrain.py`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.patch_dataset import write_patch_dataset

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _open_xarray(path: str | Path):
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "build_patch_dataset.py requires xarray. Activate the downscalewind "
            "environment or install xarray before building CNN patches."
        ) from exc

    path = Path(path)
    if path.suffix == ".zarr" or path.is_dir():
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _coord_name(da, candidates: list[str], configured: str | None = None) -> str:
    if configured:
        if configured not in da.coords and configured not in da.dims:
            raise KeyError(f"configured coord {configured!r} not found in {da.name}")
        return configured
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    raise KeyError(f"none of {candidates} found in coordinates/dims for {da.name}")


def _nearest_index(values: np.ndarray, value: float) -> int:
    arr = np.asarray(values, dtype=np.float64)
    return int(np.nanargmin(np.abs(arr - float(value))))


def _time_index(values: np.ndarray, timestamp: pd.Timestamp, tolerance_hours: float | None) -> int:
    times = pd.to_datetime(values)
    deltas = np.abs(times - timestamp)
    idx = int(np.argmin(deltas))
    if tolerance_hours is not None:
        if deltas[idx] > pd.Timedelta(hours=float(tolerance_hours)):
            raise ValueError(f"nearest time {times[idx]} exceeds tolerance for {timestamp}")
    return idx


def _slice_with_padding(arr: np.ndarray, row: int, col: int, size: int, fill_value: float) -> np.ndarray:
    half = size // 2
    r0 = row - half
    c0 = col - half
    r1 = r0 + size
    c1 = c0 + size

    out = np.full((size, size), fill_value, dtype=np.float32)
    src_r0 = max(0, r0)
    src_c0 = max(0, c0)
    src_r1 = min(arr.shape[-2], r1)
    src_c1 = min(arr.shape[-1], c1)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    out[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = arr[
        src_r0:src_r1,
        src_c0:src_c1,
    ]
    return out


class SourceReader:
    """Read one gridded variable and extract patches by lat/lon/date."""

    def __init__(self, spec: dict[str, Any], defaults: dict[str, Any]) -> None:
        self.spec = spec
        self.name = str(spec["name"])
        self.variable = str(spec["variable"])
        self.temporal = bool(spec.get("temporal", defaults.get("temporal", True)))
        self.offset_days = int(spec.get("offset_days", 0))
        self.multiplier = float(spec.get("multiplier", 1.0))
        self.fill_value = float(spec.get("fill_value", defaults.get("fill_value", 0.0)))
        self.time_tolerance_hours = spec.get("time_tolerance_hours", defaults.get("time_tolerance_hours"))

        ds = _open_xarray(spec["path"])
        if self.variable not in ds:
            raise KeyError(f"variable {self.variable!r} not found in {spec['path']}")
        da = ds[self.variable]
        self.da = da
        self.lat_name = _coord_name(da, ["lat", "latitude", "y"], spec.get("lat_name", defaults.get("lat_name")))
        self.lon_name = _coord_name(da, ["lon", "longitude", "x"], spec.get("lon_name", defaults.get("lon_name")))
        self.time_name = None
        if self.temporal:
            self.time_name = _coord_name(da, ["time", "date", "valid_time"], spec.get("time_name", defaults.get("time_name")))

        self.lat_values = np.asarray(da[self.lat_name].values)
        self.lon_values = np.asarray(da[self.lon_name].values)
        self.time_values = np.asarray(da[self.time_name].values) if self.time_name else None
        log.info("Opened source %s: %s[%s]", self.name, spec["path"], self.variable)

    def patch(self, lat: float, lon: float, date: pd.Timestamp, size: int) -> np.ndarray:
        da = self.da
        if self.temporal:
            assert self.time_name is not None and self.time_values is not None
            target_time = pd.Timestamp(date) + pd.Timedelta(days=self.offset_days)
            tidx = _time_index(self.time_values, target_time, self.time_tolerance_hours)
            da = da.isel({self.time_name: tidx})

        row = _nearest_index(self.lat_values, lat)
        col = _nearest_index(self.lon_values, lon)
        da = da.transpose(self.lat_name, self.lon_name, ...)
        values = np.asarray(da.values, dtype=np.float32)
        if values.ndim != 2:
            # Remove singleton dimensions left by source products.
            values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError(f"source {self.name} must reduce to 2D, got shape {values.shape}")
        return _slice_with_padding(values * self.multiplier, row, col, size, self.fill_value)


def _normalize_station_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "rain_mm" in df.columns and "rain_station" not in df.columns:
        rename["rain_mm"] = "rain_station"
    if rename:
        df = df.rename(columns=rename)
    required = {"station_id", "date", "lat", "lon", "rain_station"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"station table missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["rain_station"] = pd.to_numeric(df["rain_station"], errors="coerce")
    return df.dropna(subset=["station_id", "date", "lat", "lon", "rain_station"]).reset_index(drop=True)


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def main(config_path: str) -> None:
    cfg = _load_config(config_path)
    station_path = Path(cfg["patch_dataset"]["station_table"])
    output_dir = Path(cfg["patch_dataset"]["output_dir"])
    patch_size = int(cfg["patch_dataset"].get("patch_size", 64))
    max_samples = cfg["patch_dataset"].get("max_samples")

    stations = _normalize_station_columns(pd.read_parquet(station_path))
    if max_samples:
        stations = stations.head(int(max_samples)).copy()
    log.info("Building patches for %d station-days", len(stations))

    defaults = cfg.get("grid_defaults", {})
    source_specs = []
    source_specs.extend({**s, "temporal": True} for s in (cfg.get("dynamic_sources") or []))
    source_specs.extend({**s, "temporal": False} for s in (cfg.get("static_sources") or []))
    if not source_specs:
        raise ValueError("config must define at least one dynamic or static source")
    readers = [SourceReader(spec, defaults) for spec in source_specs]
    channels = [reader.name for reader in readers]

    meta_columns = list(cfg["patch_dataset"].get("metadata_columns", ["lat", "lon"]))
    for col in meta_columns:
        if col not in stations.columns:
            stations[col] = np.nan
    meta_values = stations[meta_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)

    patches = np.empty((len(stations), len(readers), patch_size, patch_size), dtype=np.float32)
    valid_rows: list[int] = []
    for i, row in enumerate(stations.itertuples(index=False)):
        if (i + 1) % 1000 == 0:
            log.info("Patch progress: %d / %d", i + 1, len(stations))
        try:
            patches[i] = np.stack(
                [
                    reader.patch(float(row.lat), float(row.lon), pd.Timestamp(row.date), patch_size)
                    for reader in readers
                ],
                axis=0,
            )
            valid_rows.append(i)
        except Exception as exc:
            log.warning("Skipping %s %s: %s", row.station_id, row.date, exc)

    valid = np.asarray(valid_rows, dtype=np.int64)
    stations_valid = stations.iloc[valid].reset_index(drop=True)
    write_patch_dataset(
        output_dir=output_dir,
        patches=patches[valid],
        rain=stations_valid["rain_station"].to_numpy(np.float32),
        meta=meta_values[valid],
        station_ids=stations_valid["station_id"].astype(str).to_numpy(),
        dates=stations_valid["date"].astype(str).to_numpy(),
        channels=channels,
        meta_columns=meta_columns,
    )
    log.info("Wrote %d patch samples to %s", len(valid), output_dir)


if __name__ == "__main__":
    main()
