"""Shared writer for Phase G unified observation Zarr stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from shared.data_io import _DEFAULT_COMPRESSOR


CANONICAL_HEIGHTS_M = np.array([10, 20, 40, 60, 80, 100], dtype=np.float32)
UNIFIED_VARS = ("u", "v", "wind_speed", "wind_dir", "t2m", "rh")


def _bytes(values: list[str], dtype: str) -> np.ndarray:
    return np.array([v.encode("ascii") for v in values], dtype=dtype)


def _create_array(group: zarr.Group, name: str, data: np.ndarray, chunks: tuple[int, ...]):
    arr = group.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        compressors=[_DEFAULT_COMPRESSOR],
        overwrite=True,
    )
    arr[...] = data
    return arr


def _validate_inputs(
    times_ns: np.ndarray,
    station_records: list[dict],
    data_per_station: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, int, int, int]:
    times_ns = np.asarray(times_ns, dtype=np.int64)
    if times_ns.ndim != 1 or times_ns.size < 1:
        raise ValueError("times_ns must be a non-empty 1D int64 array")
    if np.any(np.diff(times_ns) <= 0):
        raise ValueError("times_ns must be strictly sorted")
    hour_ns = np.int64(3_600_000_000_000)
    if times_ns.size > 1 and not np.all(np.diff(times_ns) == hour_ns):
        raise ValueError("times_ns must be hourly")

    n_times = int(times_ns.size)
    n_stations = len(station_records)
    n_heights = int(CANONICAL_HEIGHTS_M.size)
    if n_stations == 0:
        raise ValueError("at least one station is required")
    if len(data_per_station) != n_stations:
        raise ValueError("station_records and data_per_station lengths differ")

    required_station_keys = {"station_id", "lat", "lon", "elev", "country"}
    expected_shape = (n_times, n_heights)
    for i, record in enumerate(station_records):
        missing = required_station_keys - set(record)
        if missing:
            raise ValueError(f"station_records[{i}] missing {sorted(missing)}")
        for key in ("station_id", "country"):
            record[key].encode("ascii")
        for var in UNIFIED_VARS:
            if var not in data_per_station[i]:
                raise ValueError(f"data_per_station[{i}] missing {var}")
            arr = np.asarray(data_per_station[i][var])
            if arr.shape != expected_shape:
                raise ValueError(
                    f"data_per_station[{i}][{var}] shape {arr.shape}, "
                    f"expected {expected_shape}"
                )
    return times_ns, n_times, n_stations, n_heights


def write_unified_obs_zarr(
    output_path: Path,
    times_ns: np.ndarray,
    station_records: list[dict],
    data_per_station: list[dict[str, np.ndarray]],
    source: str,
) -> None:
    """Write a Phase G unified OBS Zarr store."""
    if source not in {"perdigao", "icos", "noaa_isd"}:
        raise ValueError("source must be 'perdigao', 'icos', or 'noaa_isd'")

    times_ns, n_times, n_stations, n_heights = _validate_inputs(
        times_ns, station_records, data_per_station
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w")

    station_group = root.require_group("stations")
    station_chunks = (n_stations,)
    _create_array(
        station_group,
        "station_id",
        _bytes([r["station_id"] for r in station_records], "S16"),
        station_chunks,
    )
    lat = np.array([r["lat"] for r in station_records], dtype=np.float32)
    lon = np.array([r["lon"] for r in station_records], dtype=np.float32)
    elev = np.array([r["elev"] for r in station_records], dtype=np.float32)
    _create_array(station_group, "lat", lat, station_chunks).attrs.update(
        {"units": "degrees_north"}
    )
    _create_array(station_group, "lon", lon, station_chunks).attrs.update(
        {"units": "degrees_east"}
    )
    _create_array(station_group, "elev", elev, station_chunks).attrs.update(
        {"units": "m ASL"}
    )
    _create_array(
        station_group,
        "source",
        _bytes([source] * n_stations, "S16"),
        station_chunks,
    )
    _create_array(
        station_group,
        "country",
        _bytes([r["country"] for r in station_records], "S2"),
        station_chunks,
    )
    _create_array(
        station_group,
        "z0_class_wc",
        np.full(n_stations, -1, dtype=np.int8),
        station_chunks,
    )

    heights = root.require_group("heights")
    _create_array(heights, "height_m", CANONICAL_HEIGHTS_M, (n_heights,)).attrs.update(
        {"units": "m AGL"}
    )

    coords = root.require_group("coords")
    _create_array(coords, "time", times_ns, (min(720, n_times),)).attrs.update(
        {"long_name": "time UTC", "units": "ns since epoch"}
    )

    data_group = root.require_group("data")
    chunks = (min(720, n_times), 1, n_heights)
    meta = {
        "u": ("East wind", "m s-1"),
        "v": ("North wind", "m s-1"),
        "wind_speed": ("wind speed", "m s-1"),
        "wind_dir": ("wind direction from north", "degrees"),
        "t2m": ("air temperature", "K"),
        "rh": ("relative humidity", "%"),
    }
    for var in UNIFIED_VARS:
        stacked = np.stack(
            [np.asarray(station_data[var], dtype=np.float32) for station_data in data_per_station],
            axis=1,
        )
        arr = _create_array(data_group, var, stacked.astype(np.float32, copy=False), chunks)
        long_name, units = meta[var]
        arr.attrs.update({"long_name": long_name, "units": units})

    root.attrs.update(
        {
            "Conventions": "CF-1.9",
            "title": f"Unified observation data — {source}",
            "source": source,
            "schema": "Phase G unified OBS",
            "n_stations": n_stations,
            "n_heights": n_heights,
            "time_resolution": "1h",
        }
    )


def readback_unified_obs_summary(path: Path, source: str) -> tuple[int, int, int, int]:
    """Small validation used by smoke modes."""
    root = zarr.open_group(str(path), mode="r")
    heights = root["heights/height_m"][:]
    if heights.tolist() != CANONICAL_HEIGHTS_M.tolist():
        raise AssertionError("canonical heights mismatch")
    source_values = set(root["stations/source"][:].tolist())
    if source_values != {source.encode("ascii")}:
        raise AssertionError("source values mismatch")
    ws = root["data/wind_speed"][:]
    n_valid_ws = int(np.isfinite(ws).sum())
    if n_valid_ws <= 0:
        raise AssertionError("no finite wind_speed values")
    return ws.shape[0], ws.shape[1], ws.shape[2], n_valid_ws
