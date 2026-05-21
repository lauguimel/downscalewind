"""Small Phase-G OBS Zarr writer shared by Portugal ingestion scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import zarr
from zarr.codecs import BloscCodec


OBS_VARS = ("u", "v", "wind_speed", "wind_dir", "t2m", "rh")
_COMPRESSOR = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    lat: float = np.nan
    lon: float = np.nan
    elev: float = np.nan
    source: str = "ipma_pt"
    country: str = "PT"
    z0_class_wc: int = -1


def ensure_obs_store(path: str | Path) -> zarr.Group:
    """Open or create the unified Portugal OBS store."""
    root = zarr.open_group(str(path), mode="a")
    root.attrs.update(
        {
            "source": "ipma_pt",
            "archive_provider": "ogimet",
            "country": "PT",
        }
    )
    root.attrs.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    stations = root.require_group("stations")
    _require_1d(stations, "station_id", "S16", fill_value=b"")
    _require_1d(stations, "lat", np.float32)
    _require_1d(stations, "lon", np.float32)
    _require_1d(stations, "elev", np.float32)
    _require_1d(stations, "source", "S16", fill_value=b"")
    _require_1d(stations, "country", "S2", fill_value=b"")
    _require_1d(stations, "z0_class_wc", np.int8, fill_value=-1)

    heights = root.require_group("heights")
    if "height_m" not in heights:
        arr = heights.create_array(
            "height_m",
            shape=(1,),
            dtype=np.float32,
            chunks=(1,),
            compressors=(_COMPRESSOR,),
            overwrite=True,
        )
        arr[:] = np.array([10.0], dtype=np.float32)
        arr.attrs.update({"units": "m AGL"})

    coords = root.require_group("coords")
    if "time" not in coords:
        arr = coords.create_array(
            "time",
            shape=(0,),
            dtype=np.int64,
            chunks=(720,),
            compressors=(_COMPRESSOR,),
            overwrite=True,
        )
        arr.attrs.update({"long_name": "time UTC", "units": "ns since epoch"})

    data = root.require_group("data")
    shape = (root["coords/time"].shape[0], root["stations/station_id"].shape[0], 1)
    for var in OBS_VARS:
        if var not in data:
            arr = data.create_array(
                var,
                shape=shape,
                dtype=np.float32,
                chunks=(720, 1, 1),
                compressors=(_COMPRESSOR,),
                fill_value=np.nan,
                overwrite=True,
            )
            arr.attrs.update(_var_attrs(var))
    return root


def read_station_ids(root: zarr.Group) -> list[str]:
    if "stations/station_id" not in root:
        return []
    return [_decode_bytes(v) for v in root["stations/station_id"][:]]


def ensure_stations(root: zarr.Group, stations: Iterable[StationRecord]) -> dict[str, int]:
    """Append missing stations and return station_id -> row index."""
    existing = read_station_ids(root)
    index = {sid: i for i, sid in enumerate(existing)}
    missing = [s for s in stations if s.station_id not in index]
    if not missing:
        return index

    old_s = len(existing)
    new_s = old_s + len(missing)
    for name in ("station_id", "lat", "lon", "elev", "source", "country", "z0_class_wc"):
        root[f"stations/{name}"].resize((new_s,))
    for arr_name in OBS_VARS:
        arr = root[f"data/{arr_name}"]
        arr.resize((arr.shape[0], new_s, arr.shape[2]))
        if arr.shape[0] > 0:
            arr[:, old_s:new_s, :] = np.nan

    offset = old_s
    root["stations/station_id"][old_s:new_s] = np.array(
        [s.station_id.encode("ascii") for s in missing], dtype="S16"
    )
    root["stations/lat"][old_s:new_s] = np.array([s.lat for s in missing], dtype=np.float32)
    root["stations/lon"][old_s:new_s] = np.array([s.lon for s in missing], dtype=np.float32)
    root["stations/elev"][old_s:new_s] = np.array([s.elev for s in missing], dtype=np.float32)
    root["stations/source"][old_s:new_s] = np.array(
        [s.source.encode("ascii") for s in missing], dtype="S16"
    )
    root["stations/country"][old_s:new_s] = np.array(
        [s.country.encode("ascii") for s in missing], dtype="S2"
    )
    root["stations/z0_class_wc"][old_s:new_s] = np.array(
        [s.z0_class_wc for s in missing], dtype=np.int8
    )
    for station in missing:
        index[station.station_id] = offset
        offset += 1
    return index


def write_station_timeseries(
    root: zarr.Group,
    station: StationRecord,
    times: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> None:
    """Write one station's 10 m series; missing variables are NaN-filled."""
    times_ns = _to_ns(times)
    order = np.argsort(times_ns)
    times_ns = times_ns[order]
    if times_ns.size == 0:
        ensure_stations(root, [station])
        return
    keep = np.r_[True, times_ns[1:] != times_ns[:-1]]
    times_ns = times_ns[keep]
    value_arrays = {
        var: np.asarray(values.get(var, np.full(order.size, np.nan)), dtype=np.float32)[order][keep]
        for var in OBS_VARS
    }

    station_idx = ensure_stations(root, [station])[station.station_id]
    time_map = ensure_times(root, times_ns)
    time_idx = np.array([time_map[int(t)] for t in times_ns], dtype=np.int64)
    for var, arr_values in value_arrays.items():
        _write_contiguous_runs(root[f"data/{var}"], time_idx, station_idx, arr_values)


def ensure_times(root: zarr.Group, times: np.ndarray) -> dict[int, int]:
    new_times = np.unique(_to_ns(times))
    old_times = root["coords/time"][:]
    union = np.union1d(old_times, new_times).astype(np.int64)
    if old_times.shape != union.shape or not np.array_equal(old_times, union):
        _resize_time_axis(root, old_times, union)
    return {int(t): int(i) for i, t in enumerate(union)}


def wind_to_uv(speed_ms: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(direction_deg.astype(np.float32))
    u = -speed_ms.astype(np.float32) * np.sin(rad)
    v = -speed_ms.astype(np.float32) * np.cos(rad)
    return u.astype(np.float32), v.astype(np.float32)


def _require_1d(group: zarr.Group, name: str, dtype, fill_value=np.nan) -> None:
    if name in group:
        return
    group.create_array(
        name,
        shape=(0,),
        dtype=dtype,
        chunks=(128,),
        compressors=(_COMPRESSOR,),
        fill_value=fill_value,
        overwrite=True,
    )


def _resize_time_axis(root: zarr.Group, old_times: np.ndarray, union: np.ndarray) -> None:
    old_t = len(old_times)
    new_t = len(union)
    coords = root["coords/time"]
    if old_t == 0 or (new_t >= old_t and np.array_equal(union[:old_t], old_times)):
        coords.resize((new_t,))
        coords[:] = union
        for var in OBS_VARS:
            arr = root[f"data/{var}"]
            arr.resize((new_t, arr.shape[1], arr.shape[2]))
            if new_t > old_t:
                arr[old_t:new_t, :, :] = np.nan
        return

    old_pos = np.searchsorted(union, old_times)
    for var in OBS_VARS:
        arr = root[f"data/{var}"]
        old_data = arr[:]
        arr.resize((new_t, arr.shape[1], arr.shape[2]))
        arr[:, :, :] = np.nan
        if old_data.size:
            arr[old_pos, :, :] = old_data
    coords.resize((new_t,))
    coords[:] = union


def _write_contiguous_runs(arr, time_idx: np.ndarray, station_idx: int, values: np.ndarray) -> None:
    start = 0
    while start < len(time_idx):
        end = start + 1
        while end < len(time_idx) and time_idx[end] == time_idx[end - 1] + 1:
            end += 1
        arr[time_idx[start] : time_idx[end - 1] + 1, station_idx, 0] = values[start:end]
        start = end


def _to_ns(times: np.ndarray) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[ns]").astype(np.int64)


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii").rstrip("\x00")
    return bytes(value).decode("ascii").rstrip("\x00")


def _var_attrs(var: str) -> dict[str, str]:
    attrs = {
        "u": ("eastward_wind", "m s-1"),
        "v": ("northward_wind", "m s-1"),
        "wind_speed": ("wind_speed", "m s-1"),
        "wind_dir": ("wind_from_direction", "degree"),
        "t2m": ("air_temperature", "K"),
        "rh": ("relative_humidity", "%"),
    }
    standard_name, units = attrs[var]
    return {"standard_name": standard_name, "units": units, "height": "10 m AGL"}
