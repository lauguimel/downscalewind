"""
shared.obs_io — Helpers Zarr pour lire/écrire les observations multi-sources.

Schéma Zarr unifié pour les observations Perdigão, SYNOP, AEMET, IPMA et ICOS :

data/raw/obs_unified.zarr/
  stations/
    station_id  (S,)   bytes (S16)
    lat, lon, elev      float32 (S,)
    source              (S,) bytes (S16) — "perdigao", "synop_fr", "aemet_es", "ipma_pt", "icos"
    country             (S,) bytes (S2) — "PT", "FR", "ES"
    z0_class_wc         int8 (S,) — WC 2021 mode 1km, -1 si pas calculé
  heights/
    height_m            float32 (H,) — typiquement [10, 20, 40, 60, 80, 100], NaN si non dispo
  data/  chunks=(time=720, S=1, H=-1)
    u                   float32 (T, S, H)  m/s, NaN-padded
    v                   float32 (T, S, H)  m/s, NaN-padded
    wind_speed          float32 (T, S, H)  m/s, NaN-padded
    wind_dir            float32 (T, S, H)  degrees, NaN-padded
    t2m                 float32 (T, S, H)  K, NaN-padded
    rh                  float32 (T, S, H)  %, NaN-padded
  coords/
    time                int64 (T,) ns UTC hourly

attrs.global:
  sources             list of source strings included
  n_stations          int
  time_range          [start_iso, end_iso]
  n_pairings_total    int (≈ T × S × H avec mask non-NaN)
  created_at          ISO string
  schema_version      "1.0"
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from zarr.errors import UnstableSpecificationWarning

warnings.filterwarnings("ignore", category=UnstableSpecificationWarning, message=r"The data type .*NullTerminatedBytes.*")

# ── Compression par défaut ────────────────────────────────────────────────────

_DEFAULT_COMPRESSOR = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")


# ── Constantes du schéma ──────────────────────────────────────────────────────

STATION_COLUMNS = ("station_id", "lat", "lon", "elev", "source", "country", "z0_class_wc")
DATA_VARS = ("u", "v", "wind_speed", "wind_dir", "t2m", "rh")
DATA_CHUNK_TIME = 720

VARIABLE_META: dict[str, dict[str, str]] = {
    "u": {"long_name": "U-component of wind", "standard_name": "eastward_wind", "units": "m s-1", "coordinates": "time station height_m"},
    "v": {"long_name": "V-component of wind", "standard_name": "northward_wind", "units": "m s-1", "coordinates": "time station height_m"},
    "wind_speed": {"long_name": "Wind speed", "standard_name": "wind_speed", "units": "m s-1", "coordinates": "time station height_m"},
    "wind_dir": {"long_name": "Wind direction", "standard_name": "wind_from_direction", "units": "degree", "coordinates": "time station height_m"},
    "t2m": {"long_name": "Air temperature", "standard_name": "air_temperature", "units": "K", "coordinates": "time station height_m"},
    "rh": {"long_name": "Relative humidity", "standard_name": "relative_humidity", "units": "%", "coordinates": "time station height_m"},
}


# ── Fonctions principales ─────────────────────────────────────────────────────

def create_obs_store(
    path: str | Path,
    stations_df: pd.DataFrame,
    heights_array: np.ndarray,
    time_array: np.ndarray | None = None,
) -> zarr.Group:
    """
    Crée un store Zarr OBS unifié vide ou pré-alloué.

    Args:
        path:          Chemin de destination du store .zarr
        stations_df:   Table stations avec les colonnes du groupe stations/
        heights_array: Axe des hauteurs (H,) en mètres
        time_array:    Axe temporel optionnel datetime64[ns] UTC

    Returns:
        zarr.Group racine initialisée
    """
    path = Path(path)
    _validate_stations(stations_df)

    stations_df = stations_df.loc[:, STATION_COLUMNS].copy()
    heights = np.asarray(heights_array, dtype=np.float32)
    times = _to_time_int64(time_array) if time_array is not None else np.array([], dtype=np.int64)

    n_stations = len(stations_df)
    n_heights = len(heights)
    n_times = len(times)
    root = zarr.open_group(str(path), mode="w")

    root.attrs.update({
        "sources": sorted(stations_df["source"].astype(str).unique().tolist()),
        "n_stations": int(n_stations),
        "time_range": _time_range_attr(times),
        "n_pairings_total": 0,
        "created_at": _utc_now_iso(),
        "schema_version": "1.0",
    })

    stations = root.require_group("stations")
    _create_bytes_array(stations, "station_id", (n_stations,), "S16", stations_df["station_id"])
    _create_float_array(stations, "lat", (n_stations,), stations_df["lat"])
    _create_float_array(stations, "lon", (n_stations,), stations_df["lon"])
    _create_float_array(stations, "elev", (n_stations,), stations_df["elev"])
    _create_bytes_array(stations, "source", (n_stations,), "S16", stations_df["source"])
    _create_bytes_array(stations, "country", (n_stations,), "S2", stations_df["country"])
    arr = _create_array(stations, "z0_class_wc", shape=(n_stations,), dtype=np.int8,
                        chunks=(max(n_stations, 1),))
    arr[:] = stations_df["z0_class_wc"].to_numpy(dtype=np.int8)
    arr.attrs.update({"long_name": "ESA WorldCover 2021 roughness class mode in 1 km"})

    heights_group = root.require_group("heights")
    arr = _create_array(heights_group, "height_m", shape=(n_heights,), dtype=np.float32,
                        chunks=(max(n_heights, 1),))
    arr[:] = heights
    arr.attrs.update({"long_name": "observation height above ground", "units": "m"})

    coords = root.require_group("coords")
    arr = _create_array(coords, "time", shape=(n_times,), dtype=np.int64,
                        chunks=(max(n_times, 1),))
    if n_times:
        arr[:] = times
    arr.attrs.update({"note": "UTC timestamps as int64 nanoseconds (datetime64[ns])"})

    data = root.require_group("data")
    data_shape = (n_times, n_stations, n_heights)
    data_chunks = (DATA_CHUNK_TIME, 1, max(n_heights, 1))
    for var in DATA_VARS:
        arr = _create_array(data, var, shape=data_shape, dtype=np.float32,
                            chunks=data_chunks, fill_value=np.nan)
        arr.attrs.update(VARIABLE_META[var])

    return root


def append_obs_data(
    path: str | Path,
    source: str,
    station_id: str,
    time_array: np.ndarray,
    data_dict: dict[str, np.ndarray],
    height_idx_map: dict[float, int],
) -> None:
    """
    Écrit des observations pour une station et une source.

    Args:
        path:           Chemin du store OBS unifié
        source:         Source de la station
        station_id:     Identifiant station
        time_array:     Timestamps datetime64[ns] à écrire
        data_dict:      Variables OBS, chacune de forme (T, H')
        height_idx_map: Mapping hauteur observée -> index de l'axe H du store
    """
    if not height_idx_map:
        raise ValueError("height_idx_map must contain at least one height mapping")

    root = zarr.open_group(str(path), mode="r+")
    times = _to_time_int64(time_array)
    station_idx = _find_station_index(root, source=source, station_id=station_id)
    dest_height_idx = [int(idx) for idx in height_idx_map.values()]
    _validate_height_indices(root, dest_height_idx)
    time_indices = _ensure_times(root, times)

    old_count = _count_pairings_for_station(root, station_idx)
    for var in DATA_VARS:
        values = _values_for_var(var, data_dict, len(times), len(dest_height_idx))
        arr = root[f"data/{var}"]
        for row_idx, time_idx in enumerate(time_indices):
            current = np.asarray(arr[time_idx, station_idx, :], dtype=np.float32)
            current[dest_height_idx] = values[row_idx]
            arr[time_idx, station_idx, :] = current

    new_count = _count_pairings_for_station(root, station_idx)
    root.attrs["n_pairings_total"] = int(root.attrs.get("n_pairings_total", 0) + new_count - old_count)


def merge_obs_sources(out_path: str | Path, *src_paths: str | Path) -> zarr.Group:
    """
    Fusionne plusieurs stores OBS unifiés.

    TODO(M_G5 closure): implémenter la concaténation de l'axe stations, l'union
    des coordonnées temporelles et le NaN-padding inter-source. Cette mission ne
    teste pas la fusion finale.
    """
    raise NotImplementedError("merge_obs_sources is deferred to the M_G5 closure mission")


def read_obs(
    path: str | Path,
    *,
    sources: list[str] | tuple[str, ...] | None = None,
    station_ids: list[str] | tuple[str, ...] | None = None,
    time_range: tuple[Any, Any] | list[Any] | None = None,
    heights: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Lit les observations en format long, avec filtres appliqués avant lecture.

    Args:
        path:        Chemin du store OBS unifié
        sources:     Sources à conserver
        station_ids: Stations à conserver
        time_range:  Bornes temporelles inclusives (start, end)
        heights:     Hauteurs à conserver

    Returns:
        DataFrame long avec métadonnées station, temps, hauteur et variables OBS.
    """
    root = zarr.open_group(str(path), mode="r")
    stations = _stations_frame(root)
    station_mask = np.ones(len(stations), dtype=bool)
    if sources is not None:
        station_mask &= stations["source"].isin(set(sources)).to_numpy()
    if station_ids is not None:
        station_mask &= stations["station_id"].isin(set(station_ids)).to_numpy()
    station_indices = np.flatnonzero(station_mask)

    height_values = np.asarray(root["heights/height_m"][:], dtype=np.float32)
    height_indices = _select_height_indices(height_values, heights)

    time_values = np.asarray(root["coords/time"][:], dtype=np.int64)
    time_indices = _select_time_indices(time_values, time_range)
    if len(station_indices) == 0 or len(height_indices) == 0 or len(time_indices) == 0:
        return _empty_obs_frame()

    time_sel, selected_time_values = _time_selection(time_indices, time_values)
    selected_heights = height_values[height_indices]
    selected_times = pd.to_datetime(selected_time_values, utc=True)
    frames: list[pd.DataFrame] = []

    for station_idx in station_indices:
        var_data = {}
        for var in DATA_VARS:
            slab = np.asarray(root[f"data/{var}"][time_sel, int(station_idx), :], dtype=np.float32)
            if slab.ndim == 1:
                slab = slab.reshape(1, -1)
            var_data[var] = slab[:, height_indices]

        any_data = np.zeros((len(selected_time_values), len(height_indices)), dtype=bool)
        for values in var_data.values():
            any_data |= ~np.isnan(values)
        if not any_data.any():
            continue

        time_pos, height_pos = np.nonzero(any_data)
        station = stations.iloc[int(station_idx)]
        frame_data: dict[str, Any] = {
            "station_id": station["station_id"],
            "source": station["source"],
            "country": station["country"],
            "lat": np.float32(station["lat"]),
            "lon": np.float32(station["lon"]),
            "elev": np.float32(station["elev"]),
            "time": selected_times.take(time_pos),
            "height_m": selected_heights[height_pos].astype(np.float32),
        }
        for var in DATA_VARS:
            frame_data[var] = var_data[var][time_pos, height_pos]
        frames.append(pd.DataFrame(frame_data))

    if not frames:
        return _empty_obs_frame()
    return pd.concat(frames, ignore_index=True)


def list_stations(path: str | Path, *, source: str | None = None) -> pd.DataFrame:
    """
    Retourne le groupe stations/ comme DataFrame.

    Args:
        path:   Chemin du store OBS unifié
        source: Source optionnelle à conserver

    Returns:
        DataFrame stations, éventuellement filtré.
    """
    root = zarr.open_group(str(path), mode="r")
    stations = _stations_frame(root)
    if source is not None:
        stations = stations.loc[stations["source"] == source].reset_index(drop=True)
    return stations


# ── Helpers internes ──────────────────────────────────────────────────────────

def _create_array(group: zarr.Group, name: str, **kwargs: Any) -> zarr.Array:
    return group.create_array(name, compressors=[_DEFAULT_COMPRESSOR], overwrite=True, **kwargs)


def _create_bytes_array(
    group: zarr.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    values: pd.Series,
) -> None:
    arr = _create_array(group, name, shape=shape, dtype=dtype, chunks=(max(shape[0], 1),))
    arr[:] = values.astype(str).to_numpy(dtype=dtype)
    arr.attrs.update({"long_name": name})


def _create_float_array(group: zarr.Group, name: str, shape: tuple[int, ...], values: pd.Series) -> None:
    arr = _create_array(group, name, shape=shape, dtype=np.float32, chunks=(max(shape[0], 1),))
    arr[:] = values.to_numpy(dtype=np.float32)
    arr.attrs.update({"long_name": name, "units": "degrees" if name in {"lat", "lon"} else "m"})


def _validate_stations(stations_df: pd.DataFrame) -> None:
    missing = [col for col in STATION_COLUMNS if col not in stations_df.columns]
    if missing:
        raise ValueError(f"stations_df missing required columns: {missing}")


def _validate_height_indices(root: zarr.Group, indices: list[int]) -> None:
    n_heights = int(root["heights/height_m"].shape[0])
    bad = [idx for idx in indices if idx < 0 or idx >= n_heights]
    if bad:
        raise IndexError(f"height_idx_map contains out-of-range H indices: {bad}")


def _to_time_int64(time_array: np.ndarray | pd.DatetimeIndex | list[Any] | tuple[Any, ...]) -> np.ndarray:
    values = np.asarray(time_array)
    if values.size == 0:
        return np.array([], dtype=np.int64)
    return values.astype("datetime64[ns]").astype(np.int64)


def _ensure_times(root: zarr.Group, times: np.ndarray) -> np.ndarray:
    existing = np.asarray(root["coords/time"][:], dtype=np.int64)
    if len(existing) == 0:
        _resize_time_axis(root, len(times))
        root["coords/time"][:] = times
        root.attrs["time_range"] = _time_range_attr(times)
        return np.arange(len(times), dtype=np.int64)

    positions = _positions_in_existing_times(existing, times)
    if positions is not None:
        return positions

    if times[0] > existing[-1]:
        start = len(existing)
        _resize_time_axis(root, len(existing) + len(times))
        root["coords/time"][start:] = times
        root.attrs["time_range"] = _time_range_attr(root["coords/time"][:])
        return np.arange(start, start + len(times), dtype=np.int64)

    raise ValueError("time_array must match existing times or append strictly after the current time axis")


def _positions_in_existing_times(existing: np.ndarray, times: np.ndarray) -> np.ndarray | None:
    positions = np.searchsorted(existing, times)
    if np.any(positions >= len(existing)):
        return None
    if np.array_equal(existing[positions], times):
        return positions.astype(np.int64)
    return None


def _resize_time_axis(root: zarr.Group, n_times: int) -> None:
    root["coords/time"].resize((n_times,))
    n_stations = int(root.attrs["n_stations"])
    n_heights = int(root["heights/height_m"].shape[0])
    for var in DATA_VARS:
        root[f"data/{var}"].resize((n_times, n_stations, n_heights))


def _values_for_var(
    var: str,
    data_dict: dict[str, np.ndarray],
    n_times: int,
    n_heights_mapped: int,
) -> np.ndarray:
    if var not in data_dict:
        return np.full((n_times, n_heights_mapped), np.nan, dtype=np.float32)

    values = np.asarray(data_dict[var], dtype=np.float32)
    if values.ndim == 1 and n_heights_mapped == 1:
        values = values.reshape(-1, 1)
    expected = (n_times, n_heights_mapped)
    if values.shape != expected:
        raise ValueError(f"data_dict[{var!r}] has shape {values.shape}, expected {expected}")
    return values


def _find_station_index(root: zarr.Group, *, source: str, station_id: str) -> int:
    stations = _stations_frame(root)
    matches = np.flatnonzero(
        (stations["source"].to_numpy() == source) &
        (stations["station_id"].to_numpy() == station_id)
    )
    if len(matches) != 1:
        raise KeyError(f"expected one station for source={source!r}, station_id={station_id!r}; found {len(matches)}")
    return int(matches[0])


def _count_pairings_for_station(root: zarr.Group, station_idx: int) -> int:
    mask: np.ndarray | None = None
    for var in DATA_VARS:
        values = np.asarray(root[f"data/{var}"][:, station_idx, :], dtype=np.float32)
        var_mask = ~np.isnan(values)
        mask = var_mask if mask is None else (mask | var_mask)
    if mask is None:
        return 0
    return int(mask.sum())


def _stations_frame(root: zarr.Group) -> pd.DataFrame:
    stations = root["stations"]
    station_id = _decode_bytes(stations["station_id"][:])
    source = _decode_bytes(stations["source"][:])
    country = _decode_bytes(stations["country"][:])
    return pd.DataFrame({
        "station_id": station_id,
        "source": source,
        "country": country,
        "lat": np.asarray(stations["lat"][:], dtype=np.float32),
        "lon": np.asarray(stations["lon"][:], dtype=np.float32),
        "elev": np.asarray(stations["elev"][:], dtype=np.float32),
        "z0_class_wc": np.asarray(stations["z0_class_wc"][:], dtype=np.int8),
    })


def _decode_bytes(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else str(value).rstrip("\x00")
        for value in values
    ]


def _select_height_indices(
    height_values: np.ndarray,
    heights: list[float] | tuple[float, ...] | np.ndarray | None,
) -> np.ndarray:
    if heights is None:
        return np.arange(len(height_values), dtype=np.int64)
    wanted = np.asarray(heights, dtype=np.float32)
    mask = np.zeros(len(height_values), dtype=bool)
    for height in wanted:
        mask |= np.isclose(height_values, height, equal_nan=True)
    return np.flatnonzero(mask).astype(np.int64)


def _select_time_indices(time_values: np.ndarray, time_range: tuple[Any, Any] | list[Any] | None) -> np.ndarray:
    if time_range is None:
        return np.arange(len(time_values), dtype=np.int64)
    if len(time_range) != 2:
        raise ValueError("time_range must be a two-item (start, end) sequence")
    start, end = time_range
    mask = np.ones(len(time_values), dtype=bool)
    if start is not None:
        mask &= time_values >= _to_time_int64([start])[0]
    if end is not None:
        mask &= time_values <= _to_time_int64([end])[0]
    return np.flatnonzero(mask).astype(np.int64)


def _time_selection(indices: np.ndarray, time_values: np.ndarray) -> tuple[slice | np.ndarray, np.ndarray]:
    if len(indices) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        return selection, time_values[selection]
    return indices, time_values[indices]


def _empty_obs_frame() -> pd.DataFrame:
    columns = [
        "station_id", "source", "country", "lat", "lon", "elev", "time", "height_m",
        "u", "v", "wind_speed", "wind_dir", "t2m", "rh",
    ]
    return pd.DataFrame(columns=columns)


def _time_range_attr(times: np.ndarray) -> list[str | None]:
    if len(times) == 0:
        return [None, None]
    dt = pd.to_datetime(times, utc=True)
    return [dt[0].isoformat(), dt[-1].isoformat()]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
