"""
Download ERA5 hourly radiation/cloud fields from Google Earth Engine.

Writes a side Zarr store with flat top-level arrays:
  - ssrd: surface_solar_radiation_downwards, J m-2 accumulated over the
    preceding hour. Mean flux W m-2 = ssrd / 3600.
  - tcc : total_cloud_cover, fraction [0, 1].

The target grid is the canonical Europe 0.25 degree ERA5 grid used by the
existing hourly store: lon ascending, lat descending (N to S).
"""
from __future__ import annotations

import io
import logging
import time
import urllib.request
from pathlib import Path

import click
import numpy as np

GEE_PROJECT = "ee-guillaumemaitrejean"
ERA5_COLLECTION = "ECMWF/ERA5/HOURLY"
SSRD_BAND = "surface_solar_radiation_downwards"
TCC_BAND = "total_cloud_cover"
BANDS = [SSRD_BAND, TCC_BAND]
GRID_STEP_DEG = 0.25
DEFAULT_OUTPUT = Path("data/raw/era5_radcloud_jja2023.zarr")
DEFAULT_BBOX = "36,-10,52,10"

log = logging.getLogger("ingest_era5_radcloud_gee")


def parse_bbox(bbox_str: str) -> dict[str, float]:
    """Parse 'S,W,N,E' into a bbox dict."""
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise click.BadParameter(f"bbox must be 'S,W,N,E', got {bbox_str!r}")
    south, west, north, east = (float(p) for p in parts)
    if south >= north:
        raise click.BadParameter("bbox south must be < north")
    if west >= east:
        raise click.BadParameter("bbox west must be < east")
    return {"south": south, "west": west, "north": north, "east": east}


def build_grid(bbox: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the GEE-returned 0.25 degree lon/lat vectors for a bbox.

    NOTE: the authoritative lon/lat vectors come back from GEE itself via the
    self-describing `pixelLonLat()` bands (see `fetch_grid_and_coords`). GEE
    registers ERA5 pixel CENTERS offset by 0.125 deg from the native corner grid
    (transform origin -180.125 / 90.125), so the returned grid spans
    [west+0.125, east-0.125] in lon (one fewer cell than corner-counting) and
    [north-0.125, south+0.125] in lat. This estimate is used only as a sanity
    fallback / expected-shape hint; the real coords are read from GEE.
    """
    n_lon = int(round((bbox["east"] - bbox["west"]) / GRID_STEP_DEG))
    n_lat = int(round((bbox["north"] - bbox["south"]) / GRID_STEP_DEG)) + 1
    lons = (bbox["west"] + 0.125) + GRID_STEP_DEG * np.arange(n_lon, dtype=np.float64)
    lats = (bbox["north"] - 0.125) - GRID_STEP_DEG * np.arange(n_lat, dtype=np.float64)
    if not (lats[0] > lats[-1]):
        raise ValueError("latitudes must be stored north-to-south")
    return lats.astype(np.float32), lons.astype(np.float32)


def hourly_times(start: str, end: str, smoke: bool = False) -> np.ndarray:
    """Return hourly datetime64[ns] values for [start, end)."""
    start_h = np.datetime64(start).astype("datetime64[h]")
    end_h = np.datetime64(end).astype("datetime64[h]")
    if end_h <= start_h:
        raise click.BadParameter(f"empty time range: {start} -> {end}")
    times = np.arange(start_h, end_h, np.timedelta64(1, "h")).astype("datetime64[ns]")
    if smoke:
        times = times[:24]
    return times


def _write_array(group, name: str, data: np.ndarray, chunks: tuple[int, ...] | None = None):
    kwargs = {"data": data}
    if chunks is not None:
        kwargs["chunks"] = chunks
    return group.create_array(name, **kwargs)


def write_radcloud_zarr(
    out_path: Path | str,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    ssrd: np.ndarray,
    tcc: np.ndarray,
    *,
    bbox: str | None = None,
    failed_hours: list[str] | None = None,
) -> None:
    """Write radiation/cloud arrays to an Aqua-compatible Zarr v2 store."""
    import zarr

    out_path = Path(out_path)
    times_arr = np.asarray(times)
    if np.issubdtype(times_arr.dtype, np.datetime64):
        times_ns = times_arr.astype("datetime64[ns]").astype(np.int64)
    else:
        times_ns = times_arr.astype(np.int64)
    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    ssrd = np.asarray(ssrd, dtype=np.float32)
    tcc = np.asarray(tcc, dtype=np.float32)

    expected = (times_ns.size, lats.size, lons.size)
    if ssrd.shape != expected or tcc.shape != expected:
        raise ValueError(f"expected arrays {expected}, got ssrd={ssrd.shape} tcc={tcc.shape}")
    if lats.size > 1 and not (float(lats[0]) > float(lats[-1])):
        raise ValueError("latitudes must be stored north-to-south")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        store = zarr.open_group(str(out_path), mode="w", zarr_format=2)
    except TypeError:
        store = zarr.open_group(str(out_path), mode="w")

    coords = store.create_group("coords")
    _write_array(coords, "time", times_ns.astype(np.int64), (max(1, times_ns.size),))
    coords["time"].attrs.update({
        "note": "UTC timestamps as int64 (datetime64[ns])",
        "cadence_hours": 1,
    })
    _write_array(coords, "lat", lats, (max(1, lats.size),))
    coords["lat"].attrs.update({"long_name": "latitude", "units": "degrees_north"})
    _write_array(coords, "lon", lons, (max(1, lons.size),))
    coords["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})

    chunks = (min(240, max(1, times_ns.size)), max(1, lats.size), max(1, lons.size))
    ssrd_arr = _write_array(store, "ssrd", ssrd, chunks)
    ssrd_arr.attrs.update({
        "units": "J m-2",
        "note": "accumulated over preceding 1 hour (ERA5 hourly convention); "
                "flux W/m2 = ssrd/3600",
        "gee_band": SSRD_BAND,
    })
    tcc_arr = _write_array(store, "tcc", tcc, chunks)
    tcc_arr.attrs.update({
        "units": "1",
        "long_name": "total cloud cover fraction [0,1]",
        "gee_band": TCC_BAND,
    })

    attrs = {
        "Conventions": "CF-1.9",
        "title": "ERA5 hourly ssrd + total cloud cover over Europe bbox",
        "source": f"{ERA5_COLLECTION} (GEE)",
        "cadence_hours": 1,
        "bbox": bbox or "",
        "created_by": "ingest_era5_radcloud_gee.py",
    }
    if failed_hours:
        attrs["failed_hours"] = list(failed_hours)
    store.attrs.update(attrs)


def _cache_path(cache_dir: Path, ts: np.datetime64) -> Path:
    tag = str(ts.astype("datetime64[m]")).replace("-", "").replace(":", "").replace("T", "T")
    return cache_dir / f"radcloud_{tag}.npz"


def _load_cached(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            ssrd = np.asarray(data["ssrd"], dtype=np.float32)
            tcc = np.asarray(data["tcc"], dtype=np.float32)
    except Exception as exc:
        log.warning("Ignoring invalid cache %s (%s)", path.name, exc)
        return None
    if ssrd.shape != expected_shape or tcc.shape != expected_shape:
        log.warning("Ignoring cache %s with shape ssrd=%s tcc=%s", path.name, ssrd.shape, tcc.shape)
        return None
    return ssrd, tcc


def _save_cached(path: Path, ssrd: np.ndarray, tcc: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ssrd=ssrd.astype(np.float32), tcc=tcc.astype(np.float32))
    tmp.replace(path)


def parse_region_rows(rows, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map getRegion rows onto the canonical (time, lat, lon) grid.

    The row stream order is intentionally ignored. Lat/lon indices are nearest
    matches against the caller-provided axes, so the N->S latitude orientation
    from `fetch_grid_and_coords` is preserved.
    """
    if not rows:
        raise ValueError("GEE getRegion returned no header row")
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    if lats_arr.size == 0 or lons_arr.size == 0:
        raise ValueError("lats/lons must be non-empty")

    header = [str(c) for c in rows[0]]
    try:
        lon_col = header.index("longitude")
        lat_col = header.index("latitude")
        time_col = header.index("time")
        ssrd_col = header.index(SSRD_BAND)
        tcc_col = header.index(TCC_BAND)
    except ValueError as exc:
        raise ValueError(f"getRegion header missing required column: {header}") from exc

    coord_time_max_col = max(lon_col, lat_col, time_col)
    valid_rows = []
    time_ms = []
    for row in rows[1:]:
        if len(row) <= coord_time_max_col or row[time_col] is None:
            continue
        valid_rows.append(row)
        time_ms.append(int(row[time_col]))

    if not valid_rows:
        empty = np.full((0, lats_arr.size, lons_arr.size), np.nan, dtype=np.float32)
        return empty, empty.copy(), np.asarray([], dtype="datetime64[ns]")

    time_ns_by_row = (
        np.asarray(time_ms, dtype=np.int64)
        .astype("datetime64[ms]")
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    unique_time_ns = np.unique(time_ns_by_row)
    times_block = unique_time_ns.astype("datetime64[ns]")
    time_index = {int(ns): idx for idx, ns in enumerate(unique_time_ns)}

    ssrd_block = np.full((times_block.size, lats_arr.size, lons_arr.size), np.nan, dtype=np.float32)
    tcc_block = np.full_like(ssrd_block, np.nan)
    lat_index_cache: dict[float, int] = {}
    lon_index_cache: dict[float, int] = {}

    # A query region may be buffered slightly larger than the canonical grid so
    # that getRegion includes the bbox-edge pixels it otherwise drops. Buffer
    # pixels that do not coincide with a canonical cell (nearest distance beyond
    # half a grid step) are discarded rather than snapped onto an edge cell.
    half_step = 0.5 * GRID_STEP_DEG + 1e-4

    for row, time_ns in zip(valid_rows, time_ns_by_row):
        if len(row) <= max(lon_col, lat_col) or row[lon_col] is None or row[lat_col] is None:
            continue
        row_lat = float(row[lat_col])
        row_lon = float(row[lon_col])
        i_lat = lat_index_cache.get(row_lat)
        if i_lat is None:
            i_lat_cand = int(np.argmin(np.abs(lats_arr - row_lat)))
            i_lat = i_lat_cand if abs(lats_arr[i_lat_cand] - row_lat) <= half_step else -1
            lat_index_cache[row_lat] = i_lat
        if i_lat < 0:
            continue
        j_lon = lon_index_cache.get(row_lon)
        if j_lon is None:
            j_lon_cand = int(np.argmin(np.abs(lons_arr - row_lon)))
            j_lon = j_lon_cand if abs(lons_arr[j_lon_cand] - row_lon) <= half_step else -1
            lon_index_cache[row_lon] = j_lon
        if j_lon < 0:
            continue
        k_time = time_index[int(time_ns)]

        ssrd_value = row[ssrd_col] if len(row) > ssrd_col else None
        tcc_value = row[tcc_col] if len(row) > tcc_col else None
        if ssrd_value is not None:
            ssrd_block[k_time, i_lat, j_lon] = np.float32(ssrd_value)
        if tcc_value is not None:
            tcc_block[k_time, i_lat, j_lon] = np.float32(tcc_value)

    return ssrd_block, tcc_block, times_block


def _structured_field(arr: np.ndarray, band: str, field_index: int) -> np.ndarray:
    if arr.dtype.fields:
        names = list(arr.dtype.fields)
        if band in arr.dtype.fields:
            return np.asarray(arr[band], dtype=np.float32)
        return np.asarray(arr[names[field_index]], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return np.asarray(arr[..., field_index], dtype=np.float32)
    raise ValueError(f"cannot extract {band!r} from NPY array dtype={arr.dtype} shape={arr.shape}")


def _download_npy(image, region, scale: int) -> np.ndarray:
    """getDownloadURL(NPY) → structured numpy array (one record per pixel)."""
    params = {
        "format": "NPY",
        "region": region,
        "scale": int(scale),
        "crs": "EPSG:4326",
    }
    url = image.getDownloadURL(params)
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
    return np.load(io.BytesIO(payload), allow_pickle=False)


def fetch_grid_and_coords(collection, region, scale: int, sample_iso: str):
    """Probe one hour with pixelLonLat bands → authoritative (lats N->S, lons asc).

    GEE registers ERA5 pixel centres on a grid offset by 0.125 deg from the
    bbox edges, so we read the real lon/lat back from GEE rather than guessing.
    Returns (lats float32 [n_lat], lons float32 [n_lon]).
    """
    import ee

    sample = np.datetime64(sample_iso).astype("datetime64[h]")
    t0, t1 = str(sample), str(sample + np.timedelta64(1, "h"))
    image = collection.filterDate(t0, t1).first().select(BANDS).addBands(
        ee.Image.pixelLonLat()
    )
    raw = _download_npy(image, region, scale)
    lon2d = _structured_field(raw, "longitude", 2)
    lat2d = _structured_field(raw, "latitude", 3)
    lons = np.asarray(lon2d[0, :], dtype=np.float32)
    lats = np.asarray(lat2d[:, 0], dtype=np.float32)
    if lats.size > 1 and not (float(lats[0]) > float(lats[-1])):
        # GEE returned S->N; the data fetch below uses the same orientation so
        # flipping here would desync. Enforce N->S by reversing rows downstream.
        raise ValueError("GEE returned latitudes S->N; expected N->S")
    return lats, lons


def fetch_hour_npy(image, region, scale: int, expected_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one two-band ERA5 hourly image via GEE NPY download.

    `expected_shape` is the (n_lat, n_lon) grid established by
    `fetch_grid_and_coords`. GEE returns a deterministic grid per (region,
    scale, crs) so shapes match exactly; a mismatch is a hard error.
    """
    raw = _download_npy(image, region, scale)
    ssrd = _structured_field(raw, SSRD_BAND, 0)
    tcc = _structured_field(raw, TCC_BAND, 1)
    if ssrd.shape != expected_shape or tcc.shape != expected_shape:
        raise ValueError(
            f"GEE NPY shape mismatch: ssrd={ssrd.shape} tcc={tcc.shape} "
            f"expected={expected_shape}"
        )
    return ssrd.astype(np.float32, copy=False), tcc.astype(np.float32, copy=False)


def download_with_cache(
    collection,
    region,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cache_dir: Path,
    scale: int,
    retries: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download/cache each hour. Failed hours are left as NaN and reported."""
    expected_shape = (lats.size, lons.size)
    ssrd_all = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    tcc_all = np.full_like(ssrd_all, np.nan)
    failures: list[str] = []

    for idx, ts in enumerate(times):
        cache_file = _cache_path(cache_dir, ts)
        cached = _load_cached(cache_file, expected_shape)
        if cached is not None:
            ssrd_all[idx], tcc_all[idx] = cached
            if (idx + 1) % 24 == 0 or idx == times.size - 1:
                log.info("Loaded cached hour %d/%d", idx + 1, times.size)
            continue

        ts_h = ts.astype("datetime64[h]")
        t0 = str(ts_h)
        t1 = str(ts_h + np.timedelta64(1, "h"))
        image = collection.filterDate(t0, t1).first().select(BANDS)
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                ssrd, tcc = fetch_hour_npy(image, region, scale, expected_shape)
                ssrd_all[idx], tcc_all[idx] = ssrd, tcc
                _save_cached(cache_file, ssrd, tcc)
                break
            except Exception as exc:
                last_exc = exc
                log.warning("GEE fetch failed %s attempt %d/%d: %s", t0, attempt, retries, exc)
                time.sleep(min(30.0, 2.0 * attempt))
        else:
            failures.append(t0)
            log.error("Giving up on %s after %d retries: %s", t0, retries, last_exc)

        if (idx + 1) % 24 == 0 or idx == times.size - 1:
            log.info("Processed hour %d/%d", idx + 1, times.size)

    return ssrd_all, tcc_all, failures


def download_batched(
    collection,
    region,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cache_dir: Path,
    scale: int,
    chunk_days: int = 8,
    retries: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download/cache multi-day getRegion chunks. Failed ranges remain NaN."""
    ssrd_all = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    tcc_all = np.full_like(ssrd_all, np.nan)
    failures: list[str] = []
    retries = max(1, int(retries))
    chunk_days = max(1, int(chunk_days))
    global_time_ns = times.astype("datetime64[ns]").astype(np.int64)
    global_time_index = {int(ns): idx for idx, ns in enumerate(global_time_ns)}

    def date_tag(ts: np.datetime64) -> str:
        return str(ts.astype("datetime64[D]")).replace("-", "")

    def cache_path(t0: np.datetime64, t1: np.datetime64) -> Path:
        return cache_dir / f"radcloud_chunk_{date_tag(t0)}_{date_tag(t1)}.npz"

    def load_chunk(path: Path):
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                ssrd = np.asarray(data["ssrd"], dtype=np.float32)
                tcc = np.asarray(data["tcc"], dtype=np.float32)
                chunk_times = np.asarray(data["times"])
        except Exception as exc:
            log.warning("Ignoring invalid chunk cache %s (%s)", path.name, exc)
            return None
        if np.issubdtype(chunk_times.dtype, np.datetime64):
            chunk_times = chunk_times.astype("datetime64[ns]")
        else:
            chunk_times = chunk_times.astype(np.int64).astype("datetime64[ns]")
        expected_tail = (lats.size, lons.size)
        if (
            ssrd.shape != tcc.shape
            or ssrd.ndim != 3
            or ssrd.shape[1:] != expected_tail
            or ssrd.shape[0] != chunk_times.size
        ):
            log.warning("Ignoring cache %s with shape ssrd=%s tcc=%s", path.name, ssrd.shape, tcc.shape)
            return None
        return ssrd, tcc, chunk_times

    def save_chunk(path: Path, ssrd: np.ndarray, tcc: np.ndarray, chunk_times: np.ndarray) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            ssrd=ssrd.astype(np.float32),
            tcc=tcc.astype(np.float32),
            times=chunk_times.astype("datetime64[ns]"),
        )
        tmp.replace(path)

    def place_block(ssrd_block: np.ndarray, tcc_block: np.ndarray, chunk_times: np.ndarray) -> int:
        filled = 0
        for k_time, ns in enumerate(chunk_times.astype("datetime64[ns]").astype(np.int64)):
            global_idx = global_time_index.get(int(ns))
            if global_idx is None:
                continue
            ssrd_all[global_idx] = ssrd_block[k_time]
            tcc_all[global_idx] = tcc_block[k_time]
            filled += 1
        return filled

    def is_region_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in ("too many values", "image.getregion", "exceeds", "memory"))

    def fail_range(t0: np.datetime64, t1: np.datetime64, exc: Exception | None) -> None:
        label = f"{str(t0)}..{str(t1)}"
        failures.append(label)
        log.error("Giving up on chunk %s after %d retries: %s", label, retries, exc)

    def process_range(t0: np.datetime64, t1: np.datetime64, label: str) -> None:
        path = cache_path(t0, t1)
        cached = load_chunk(path)
        if cached is not None:
            ssrd_block, tcc_block, chunk_times = cached
            filled = place_block(ssrd_block, tcc_block, chunk_times)
            log.info("Chunk %s %s..%s: cached -> filled %d hours", label, str(t0), str(t1), filled)
            return

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                rows = collection.filterDate(str(t0), str(t1)).getRegion(region, int(scale)).getInfo()
                ssrd_block, tcc_block, chunk_times = parse_region_rows(rows, lats, lons)
                filled = place_block(ssrd_block, tcc_block, chunk_times)
                save_chunk(path, ssrd_block, tcc_block, chunk_times)
                log.info(
                    "Chunk %s %s..%s: %d rows -> filled %d hours",
                    label, str(t0), str(t1), max(0, len(rows) - 1), filled,
                )
                return
            except Exception as exc:
                last_exc = exc
                if is_region_limit_error(exc):
                    span_hours = int((t1 - t0) / np.timedelta64(1, "h"))
                    span_days = max(1, int(np.ceil(span_hours / 24.0)))
                    if span_days <= 1:
                        fail_range(t0, t1, exc)
                        return
                    split_days = max(1, span_days // 2)
                    mid = t0 + np.timedelta64(split_days, "D")
                    if mid <= t0 or mid >= t1:
                        fail_range(t0, t1, exc)
                        return
                    log.warning(
                        "Chunk %s %s..%s too large (%s); splitting at %s",
                        label, str(t0), str(t1), exc, str(mid),
                    )
                    process_range(t0, mid, f"{label}a")
                    process_range(mid, t1, f"{label}b")
                    return
                log.warning(
                    "GEE batch fetch failed %s..%s attempt %d/%d: %s",
                    str(t0), str(t1), attempt, retries, exc,
                )
                time.sleep(min(30.0, 2.0 * attempt))
        fail_range(t0, t1, last_exc)

    start_h = times[0].astype("datetime64[h]")
    end_h = times[-1].astype("datetime64[h]") + np.timedelta64(1, "h")
    step = np.timedelta64(chunk_days, "D")
    ranges: list[tuple[np.datetime64, np.datetime64]] = []
    t0 = start_h
    while t0 < end_h:
        t1 = t0 + step
        if t1 > end_h:
            t1 = end_h
        ranges.append((t0, t1))
        t0 = t1

    for idx, (t0, t1) in enumerate(ranges, start=1):
        process_range(t0, t1, f"{idx}/{len(ranges)}")

    return ssrd_all, tcc_all, failures


def verify_store(path: Path) -> None:
    """Re-open the written store and log shapes plus one sample value."""
    import zarr

    g = zarr.open_group(str(path), mode="r")
    ssrd = g["ssrd"]
    tcc = g["tcc"]
    i = min(1, ssrd.shape[1] - 1)
    j = min(1, ssrd.shape[2] - 1)
    log.info("Verification - coords/time shape=%s", g["coords/time"].shape)
    log.info("Verification - coords/lat shape=%s lon shape=%s", g["coords/lat"].shape, g["coords/lon"].shape)
    log.info("Verification - ssrd shape=%s tcc shape=%s", ssrd.shape, tcc.shape)
    log.info("Verification - sample ssrd=%.3g J/m2 tcc=%.3g", float(ssrd[0, i, j]), float(tcc[0, i, j]))


@click.command()
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, show_default=True)
@click.option("--start", default="2023-06-01", show_default=True, help="Start date YYYY-MM-DD.")
@click.option("--end", default="2023-09-01", show_default=True, help="Exclusive end date YYYY-MM-DD.")
@click.option("--bbox", default=DEFAULT_BBOX, show_default=True, help="Bounding box 'S,W,N,E'.")
@click.option("--smoke", is_flag=True, default=False, help="Only download a short initial time window.")
@click.option("--scale", default=27830, show_default=True, type=int, help="GEE download scale in meters.")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None, help=".npz cache directory.")
@click.option("--allow-gaps", is_flag=True, default=False, help="Write NaNs for failed hours instead of raising.")
@click.option("--retries", default=3, show_default=True, type=int, help="Retries per hourly image.")
@click.option(
    "--mode",
    type=click.Choice(["batched", "hourly"]),
    default="batched",
    show_default=True,
    help="Download mode.",
)
@click.option("--chunk-days", default=8, show_default=True, type=int, help="Batched getRegion chunk size in days.")
def main(output: Path, start: str, end: str, bbox: str, smoke: bool, scale: int,
         cache_dir: Path | None, allow_gaps: bool, retries: int, mode: str,
         chunk_days: int) -> None:
    """Ingest ERA5 ssrd + tcc from GEE and write the side Zarr store."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bbox_d = parse_bbox(bbox)
    est_lats, est_lons = build_grid(bbox_d)  # sanity fallback only
    times = hourly_times(start, end, smoke=smoke)
    cache = cache_dir or output.parent / "_cache_radcloud"
    cache.mkdir(parents=True, exist_ok=True)

    log.info("Initializing Google Earth Engine project=%s", GEE_PROJECT)
    import ee

    ee.Initialize(project=GEE_PROJECT)
    region = ee.Geometry.Rectangle(
        [bbox_d["west"], bbox_d["south"], bbox_d["east"], bbox_d["north"]],
        proj="EPSG:4326",
        geodesic=False,
    )
    collection = ee.ImageCollection(ERA5_COLLECTION).filterDate(start, end).select(BANDS)

    # Authoritative coords come from GEE itself (pixelLonLat), not build_grid.
    lats, lons = fetch_grid_and_coords(collection, region, scale, str(times[0]))
    log.info(
        "Grid: %d times x %d lat x %d lon (lat %.3f..%.3f, lon %.3f..%.3f; "
        "build_grid estimate was %dx%d)",
        times.size, lats.size, lons.size,
        float(lats[0]), float(lats[-1]), float(lons[0]), float(lons[-1]),
        est_lats.size, est_lons.size,
    )

    if mode == "batched":
        # getRegion drops the outermost bbox-edge pixels that getDownloadURL(NPY)
        # keeps (e.g. the southern-most canonical lat row). Query a slightly
        # larger rectangle so those edge pixels are returned; parse_region_rows
        # snaps each row to the nearest canonical cell only within half a grid
        # step, so the extra ring of buffer pixels is discarded and every
        # canonical cell is filled with its exact value.
        buf = GRID_STEP_DEG
        region_query = ee.Geometry.Rectangle(
            [bbox_d["west"] - buf, bbox_d["south"] - buf,
             bbox_d["east"] + buf, bbox_d["north"] + buf],
            proj="EPSG:4326",
            geodesic=False,
        )
        ssrd, tcc, failures = download_batched(
            collection, region_query, times[:48] if smoke else times, lats, lons, cache,
            scale, chunk_days=max(1, chunk_days), retries=max(1, retries),
        )
        if smoke:
            times = times[:48]
    else:
        ssrd, tcc, failures = download_with_cache(
            collection, region, times, lats, lons, cache, scale, max(1, retries)
        )
    if failures and not allow_gaps:
        raise RuntimeError(f"{len(failures)} GEE hours failed; first failures: {failures[:5]}")

    log.info("Writing Zarr v2 store: %s", output)
    write_radcloud_zarr(output, times, lats, lons, ssrd, tcc, bbox=bbox, failed_hours=failures)
    verify_store(output)
    log.info("DONE")


if __name__ == "__main__":
    main()
