"""
Build input-only v2 grid.zarr stores for frozen station/timestamp cases.

The station FWI validation pipeline selects independent SYNOP/OMM stations and
fire-weather timestamps. This script turns those rows into the native surrogate
input contract:
  - 180 x 180 terrain over a 6 km x 6 km patch
  - terrain-following z coordinates
  - ERA5 3 x 3 pressure/surface predictors at the requested timestamp
  - station metadata and z0_eff

It does not create CFD targets. The output stores are for inference with
run_station_surrogate_inference.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


logger = logging.getLogger(__name__)

NI = 180
NJ = 180
NK = 40
HALF_EXTENT_M = 3000.0
DX = 2.0 * HALF_EXTENT_M / NI
M_PER_DEG_LAT = 111_320.0
COPDEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


@dataclass(frozen=True)
class CaseBuildResult:
    case_id: str
    station_id: str
    timestamp_utc: str
    lat: float
    lon: float
    grid_zarr: str
    terrain_source: str
    terrain_min_m: float
    terrain_max_m: float
    era5_time: str
    status: str
    error: str = ""


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def zarr_times_to_ns(times: np.ndarray) -> np.ndarray:
    if np.issubdtype(times.dtype, np.datetime64):
        return times.astype("datetime64[ns]").astype(np.int64)
    return times.astype(np.int64)


def ns_to_iso(value: int | np.integer) -> str:
    return pd.Timestamp(int(value), unit="ns").isoformat()


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(values.astype(float) - float(value))))


def centered_three_indices(values: np.ndarray, value: float) -> np.ndarray:
    idx = nearest_index(values, value)
    if len(values) < 3:
        raise ValueError("Need at least 3 ERA5 grid points")
    start = max(0, min(idx - 1, len(values) - 3))
    return np.arange(start, start + 3, dtype=np.int64)


def station_bounds(lat: float, lon: float) -> tuple[float, float, float, float]:
    cos_lat = max(math.cos(math.radians(lat)), 1e-3)
    dlat = HALF_EXTENT_M / M_PER_DEG_LAT
    dlon = HALF_EXTENT_M / (M_PER_DEG_LAT * cos_lat)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (np.arange(NJ, dtype=np.float32) + 0.5) * DX - HALF_EXTENT_M
    y = (np.arange(NI, dtype=np.float32) + 0.5) * DX - HALF_EXTENT_M
    agl = np.linspace(0.0, 100.0, NK, dtype=np.float32)
    return x, y, agl


def zarr_write(group, name: str, data: np.ndarray) -> None:
    arr = np.asarray(data)
    group.create_dataset(name, data=arr, shape=arr.shape, dtype=arr.dtype)


def copdem_token_lat(lat_floor: int) -> str:
    hemi = "N" if lat_floor >= 0 else "S"
    return f"{hemi}{abs(lat_floor):02d}"


def copdem_token_lon(lon_floor: int) -> str:
    hemi = "E" if lon_floor >= 0 else "W"
    return f"{hemi}{abs(lon_floor):03d}"


def copdem_tile_name(lat_floor: int, lon_floor: int) -> str:
    return (
        "Copernicus_DSM_COG_10_"
        f"{copdem_token_lat(lat_floor)}_00_{copdem_token_lon(lon_floor)}_00_DEM.tif"
    )


def copdem_tile_url(lat_floor: int, lon_floor: int) -> str:
    name = copdem_tile_name(lat_floor, lon_floor).removesuffix(".tif")
    return f"{COPDEM_BASE}/{name}/{name}.tif"


def download_copdem_tile(tile_dir: Path, lat_floor: int, lon_floor: int) -> Path:
    import urllib.request

    tile_dir.mkdir(parents=True, exist_ok=True)
    name = copdem_tile_name(lat_floor, lon_floor)
    dest = tile_dir / name
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    url = copdem_tile_url(lat_floor, lon_floor)
    logger.info("Downloading missing COPDEM tile %s", name)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def needed_copdem_tiles(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    west, south, east, north = bounds
    eps = 1e-9
    lat_range = range(math.floor(south), math.floor(north - eps) + 1)
    lon_range = range(math.floor(west), math.floor(east - eps) + 1)
    return [(ilat, ilon) for ilat in lat_range for ilon in lon_range]


def resolve_copdem_tiles(
    tile_dir: Path,
    bounds: tuple[float, float, float, float],
    *,
    download_missing: bool,
) -> list[Path]:
    paths = []
    missing = []
    for lat_floor, lon_floor in needed_copdem_tiles(bounds):
        name = copdem_tile_name(lat_floor, lon_floor)
        direct = tile_dir / name
        if direct.exists():
            paths.append(direct)
            continue
        matches = list(tile_dir.glob(f"*{copdem_token_lat(lat_floor)}_00_{copdem_token_lon(lon_floor)}_00_DEM.tif"))
        if matches:
            paths.append(matches[0])
        elif download_missing:
            paths.append(download_copdem_tile(tile_dir, lat_floor, lon_floor))
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Missing COPDEM tiles in {tile_dir}: {missing}")
    return paths


def terrain_from_copdem(
    lat: float,
    lon: float,
    tile_dir: Path,
    *,
    download_missing: bool,
) -> tuple[np.ndarray, str]:
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError("terrain-mode=copdem requires rasterio") from exc

    bounds = station_bounds(lat, lon)
    tile_paths = resolve_copdem_tiles(tile_dir, bounds, download_missing=download_missing)
    srcs = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, src_transform = merge(srcs, bounds=bounds)
        dst = np.full((NI, NJ), np.nan, dtype=np.float32)
        dst_transform = from_bounds(*bounds, width=NJ, height=NI)
        reproject(
            source=mosaic[0],
            destination=dst,
            src_transform=src_transform,
            src_crs=srcs[0].crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=srcs[0].nodata,
            dst_nodata=np.nan,
        )
    finally:
        for src in srcs:
            src.close()

    # Raster row 0 is north. The surrogate grid uses y increasing with axis 0,
    # so store rows from south to north.
    terrain = np.flipud(dst).astype(np.float32)
    if not np.isfinite(terrain).all():
        finite = terrain[np.isfinite(terrain)]
        if finite.size == 0:
            raise ValueError("COPDEM patch contains no finite elevation")
        terrain = np.where(np.isfinite(terrain), terrain, float(np.nanmedian(finite))).astype(np.float32)
    return terrain, "copdem"


def terrain_from_static_netcdf(
    lat: float,
    lon: float,
    static_nc: Path,
) -> tuple[np.ndarray, str]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("terrain-mode=coarse-netcdf requires netCDF4") from exc

    x, y, _ = grid_axes()
    cos_lat = max(math.cos(math.radians(lat)), 1e-3)
    lat_grid = lat + y[:, None] / M_PER_DEG_LAT
    lon_grid = lon + x[None, :] / (M_PER_DEG_LAT * cos_lat)

    with Dataset(static_nc) as ds:
        lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        elevation = np.asarray(ds.variables["elevation"][:], dtype=np.float32)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        elevation = elevation[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        elevation = elevation[:, ::-1]

    i1 = np.searchsorted(lats, lat_grid, side="left")
    j1 = np.searchsorted(lons, lon_grid, side="left")
    i1 = np.clip(i1, 1, len(lats) - 1)
    j1 = np.clip(j1, 1, len(lons) - 1)
    i0 = i1 - 1
    j0 = j1 - 1

    lat0 = lats[i0]
    lat1 = lats[i1]
    lon0 = lons[j0]
    lon1 = lons[j1]
    wy = np.divide(lat_grid - lat0, lat1 - lat0, out=np.zeros_like(lat_grid), where=(lat1 != lat0))
    wx = np.divide(lon_grid - lon0, lon1 - lon0, out=np.zeros_like(lon_grid), where=(lon1 != lon0))

    q00 = elevation[i0, j0]
    q01 = elevation[i0, j1]
    q10 = elevation[i1, j0]
    q11 = elevation[i1, j1]
    terrain = (
        (1.0 - wy) * ((1.0 - wx) * q00 + wx * q01)
        + wy * ((1.0 - wx) * q10 + wx * q11)
    )
    return terrain.astype(np.float32), "coarse-netcdf"


def build_terrain(
    lat: float,
    lon: float,
    *,
    terrain_mode: str,
    tile_dir: Path | None,
    static_nc: Path | None,
    fallback_static_nc: Path | None,
    download_missing_copdem: bool,
) -> tuple[np.ndarray, str]:
    if terrain_mode == "copdem":
        if tile_dir is None:
            raise ValueError("--tile-dir is required for --terrain-mode copdem")
        try:
            return terrain_from_copdem(lat, lon, tile_dir, download_missing=download_missing_copdem)
        except Exception:
            if fallback_static_nc is None:
                raise
            logger.exception("COPDEM failed; falling back to coarse terrain at %.4f, %.4f", lat, lon)
            return terrain_from_static_netcdf(lat, lon, fallback_static_nc)
    if terrain_mode == "coarse-netcdf":
        if static_nc is None:
            raise ValueError("--terrain-static-nc is required for --terrain-mode coarse-netcdf")
        return terrain_from_static_netcdf(lat, lon, static_nc)
    raise ValueError(f"Unknown terrain mode: {terrain_mode}")


def sample_era5_at_station(
    era5,
    timestamp: str,
    lat: float,
    lon: float,
) -> tuple[dict[str, object], str]:
    times = np.asarray(era5["coords/time"][:])
    time_ns = zarr_times_to_ns(times)
    target = np.datetime64(pd.Timestamp(timestamp), "ns").astype(np.int64)
    it = int(np.argmin(np.abs(time_ns - target)))

    lats = np.asarray(era5["coords/lat"][:], dtype=np.float64)
    lons = np.asarray(era5["coords/lon"][:], dtype=np.float64)
    lat_idx = centered_three_indices(lats, lat)
    lon_idx = centered_three_indices(lons, lon)
    lat_idx = np.asarray(sorted(lat_idx, key=lambda i: lats[int(i)]), dtype=np.int64)
    lon_idx = np.asarray(sorted(lon_idx, key=lambda i: lons[int(i)]), dtype=np.int64)

    levels = np.asarray(era5["coords/level"][:], dtype=np.float32)
    out_3d: dict[str, np.ndarray] = {}
    for source_var, target_var in [("u", "u"), ("v", "v"), ("t", "T"), ("q", "q")]:
        arr = np.asarray(era5[f"pressure/{source_var}"][it, :, lat_idx, :][:, :, lon_idx], dtype=np.float32)
        out_3d[target_var] = np.transpose(arr, (1, 2, 0))

    out_surface: dict[str, np.ndarray] = {}
    for var in ("t2m", "d2m", "u10", "v10"):
        out_surface[var] = np.asarray(era5[f"surface/{var}"][it, lat_idx, :][:, lon_idx], dtype=np.float32)

    return {
        "era5_pressure_levels": levels,
        "era5_3d": out_3d,
        "era5_surface": out_surface,
        "era5_lat_indices": lat_idx.astype(int).tolist(),
        "era5_lon_indices": lon_idx.astype(int).tolist(),
        "era5_lats": lats[lat_idx].astype(float).tolist(),
        "era5_lons": lons[lon_idx].astype(float).tolist(),
    }, ns_to_iso(time_ns[it])


def row_case_id(row: pd.Series) -> str:
    if "case_id" in row and pd.notna(row["case_id"]):
        return str(row["case_id"])
    sid = str(row["station_id"])
    ts = pd.Timestamp(row["timestamp_utc"]).strftime("%Y%m%dT%H%MZ")
    return f"{sid}_{ts}"


def row_timestamp(row: pd.Series, target_hour_utc: int) -> str:
    if "timestamp_utc" in row and pd.notna(row["timestamp_utc"]):
        return pd.Timestamp(row["timestamp_utc"]).strftime("%Y-%m-%d %H:%M:%S")
    return pd.Timestamp(row["date"]).replace(hour=target_hour_utc).strftime("%Y-%m-%d %H:%M:%S")


def write_grid_zarr(
    output: Path,
    *,
    case_id: str,
    station_id: str,
    timestamp_utc: str,
    lat: float,
    lon: float,
    alt_m: float | None,
    z0_eff: float,
    terrain: np.ndarray,
    terrain_source: str,
    era5_payload: dict[str, object],
    era5_time: str,
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            return
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    x, y, agl = grid_axes()
    z = terrain[:, :, None] + agl[None, None, :]

    g = zarr.open_group(str(output), mode="w")
    coords = g.create_group("coords")
    zarr_write(coords, "x", x.astype(np.float32))
    zarr_write(coords, "y", y.astype(np.float32))
    zarr_write(coords, "z", z.astype(np.float32))
    zarr_write(coords, "z_agl", agl.astype(np.float32))

    inp = g.create_group("input")
    zarr_write(inp, "terrain", terrain.astype(np.float32))
    inp.attrs["z0_eff"] = float(z0_eff)
    inp.attrs["lat"] = float(lat)
    inp.attrs["lon"] = float(lon)
    if alt_m is not None and np.isfinite(alt_m):
        inp.attrs["station_alt_m"] = float(alt_m)
    zarr_write(inp, "era5_pressure_levels", np.asarray(era5_payload["era5_pressure_levels"], dtype=np.float32))

    e3d = inp.create_group("era5_3d")
    for var, arr in era5_payload["era5_3d"].items():
        zarr_write(e3d, str(var), np.asarray(arr, dtype=np.float32))

    esrf = inp.create_group("era5_surface")
    for var, arr in era5_payload["era5_surface"].items():
        zarr_write(esrf, str(var), np.asarray(arr, dtype=np.float32))

    meta = inp.create_group("inflow_meta")
    meta.attrs["timestamp"] = timestamp_utc
    meta.attrs["site_id"] = station_id
    meta.attrs["case_id"] = case_id
    meta.attrs["source"] = "station_fwi_validation"
    meta.attrs["terrain_source"] = terrain_source
    meta.attrs["era5_time"] = era5_time
    meta.attrs["era5_lat_indices"] = json.dumps(era5_payload["era5_lat_indices"])
    meta.attrs["era5_lon_indices"] = json.dumps(era5_payload["era5_lon_indices"])
    meta.attrs["era5_lats"] = json.dumps(era5_payload["era5_lats"])
    meta.attrs["era5_lons"] = json.dumps(era5_payload["era5_lons"])

    g.attrs.update(
        {
            "schema_version": "v2.0-input-only",
            "site_id": station_id,
            "case_id": case_id,
            "timestamp_utc": timestamp_utc,
            "grid_shape": [NI, NJ, NK],
            "dx_m": DX,
            "half_extent_m": HALF_EXTENT_M,
            "target_agl_min_m": float(agl.min()),
            "target_agl_max_m": float(agl.max()),
        }
    )


def build_cases(
    manifest: pd.DataFrame,
    *,
    era5_zarr: Path,
    output_dir: Path,
    terrain_mode: str,
    tile_dir: Path | None,
    terrain_static_nc: Path | None,
    fallback_terrain_static_nc: Path | None,
    download_missing_copdem: bool,
    z0_eff: float,
    target_hour_utc: int,
    overwrite: bool,
) -> pd.DataFrame:
    era5 = zarr.open_group(str(era5_zarr), mode="r")
    results: list[CaseBuildResult] = []
    terrain_cache: dict[tuple[str, float, float], tuple[np.ndarray, str]] = {}
    for i, row in manifest.iterrows():
        case_id = row_case_id(row)
        station_id = str(row["station_id"])
        timestamp_utc = row_timestamp(row, target_hour_utc)
        lat = float(row["lat"])
        lon = float(row["lon"])
        alt_m = float(row["alt_m"]) if "alt_m" in row and pd.notna(row["alt_m"]) else None
        output = output_dir / case_id / "grid.zarr"
        try:
            terrain_key = (station_id, round(lat, 6), round(lon, 6))
            if terrain_key not in terrain_cache:
                terrain_cache[terrain_key] = build_terrain(
                    lat,
                    lon,
                    terrain_mode=terrain_mode,
                    tile_dir=tile_dir,
                    static_nc=terrain_static_nc,
                    fallback_static_nc=fallback_terrain_static_nc,
                    download_missing_copdem=download_missing_copdem,
                )
            terrain, terrain_source = terrain_cache[terrain_key]
            era5_payload, era5_time = sample_era5_at_station(era5, timestamp_utc, lat, lon)
            write_grid_zarr(
                output,
                case_id=case_id,
                station_id=station_id,
                timestamp_utc=timestamp_utc,
                lat=lat,
                lon=lon,
                alt_m=alt_m,
                z0_eff=z0_eff,
                terrain=terrain,
                terrain_source=terrain_source,
                era5_payload=era5_payload,
                era5_time=era5_time,
                overwrite=overwrite,
            )
            status = "ok"
            error = ""
            logger.info("OK %04d/%04d %s -> %s", i + 1, len(manifest), case_id, output)
        except Exception as exc:
            terrain_source = ""
            terrain = np.full((NI, NJ), np.nan, dtype=np.float32)
            era5_time = ""
            status = "error"
            error = str(exc)
            logger.exception("FAILED %s", case_id)
        results.append(
            CaseBuildResult(
                case_id=case_id,
                station_id=station_id,
                timestamp_utc=timestamp_utc,
                lat=lat,
                lon=lon,
                grid_zarr=str(output),
                terrain_source=terrain_source,
                terrain_min_m=float(np.nanmin(terrain)),
                terrain_max_m=float(np.nanmax(terrain)),
                era5_time=era5_time,
                status=status,
                error=error,
            )
        )
    return pd.DataFrame([r.__dict__ for r in results])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--era5-zarr", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path)
    ap.add_argument("--terrain-mode", choices=["copdem", "coarse-netcdf"], default="copdem")
    ap.add_argument("--tile-dir", type=Path)
    ap.add_argument("--terrain-static-nc", type=Path)
    ap.add_argument("--fallback-terrain-static-nc", type=Path)
    ap.add_argument("--download-missing-copdem", action="store_true")
    ap.add_argument("--z0-eff", type=float, default=0.05)
    ap.add_argument("--target-hour-utc", type=int, default=12)
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    manifest = load_frame(args.manifest)
    required = {"station_id", "lat", "lon"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{args.manifest} missing required columns: {sorted(missing)}")
    if args.max_cases is not None:
        manifest = manifest.head(args.max_cases).copy()

    out_manifest = args.output_manifest or args.output_dir / "station_v2_grid_manifest.csv"
    results = build_cases(
        manifest,
        era5_zarr=args.era5_zarr,
        output_dir=args.output_dir,
        terrain_mode=args.terrain_mode,
        tile_dir=args.tile_dir,
        terrain_static_nc=args.terrain_static_nc,
        fallback_terrain_static_nc=args.fallback_terrain_static_nc,
        download_missing_copdem=args.download_missing_copdem,
        z0_eff=args.z0_eff,
        target_hour_utc=args.target_hour_utc,
        overwrite=args.overwrite,
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_manifest, index=False)
    ok = int((results["status"] == "ok").sum())
    logger.info("Wrote %s (%d/%d ok)", out_manifest, ok, len(results))
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
