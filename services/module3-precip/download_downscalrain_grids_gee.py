"""
Download gridded DownscalRain inputs from Google Earth Engine.

Outputs aligned NetCDF grids:
  - IMERG daily precipitation in mm/day
  - ERA5-Land daily precipitation in mm/day
  - static terrain variables at the same grid

Example:
    python download_downscalrain_grids_gee.py \
      --start 2021-12-30 --end 2022-12-31 \
      --bbox -10 35 30 60 \
      --output-dir ../../data/raw/downscalrain_grids/gee_2022
"""

from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import click
import ee
import numpy as np
import pandas as pd
import requests
import rasterio
import xarray as xr

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _region(bbox: tuple[float, float, float, float]) -> ee.Geometry:
    lon_min, lat_min, lon_max, lat_max = bbox
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False)


def _download_image_array(
    image: ee.Image,
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    timeout: int = 180,
    retries: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = {
        "region": _region(bbox),
        "dimensions": f"{nx}x{ny}",
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            url = image.getDownloadURL(params)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return _read_geotiff_response(resp.content)
        except Exception as exc:
            last_error = exc
            sleep_s = min(30, 2**attempt)
            log.warning("Download attempt %d/%d failed: %s; sleeping %ds", attempt, retries, exc, sleep_s)
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def _read_geotiff_response(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise ValueError("Earth Engine response zip contains no GeoTIFF")
            payload = zf.read(tif_names[0])
    else:
        payload = content

    with rasterio.open(io.BytesIO(payload)) as src:
        arr = src.read().astype(np.float32)
        transform = src.transform
        rows = np.arange(src.height)
        cols = np.arange(src.width)
        lon = transform.c + (cols + 0.5) * transform.a
        lat = transform.f + (rows + 0.5) * transform.e
    return arr, lat.astype(np.float64), lon.astype(np.float64)


def _imerg_daily_image(day: pd.Timestamp) -> ee.Image:
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
        .filterDate(start, end)
        .select("precipitation")
    )
    # IMERG is mm/hr at 30 min steps. Daily total = sum(rate * 0.5 h).
    return col.sum().multiply(0.5).rename("rain")


def _era5land_daily_image(day: pd.Timestamp) -> ee.Image:
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .first()
        .select("total_precipitation_sum")
    )
    # ERA5-Land total precipitation is in meters.
    return ee.Image(img).multiply(1000.0).rename("rain")


def _terrain_image() -> ee.Image:
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(srtm).rename("slope")
    aspect = ee.Terrain.aspect(srtm)
    aspect_rad = aspect.multiply(np.pi / 180.0)
    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")
    return srtm.rename("elevation").addBands([slope, aspect_sin, aspect_cos])


def _download_daily_stack(
    name: str,
    image_fn,
    dates: pd.DatetimeIndex,
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    output_path: Path,
    resume_dir: Path,
) -> None:
    resume_dir.mkdir(parents=True, exist_ok=True)
    arrays = []
    lat = None
    lon = None
    for i, day in enumerate(dates, start=1):
        cache_path = resume_dir / f"{name}_{day.strftime('%Y%m%d')}.npy"
        if cache_path.exists():
            arr = np.load(cache_path)
            if lat is None or lon is None:
                _, lat, lon = _download_image_array(image_fn(day), bbox, nx, ny)
            arrays.append(arr)
            continue
        log.info("%s %s (%d/%d)", name, day.date(), i, len(dates))
        arr_bands, lat, lon = _download_image_array(image_fn(day), bbox, nx, ny)
        arr = np.asarray(arr_bands[0], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(cache_path, arr)
        arrays.append(arr)

    if lat is None or lon is None:
        _, lat, lon = _download_image_array(image_fn(dates[0]), bbox, nx, ny)
    stack = np.stack(arrays, axis=0)
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), stack)},
        coords={"time": dates, "lat": lat, "lon": lon},
        attrs={"source": name, "bbox": list(bbox)},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    log.info("Wrote %s", output_path)


def _download_terrain(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    output_path: Path,
) -> None:
    log.info("Downloading terrain")
    arr, lat, lon = _download_image_array(_terrain_image(), bbox, nx, ny)
    data_vars: dict[str, Any] = {}
    for i, name in enumerate(["elevation", "slope", "aspect_sin", "aspect_cos"]):
        data_vars[name] = (("lat", "lon"), np.nan_to_num(arr[i], nan=0.0).astype(np.float32))
    ds = xr.Dataset(data_vars, coords={"lat": lat, "lon": lon}, attrs={"source": "SRTM via Earth Engine"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    log.info("Wrote %s", output_path)


@click.command()
@click.option("--start", required=True, help="Start date inclusive, YYYY-MM-DD")
@click.option("--end", required=True, help="End date inclusive, YYYY-MM-DD")
@click.option("--bbox", nargs=4, type=float, required=True, metavar="LON_MIN LAT_MIN LON_MAX LAT_MAX")
@click.option("--resolution-deg", default=0.1, show_default=True, type=float)
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--gee-project", default="ee-guillaumemaitrejean", show_default=True)
@click.option("--skip-imerg", is_flag=True, default=False)
@click.option("--skip-era5land", is_flag=True, default=False)
@click.option("--skip-terrain", is_flag=True, default=False)
def main(
    start: str,
    end: str,
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
    output_dir: str,
    gee_project: str,
    skip_imerg: bool,
    skip_era5land: bool,
    skip_terrain: bool,
) -> None:
    ee.Initialize(project=gee_project)
    out = Path(output_dir)
    lon_min, lat_min, lon_max, lat_max = bbox
    nx = int(round((lon_max - lon_min) / resolution_deg))
    ny = int(round((lat_max - lat_min) / resolution_deg))
    dates = pd.date_range(start, end, freq="D")
    log.info("Grid dimensions: %dx%d, dates=%d", nx, ny, len(dates))

    if not skip_imerg:
        _download_daily_stack(
            "imerg",
            _imerg_daily_image,
            dates,
            bbox,
            nx,
            ny,
            out / "imerg_daily.nc",
            out / "_daily_cache",
        )
    if not skip_era5land:
        _download_daily_stack(
            "era5land",
            _era5land_daily_image,
            dates,
            bbox,
            nx,
            ny,
            out / "era5land_daily.nc",
            out / "_daily_cache",
        )
    if not skip_terrain:
        _download_terrain(bbox, nx, ny, out / "terrain_static.nc")


if __name__ == "__main__":
    main()
