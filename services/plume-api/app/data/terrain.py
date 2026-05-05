"""Fetch + crop + resample COP-DEM to a 128×128 grid around a (lat, lon) center.

COP-DEM GLO-30 tiles are public on AWS Open Data (no auth required).
Each tile covers 1° × 1° at ~30 m resolution.

For Plume: domain is 4 km × 4 km, which is well under 1° at European latitudes,
so at most 4 tiles are ever needed. In practice, 1-2 tiles cover the domain.

Caching: raw tiles are cached to disk forever (keyed by tile name); cropped
grids are cached keyed by (lat, lon) rounded to 0.001°.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np

from ..config import settings

COPDEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    """Return COP-DEM GLO-30 tile name for a SW corner at (lat, lon) integer degrees."""
    lat_hemi = "N" if lat >= 0 else "S"
    lon_hemi = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{lat_hemi}{abs(lat):02d}_00_{lon_hemi}{abs(lon):03d}_00_DEM"


def tile_url(lat: int, lon: int) -> str:
    name = tile_name(lat, lon)
    return f"{COPDEM_BASE}/{name}/{name}.tif"


def download_tile(lat: int, lon: int, cache_dir: Path) -> Path:
    """Download a tile if not cached; return local path. Raises on missing tiles (ocean)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = tile_name(lat, lon)
    dest = cache_dir / f"{name}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = tile_url(lat, lon)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"COP-DEM tile download failed ({url}): {e}") from e
    return dest


def tiles_for_domain(lat_center: float, lon_center: float, domain_km: float) -> list[tuple[int, int]]:
    """Integer (lat, lon) SW corners of all COP-DEM tiles intersecting the domain."""
    # Small-angle: convert half domain to degrees
    dlat = (domain_km * 0.5) / 111.0
    dlon = (domain_km * 0.5) / (111.0 * math.cos(math.radians(lat_center)))
    south = lat_center - dlat
    north = lat_center + dlat
    west = lon_center - dlon
    east = lon_center + dlon
    return [
        (lat, lon)
        for lat in range(math.floor(south), math.floor(north) + 1)
        for lon in range(math.floor(west), math.floor(east) + 1)
    ]


def fetch_terrain(
    lat_center: float,
    lon_center: float,
    domain_km: float | None = None,
    ny: int | None = None,
    nx: int | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return (terrain[ny,nx] in meters, (south, west, north, east)).

    Uses rasterio to merge tiles and resample to the target grid with bilinear.
    Caches the result to disk keyed by rounded (lat, lon).
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    domain_km = domain_km or settings.domain_km
    ny = ny or settings.grid_ny
    nx = nx or settings.grid_nx

    # Disk cache for the cropped result
    tcache_dir = settings.cache_dir / "terrain"
    tcache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{round(lat_center, 3)}_{round(lon_center, 3)}_{domain_km:.1f}_{ny}x{nx}.npz"
    cache_path = tcache_dir / key
    if cache_path.exists():
        data = np.load(cache_path)
        return data["terrain"], tuple(data["bounds"])

    # Compute bounds in degrees
    dlat = (domain_km * 0.5) / 111.0
    dlon = (domain_km * 0.5) / (111.0 * math.cos(math.radians(lat_center)))
    south = lat_center - dlat
    north = lat_center + dlat
    west = lon_center - dlon
    east = lon_center + dlon

    # Download + merge source tiles
    tiles_dir = settings.cache_dir / "copdem_tiles"
    tile_paths: list[Path] = []
    for lat, lon in tiles_for_domain(lat_center, lon_center, domain_km):
        try:
            tile_paths.append(download_tile(lat, lon, tiles_dir))
        except RuntimeError as e:
            print(f"[terrain] skip tile {lat},{lon}: {e}")
    if not tile_paths:
        raise RuntimeError(f"no COP-DEM tiles available for ({lat_center}, {lon_center})")

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        merged, merged_transform = merge(srcs, bounds=(west, south, east, north))
        # merged: (1, H, W)
        dst_transform = from_bounds(west, south, east, north, nx, ny)
        dst = np.zeros((ny, nx), dtype=np.float32)
        reproject(
            source=merged[0].astype(np.float32),
            destination=dst,
            src_transform=merged_transform,
            src_crs=srcs[0].crs,
            dst_transform=dst_transform,
            dst_crs=srcs[0].crs,
            resampling=Resampling.bilinear,
        )
    finally:
        for s in srcs:
            s.close()

    # rasterio has north at row 0; our convention matches (y=0 is north for the image,
    # and we flip in JS when painting the slice). Keep north-at-top here.
    bounds = (south, west, north, east)
    np.savez_compressed(cache_path, terrain=dst, bounds=np.array(bounds, dtype=np.float32))
    return dst, bounds
