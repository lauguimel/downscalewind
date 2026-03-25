"""
download_srtm_region.py — Download and merge Copernicus GLO-30 DEM tiles.

Downloads 1°×1° tiles from AWS Open Data, merges them into a single GeoTIFF.

Usage:
    python download_srtm_region.py \
        --bbox -10 36 10 55 \
        --output /path/to/srtm_europe.tif

Source: Copernicus DEM GLO-30 (30m, EPSG:4326)
    https://copernicus-dem-30m.s3.amazonaws.com/
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

AWS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    """Generate Copernicus DEM tile name for a 1°×1° cell.

    Convention: Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM
    The tile covers [lat, lat+1) × [lon, lon+1).
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tile_url(lat: int, lon: int) -> str:
    """Full URL for a Copernicus GLO-30 tile on AWS."""
    name = tile_name(lat, lon)
    return f"{AWS_BASE}/{name}/{name}.tif"


def download_tile(url: str, dest: Path) -> bool:
    """Download a single tile. Returns False if 404 (ocean tile)."""
    import requests

    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 404:
            return False  # ocean or missing tile
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def merge_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """Merge multiple GeoTIFF tiles into a single file using rasterio."""
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)

    profile = datasets[0].profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "compress": "deflate",
        "bigtiff": "yes",  # needed for > 4 GB
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    logger.info("Merged %d tiles → %s (%d×%d px)",
                len(tile_paths), output_path, mosaic.shape[2], mosaic.shape[1])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download and merge Copernicus GLO-30 DEM")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, default=None,
                        help="Directory to cache individual tiles (default: temp)")
    args = parser.parse_args()

    lon_min, lat_min, lon_max, lat_max = args.bbox

    # Generate tile grid
    lats = range(int(np.floor(lat_min)), int(np.ceil(lat_max)))
    lons = range(int(np.floor(lon_min)), int(np.ceil(lon_max)))
    n_tiles = len(lats) * len(lons)
    logger.info("Region: [%.0f,%.0f] → [%.0f,%.0f], %d tiles",
                lon_min, lat_min, lon_max, lat_max, n_tiles)

    # Download
    tile_dir = args.tile_dir or Path(tempfile.mkdtemp(prefix="srtm_"))
    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = []
    n_downloaded = 0
    n_cached = 0
    n_missing = 0

    for lat in lats:
        for lon in lons:
            name = tile_name(lat, lon)
            dest = tile_dir / f"{name}.tif"

            if dest.exists():
                tile_paths.append(dest)
                n_cached += 1
                continue

            url = tile_url(lat, lon)
            if download_tile(url, dest):
                tile_paths.append(dest)
                n_downloaded += 1
                if (n_downloaded + n_cached) % 20 == 0:
                    logger.info("  %d / %d tiles (downloaded=%d, cached=%d, missing=%d)",
                                n_downloaded + n_cached + n_missing, n_tiles,
                                n_downloaded, n_cached, n_missing)
            else:
                n_missing += 1  # ocean

    logger.info("Tiles: %d downloaded, %d cached, %d missing (ocean)",
                n_downloaded, n_cached, n_missing)

    if not tile_paths:
        logger.error("No tiles downloaded!")
        return

    # Merge
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merge_tiles(tile_paths, args.output)

    size_mb = args.output.stat().st_size / 1e6
    logger.info("Output: %s (%.0f MB)", args.output, size_mb)


if __name__ == "__main__":
    main()
