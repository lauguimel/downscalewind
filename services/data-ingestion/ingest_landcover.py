"""
ingest_landcover.py — Land cover → roughness z₀, displacement height d, LAD

Data sources (10 m resolution, free):
  1. ESA WorldCover 2021 — land cover classification (Sentinel-1 + Sentinel-2)
     https://esa-worldcover.org/en/data-access
  2. ETH Global Canopy Height 2020 — tree height from Sentinel-2 + GEDI LiDAR
     https://langnico.github.io/globalcanopyheight/

Output: multi-band GeoTIFF with 3 bands:
  Band 1: z₀ [m]  — aerodynamic roughness length
  Band 2: d  [m]  — displacement height
  Band 3: LAD [1/m] — leaf area density (column-average)

z₀ lookup table from Global Wind Atlas v4 (DTU):
  Tree cover / Shrubland: z₀ = 0.1 × h_canopy, d = 2/3 × h_canopy
  Other classes: fixed z₀ from GWA4 standard table

Usage:
    # Auto-download from Google Earth Engine:
    python ingest_landcover.py --site perdigao --download \\
        --output ../../data/raw/landcover_perdigao.tif

    # With pre-downloaded data:
    python ingest_landcover.py --site perdigao \\
        --worldcover /path/to/ESA_WorldCover_10m.tif \\
        --canopy-height /path/to/ETH_GlobalCanopyHeight_10m.tif \\
        --output ../../data/raw/landcover_perdigao.tif

    # Fallback (no data available):
    python ingest_landcover.py --site perdigao \\
        --output ../../data/raw/landcover_perdigao.tif

References:
    Zanaga et al. (2022) ESA WorldCover 10m 2021 v200
    Lang et al. (2023) A high-resolution canopy height model of the Earth. Nature E&E.
    Global Wind Atlas v4 (DTU) — roughness length conversion table
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import sha256_file

log = get_logger("ingest_landcover")

# ── ESA WorldCover 2021 class → z₀ table (GWA4 / DTU standard) ──────────────
# Classes 10 (tree cover) and 20 (shrubland) use canopy height: z₀ = 0.1 × h
# Other classes: fixed z₀ from GWA4

WORLDCOVER_Z0: dict[int, float] = {
    10:  None,     # Tree cover      — z₀ from canopy height
    20:  None,     # Shrubland        — z₀ from canopy height
    30:  0.03,     # Grassland
    40:  0.10,     # Cropland
    50:  0.50,     # Built-up
    60:  0.005,    # Bare / sparse vegetation
    70:  0.001,    # Snow and Ice
    80:  0.0002,   # Permanent water bodies
    90:  0.03,     # Herbaceous wetland
    95:  None,     # Mangroves         — z₀ from canopy height
    100: 0.01,     # Moss and lichen
}

# Default z₀ for unknown classes
Z0_DEFAULT = 0.05

# Typical LAI values by land cover (used to estimate LAD = LAI / h_canopy)
# Source: various remote sensing reviews for Mediterranean/Iberian vegetation
WORLDCOVER_LAI: dict[int, float] = {
    10:  2.5,      # Tree cover (eucalyptus plantation at Perdigão)
    20:  1.5,      # Shrubland
    30:  1.0,      # Grassland
    40:  2.0,      # Cropland
    95:  3.0,      # Mangroves
}


def compute_z0_d_lad(
    lc: np.ndarray,
    canopy_h: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert land cover + canopy height to z₀, d, and LAD arrays.

    Parameters
    ----------
    lc : 2-D array of ESA WorldCover class codes (uint8).
    canopy_h : 2-D array of canopy height [m] (float32), or None.

    Returns
    -------
    z0  : roughness length [m]
    d   : displacement height [m]
    lad : leaf area density [1/m] (column-average)
    """
    z0  = np.full(lc.shape, Z0_DEFAULT, dtype=np.float32)
    d   = np.zeros(lc.shape, dtype=np.float32)
    lad = np.zeros(lc.shape, dtype=np.float32)

    # Fixed z₀ classes
    for cls, z0_val in WORLDCOVER_Z0.items():
        if z0_val is not None:
            mask = lc == cls
            z0[mask] = z0_val

    # Canopy-height-dependent classes (10, 20, 95)
    if canopy_h is not None:
        h = np.maximum(canopy_h, 0.0)  # clip negatives
        for cls in (10, 20, 95):
            mask = lc == cls
            if not np.any(mask):
                continue
            h_cls = h[mask]
            # z₀ = 0.1 × h (ORA model, Wieringa 1992)
            z0[mask] = np.maximum(0.1 * h_cls, 0.01)
            # d = 2/3 × h (displacement height)
            d[mask] = (2.0 / 3.0) * h_cls
            # LAD = LAI / h  (column-average leaf area density)
            lai = WORLDCOVER_LAI.get(cls, 2.0)
            lad[mask] = np.where(h_cls > 0.5, lai / h_cls, 0.0)
    else:
        # No canopy height: use fixed z₀ for tree/shrub classes
        log.warning("No canopy height data — using fixed z₀ for tree/shrub classes")
        for cls in (10, 20, 95):
            mask = lc == cls
            z0[mask] = 1.0   # conservative estimate
            d[mask]  = 6.0   # ~2/3 × 9 m (typical Perdigão eucalyptus)
            lad[mask] = 0.3  # LAI~2.5 / 9m

    return z0, d, lad


# ── Google Earth Engine download ──────────────────────────────────────────────

def download_from_gee(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    site: str,
    scale: int = 10,
) -> tuple[Path, Path]:
    """Download WorldCover + Canopy Height from GEE for a bounding box.

    Parameters
    ----------
    bbox : (west, south, east, north) in degrees.
    output_dir : Directory for downloaded GeoTIFFs.
    site : Site name (used in filenames).
    scale : Download resolution in metres (default 10).

    Returns
    -------
    (worldcover_path, canopy_height_path) : Paths to downloaded GeoTIFFs.
    """
    import os
    import ee
    import requests

    project = os.environ.get("EARTHENGINE_PROJECT", "ee-guillaumemaitrejean")
    ee.Initialize(project=project)
    log.info("GEE initialized", extra={"project": ee.data.getAssetRoots()})

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    output_dir.mkdir(parents=True, exist_ok=True)
    wc_path = output_dir / f"worldcover_{site}.tif"
    ch_path = output_dir / f"canopy_height_{site}.tif"

    # Estimate download size and auto-coarsen if needed (GEE limit: ~50 MB)
    # GEE uses ~2 bytes/pixel (int16 or overhead), so budget 24M pixels max
    deg_span_x = east - west
    deg_span_y = north - south
    n_pixels = (deg_span_x * 111_000 / scale) * (deg_span_y * 111_000 / scale)
    max_pixels = 24_000_000  # ~48 MB at 2 bytes/pixel
    effective_scale = scale
    if n_pixels > max_pixels:
        effective_scale = int(scale * (n_pixels / max_pixels) ** 0.5) + 1
        log.info("Auto-coarsening download", extra={
            "original_scale": scale, "effective_scale": effective_scale,
            "reason": f"{n_pixels/1e6:.1f}M pixels > {max_pixels/1e6:.0f}M limit",
        })

    # ── ESA WorldCover 2021 ───────────────────────────────────────────────
    if not wc_path.exists():
        log.info("Downloading ESA WorldCover 2021 from GEE",
                 extra={"scale_m": effective_scale})
        wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(region)
        url = wc.getDownloadURL({
            "scale": effective_scale,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        })
        _download_geotiff(url, wc_path)
        log.info("WorldCover downloaded", extra={"path": str(wc_path)})
    else:
        log.info("WorldCover already exists, skipping", extra={"path": str(wc_path)})

    # ── ETH Global Canopy Height 2020 ─────────────────────────────────────
    if not ch_path.exists():
        log.info("Downloading ETH Canopy Height 2020 from GEE",
                 extra={"scale_m": effective_scale})
        ch = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").clip(region)
        url = ch.getDownloadURL({
            "scale": effective_scale,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        })
        _download_geotiff(url, ch_path)
        log.info("Canopy height downloaded", extra={"path": str(ch_path)})
    else:
        log.info("Canopy height already exists, skipping", extra={"path": str(ch_path)})

    return wc_path, ch_path


def _download_geotiff(url: str, output_path: Path) -> None:
    """Download a GeoTIFF from a GEE download URL (handles zip or raw tiff)."""
    import requests

    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "zip" in content_type or output_path.suffix != ".zip" and resp.content[:2] == b"PK":
        # GEE returns a zip file containing the GeoTIFF
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                raise ValueError(f"No .tif found in zip from {url}")
            with zf.open(tif_names[0]) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
    else:
        output_path.write_bytes(resp.content)

    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info("Downloaded", extra={"path": str(output_path), "size_mb": f"{size_mb:.1f}"})


def _clip_raster(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    band: int = 1,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read and clip a raster to a bounding box (west, south, east, north).

    Returns (data, transform, crs).
    """
    west, south, east, north = bbox
    with rasterio.open(str(src_path)) as src:
        from rasterio.windows import from_bounds as win_from_bounds
        window = win_from_bounds(west, south, east, north, src.transform)
        data = src.read(band, window=window)
        transform = src.window_transform(window)
        return data, transform, src.crs


@click.command()
@click.option("--site", required=True, help="Site identifier (e.g. perdigao)")
@click.option("--output", required=True, help="Output GeoTIFF path (3 bands: z0, d, LAD)")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Site config directory")
@click.option("--worldcover", default=None, type=click.Path(exists=True),
              help="ESA WorldCover 2021 GeoTIFF (10 m)")
@click.option("--canopy-height", default=None, type=click.Path(exists=True),
              help="ETH Global Canopy Height 2020 GeoTIFF (10 m)")
@click.option("--download", is_flag=True, default=False,
              help="Auto-download from Google Earth Engine (requires earthengine-api)")
@click.option("--resolution-m", default=10, show_default=True,
              help="Output resolution in metres")
def main(site, output, config_dir, worldcover, canopy_height, download, resolution_m):
    """Generate z₀/d/LAD raster from ESA WorldCover + ETH Canopy Height."""
    log.info("Starting land cover ingestion", extra={"site": site})

    # Load site config
    config_path = Path(config_dir) / f"{site}.yaml"
    if not config_path.exists():
        log.error("Site config not found", extra={"path": str(config_path)})
        sys.exit(1)

    with open(config_path) as f:
        site_cfg = yaml.safe_load(f)

    domain = site_cfg["era5_domain"]
    margin = 0.05  # ~5 km margin
    bbox = (
        domain["west"]  - margin,
        domain["south"] - margin,
        domain["east"]  + margin,
        domain["north"] + margin,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Auto-download from GEE if requested ──────────────────────────────────
    if download and not worldcover and not canopy_height:
        raw_dir = output_path.parent
        wc_path, ch_path = download_from_gee(bbox, raw_dir, site, scale=resolution_m)
        worldcover = str(wc_path)
        canopy_height = str(ch_path)

    lc_data = None
    ch_data = None
    transform = None
    source_label = "uniform_fallback"

    # ── Read WorldCover ──────────────────────────────────────────────────────
    if worldcover:
        log.info("Reading WorldCover", extra={"path": worldcover})
        lc_data, transform, _ = _clip_raster(Path(worldcover), bbox)
        source_label = "ESA_WorldCover_2021"
    else:
        log.warning(
            "No WorldCover data. Use --download or provide --worldcover path"
        )

    # ── Read ETH Canopy Height ───────────────────────────────────────────────
    if canopy_height:
        log.info("Reading ETH canopy height", extra={"path": canopy_height})
        ch_data, ch_transform, _ = _clip_raster(Path(canopy_height), bbox)
        ch_data = ch_data.astype(np.float32)
        ch_data[ch_data > 100] = 0  # mask nodata (255 = no data in ETH product)
        ch_data[ch_data < 0] = 0

        # Resample canopy height to match WorldCover grid if needed
        if lc_data is not None and ch_data.shape != lc_data.shape:
            log.info("Resampling canopy height to WorldCover grid")
            ch_resampled = np.zeros(lc_data.shape, dtype=np.float32)
            reproject(
                ch_data, ch_resampled,
                src_transform=ch_transform, dst_transform=transform,
                src_crs="EPSG:4326", dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            ch_data = ch_resampled
        source_label += "+ETH_CanopyHeight_2020"
    else:
        log.warning(
            "No canopy height data. Use --download or provide --canopy-height path"
        )

    # ── Compute z₀, d, LAD ──────────────────────────────────────────────────
    if lc_data is not None:
        z0, d, lad = compute_z0_d_lad(lc_data, ch_data)
    else:
        # Uniform fallback
        resolution_deg = resolution_m / 111320.0
        nrows = max(1, int((bbox[3] - bbox[1]) / resolution_deg))
        ncols = max(1, int((bbox[2] - bbox[0]) / resolution_deg))
        z0  = np.full((nrows, ncols), Z0_DEFAULT, dtype=np.float32)
        d   = np.zeros((nrows, ncols), dtype=np.float32)
        lad = np.zeros((nrows, ncols), dtype=np.float32)
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], ncols, nrows)
        source_label = "uniform_fallback_0.05m"

    # ── Write multi-band GeoTIFF ─────────────────────────────────────────────
    profile = {
        "driver": "GTiff",
        "height": z0.shape[0],
        "width":  z0.shape[1],
        "count":  3,
        "dtype":  "float32",
        "crs":    "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "nodata": -9999.0,
    }

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(z0,  1)
        dst.write(d,   2)
        dst.write(lad, 3)
        dst.update_tags(
            source=source_label,
            site=site,
            band1="z0_m",
            band2="displacement_height_m",
            band3="LAD_1_per_m",
        )
        dst.set_band_description(1, "roughness_length_z0_m")
        dst.set_band_description(2, "displacement_height_d_m")
        dst.set_band_description(3, "leaf_area_density_LAD_1_per_m")

    sha_out = sha256_file(output_path)
    log.info("Land cover raster exported", extra={
        "output": str(output_path),
        "source": source_label,
        "shape": f"{z0.shape[0]}x{z0.shape[1]}",
        "z0_range": f"[{float(np.nanmin(z0)):.4f}, {float(np.nanmax(z0)):.3f}]",
        "d_range": f"[{float(np.nanmin(d)):.1f}, {float(np.nanmax(d)):.1f}]",
        "sha256": sha_out[:16] + "...",
    })


if __name__ == "__main__":
    main()
