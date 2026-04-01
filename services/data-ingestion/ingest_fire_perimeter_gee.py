#!/usr/bin/env python3
"""Download MODIS burned area / active fire data for Pedrógão Grande (June 2017) from GEE.

Tries datasets in order:
  1. MODIS/061/MCD64A1  (Burned Area Monthly, 500m)
  2. MODIS/061/MOD14A1  (Active Fire Daily, 1km)
  3. ESA/CCI/FireCCI/5_1 (ESA CCI Fire, 250m)

Output: data/raw/pedrogao_burned_area_2017.tif
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import ee
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT = "ee-guillaumemaitrejean"
CENTER = (39.93, -8.23)  # Pedrógão Grande
BBOX = [-8.7, 39.5, -7.7, 40.3]  # [west, south, east, north]
DATE_START = "2017-06-01"
DATE_END = "2017-08-01"
SCALE = 500  # metres
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def init_gee():
    ee.Initialize(project=PROJECT)
    print(f"GEE initialized (project={PROJECT})")


def make_region():
    """Return ee.Geometry for the bounding box."""
    west, south, east, north = BBOX
    return ee.Geometry.Rectangle([west, south, east, north])


def download_geotiff(image, region, out_path, scale=SCALE):
    """Download a single-band image as GeoTIFF via getDownloadURL."""
    url = image.getDownloadURL({
        "region": region,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })
    print(f"  Downloading from: {url[:120]}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(out_path))
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def try_mcd64a1(region):
    """MODIS Burned Area Monthly (MCD64A1) — 500m, band 'BurnDate'."""
    print("\n[1] Trying MODIS/061/MCD64A1 (Burned Area Monthly)...")
    col = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(region)
    )
    count = col.size().getInfo()
    print(f"  Images found: {count}")
    if count == 0:
        return None, None

    # BurnDate: day-of-year when pixel burned (0 = unburned)
    burn = col.select("BurnDate").max()

    # Burned mask: any pixel with BurnDate > 0
    burned_mask = burn.gt(0).selfMask()

    # Compute burned area (km²)
    pixel_area = ee.Image.pixelArea()  # m²
    burned_area_m2 = burned_mask.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=SCALE,
        maxPixels=1e9,
    )
    area_km2 = ee.Number(burned_area_m2.get("BurnDate")).divide(1e6).getInfo()

    return burn, area_km2


def try_mod14a1(region):
    """MODIS Active Fire Daily (MOD14A1) — 1km, band 'FireMask'."""
    print("\n[2] Trying MODIS/061/MOD14A1 (Active Fire Daily)...")
    col = (
        ee.ImageCollection("MODIS/061/MOD14A1")
        .filterDate("2017-06-15", "2017-07-15")
        .filterBounds(region)
    )
    count = col.size().getInfo()
    print(f"  Images found: {count}")
    if count == 0:
        return None, None

    # FireMask >= 7 means fire detected (7=low, 8=nominal, 9=high confidence)
    fire = col.select("FireMask").max()
    fire_mask = fire.gte(7).selfMask()

    pixel_area = ee.Image.pixelArea()
    area_m2 = fire_mask.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=1000,
        maxPixels=1e9,
    )
    area_km2 = ee.Number(area_m2.get("FireMask")).divide(1e6).getInfo()

    return fire, area_km2


def try_fireCCI(region):
    """ESA CCI Fire (FireCCI51) — 250m."""
    print("\n[3] Trying ESA/CCI/FireCCI/5_1 (ESA CCI Fire)...")
    col = (
        ee.ImageCollection("ESA/CCI/FireCCI/5_1")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(region)
    )
    count = col.size().getInfo()
    print(f"  Images found: {count}")
    if count == 0:
        return None, None

    burn = col.select("BurnDate").max()
    burned_mask = burn.gt(0).selfMask()

    pixel_area = ee.Image.pixelArea()
    area_m2 = burned_mask.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=250,
        maxPixels=1e9,
    )
    area_km2 = ee.Number(area_m2.get("BurnDate")).divide(1e6).getInfo()

    return burn, area_km2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "pedrogao_burned_area_2017.tif")
    args = parser.parse_args()

    init_gee()
    region = make_region()

    # Try datasets in order of preference
    datasets = [
        ("MCD64A1", try_mcd64a1),
        ("MOD14A1", try_mod14a1),
        ("FireCCI51", try_fireCCI),
    ]

    for name, fn in datasets:
        try:
            image, area_km2 = fn(region)
            if image is not None:
                print(f"\n  ✓ {name}: burned area ≈ {area_km2:.1f} km²")
                download_geotiff(image, region, args.output)
                print(f"\n── Summary ──")
                print(f"  Dataset:     {name}")
                print(f"  Burned area: {area_km2:.1f} km²")
                print(f"  Bbox:        {BBOX}")
                print(f"  Output:      {args.output}")
                return
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            continue

    print("All datasets failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
