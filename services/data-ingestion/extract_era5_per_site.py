"""
extract_era5_per_site.py — Extract 3×3 ERA5 grids per site from the Europe Zarr.

Reads era5_europe.zarr (single big request) and creates per-site Zarr stores
era5_site_NNNNN.zarr with the 3×3 grid centered on each site coordinates.

Usage:
    python extract_era5_per_site.py \
        --europe-zarr ../../data/raw/era5_europe.zarr \
        --sites ../../data/campaign/9k/sites.csv \
        --output-dir ../../data/raw/era5_campaign/
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def find_3x3_indices(lat: float, lon: float,
                     lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Find indices of the 3×3 ERA5 grid centered on (lat, lon).

    Returns (i_slice, j_slice) for the lat and lon dimensions.
    """
    i_center = int(np.argmin(np.abs(lats - lat)))
    j_center = int(np.argmin(np.abs(lons - lon)))

    # Take 3 points around center (handle edges)
    i_lo = max(0, i_center - 1)
    i_hi = min(len(lats), i_lo + 3)
    if i_hi - i_lo < 3:
        i_lo = max(0, i_hi - 3)

    j_lo = max(0, j_center - 1)
    j_hi = min(len(lons), j_lo + 3)
    if j_hi - j_lo < 3:
        j_lo = max(0, j_hi - 3)

    return slice(i_lo, i_hi), slice(j_lo, j_hi)


def extract_site(europe_data: dict, site_id: str, lat: float, lon: float,
                 output_dir: Path) -> bool:
    """Extract 3×3 grid for one site and write to Zarr."""
    import zarr

    out_path = output_dir / f"era5_{site_id}.zarr"
    if out_path.exists():
        return True

    i_slice, j_slice = find_3x3_indices(
        lat, lon, europe_data["lats"], europe_data["lons"])

    site_lats = europe_data["lats"][i_slice]
    site_lons = europe_data["lons"][j_slice]

    # Verify shape
    if len(site_lats) != 3 or len(site_lons) != 3:
        logger.warning("  %s: bad slice (%d, %d)", site_id, len(site_lats), len(site_lons))
        return False

    store = zarr.open_group(str(out_path), mode="w")

    coords = store.create_group("coords")
    coords.create_array("time", data=europe_data["times"])
    coords.create_array("lat", data=site_lats)
    coords.create_array("lon", data=site_lons)
    coords.create_array("level", data=europe_data["levels"])

    pressure = store.create_group("pressure")
    for var in ["u", "v", "z", "t", "q"]:
        pressure.create_array(
            var,
            data=europe_data[var][:, :, i_slice, j_slice].copy()
        )

    surface = store.create_group("surface")
    for var in ["u10", "v10", "t2m"]:
        surface.create_array(
            var,
            data=europe_data[var][:, i_slice, j_slice].copy()
        )

    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--europe-zarr", required=True, type=Path)
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sites", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Europe ERA5 in memory
    import zarr
    logger.info("Loading %s into memory...", args.europe_zarr)
    z = zarr.open_group(str(args.europe_zarr), mode="r")
    europe_data = {
        "times": np.array(z["coords/time"][:]),
        "lats": np.array(z["coords/lat"][:]),
        "lons": np.array(z["coords/lon"][:]),
        "levels": np.array(z["coords/level"][:]),
    }
    for var in ["u", "v", "z", "t", "q"]:
        europe_data[var] = np.array(z[f"pressure/{var}"][:])
    for var in ["u10", "v10", "t2m"]:
        europe_data[var] = np.array(z[f"surface/{var}"][:])
    logger.info("Loaded: %d times, %d levels, lats %.2f-%.2f, lons %.2f-%.2f",
                len(europe_data["times"]), len(europe_data["levels"]),
                europe_data["lats"][-1], europe_data["lats"][0],
                europe_data["lons"][0], europe_data["lons"][-1])

    # Read sites
    with open(args.sites) as f:
        sites = list(csv.DictReader(f))
    if args.max_sites:
        sites = sites[:args.max_sites]
    logger.info("Extracting %d sites", len(sites))

    t0 = time.time()
    n_ok = n_fail = 0
    for site in sites:
        site_id = site["site_id"]
        lat = float(site["lat"])
        lon = float(site["lon"])
        ok = extract_site(europe_data, site_id, lat, lon, args.output_dir)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Done: %d extracted, %d failed in %.1fs",
                n_ok, n_fail, time.time() - t0)


if __name__ == "__main__":
    main()
