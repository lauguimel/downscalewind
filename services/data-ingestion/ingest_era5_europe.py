"""
ingest_era5_europe.py — Single CDS request for the entire Europe bounding box.

Instead of 450 individual CDS requests, download one large box covering all
sites, then extract 3×3 grids per site in post-processing.

Usage:
    python ingest_era5_europe.py \
        --sites ../../data/campaign/9k/sites.csv \
        --output ../../data/raw/era5_europe.zarr \
        --start 2017-05 --end 2017-06
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger
from shared.data_io import create_empty_store, open_store, VARIABLE_META

log = get_logger("ingest_era5_europe")

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
PRESSURE_VARIABLES = [
    "u_component_of_wind", "v_component_of_wind",
    "geopotential", "temperature", "specific_humidity",
]
SURFACE_VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature",
    "surface_pressure",
]
CDS_TO_SHORT = {
    "u_component_of_wind": "u", "v_component_of_wind": "v",
    "geopotential": "z", "temperature": "t", "specific_humidity": "q",
    "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
}
HOURS_6H = ["00:00", "06:00", "12:00", "18:00"]


def compute_bbox(sites_csv: Path, margin: float = 0.5) -> dict:
    """Compute Europe bounding box from sites + margin."""
    lats, lons = [], []
    with open(sites_csv) as f:
        for row in csv.DictReader(f):
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    return {
        "north": max(lats) + margin,
        "south": min(lats) - margin,
        "east": max(lons) + margin,
        "west": min(lons) - margin,
    }


def download_month(client, year: int, month: int, bbox: dict, target_path: Path):
    """Download one month of ERA5 for the bbox."""
    import calendar
    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    # Pressure levels
    log.info(f"  Downloading pressure levels {year}-{month:02d}...")
    pl_request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": PRESSURE_VARIABLES,
        "pressure_level": [str(p) for p in PRESSURE_LEVELS],
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": HOURS_6H,
        "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
    }
    pl_path = target_path.parent / f"_pl_{year}_{month:02d}.nc"
    client.retrieve("reanalysis-era5-pressure-levels", pl_request, str(pl_path))

    # Surface
    log.info(f"  Downloading surface {year}-{month:02d}...")
    sf_request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": SURFACE_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": HOURS_6H,
        "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
    }
    sf_path = target_path.parent / f"_sf_{year}_{month:02d}.nc"
    client.retrieve("reanalysis-era5-single-levels", sf_request, str(sf_path))

    return pl_path, sf_path


def nc_to_arrays(pl_path: Path, sf_path: Path) -> dict:
    """Load NetCDF and return as numpy arrays."""
    import xarray as xr

    pl_ds = xr.open_dataset(pl_path)
    sf_ds = xr.open_dataset(sf_path)

    if pl_ds.latitude[0] < pl_ds.latitude[-1]:
        pl_ds = pl_ds.isel(latitude=slice(None, None, -1))
        sf_ds = sf_ds.isel(latitude=slice(None, None, -1))

    level_var = "pressure_level" if "pressure_level" in pl_ds.dims else "level"
    if level_var != "level":
        pl_ds = pl_ds.rename({level_var: "level"})

    time_var = "valid_time" if "valid_time" in pl_ds.dims else "time"
    times = pl_ds[time_var].values

    return {
        "times": times.astype("datetime64[ns]"),
        "lats": pl_ds.latitude.values.astype(np.float32),
        "lons": pl_ds.longitude.values.astype(np.float32),
        "levels": pl_ds.level.values.astype(np.float32),
        "u": pl_ds.u.values.astype(np.float32),  # (time, level, lat, lon)
        "v": pl_ds.v.values.astype(np.float32),
        "z": pl_ds.z.values.astype(np.float32),
        "t": pl_ds.t.values.astype(np.float32),
        "q": pl_ds.q.values.astype(np.float32),
        "u10": sf_ds.u10.values.astype(np.float32),  # (time, lat, lon)
        "v10": sf_ds.v10.values.astype(np.float32),
        "t2m": sf_ds.t2m.values.astype(np.float32),
    }


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output Zarr store")
    parser.add_argument("--start", default="2017-05",
                        help="Start month YYYY-MM")
    parser.add_argument("--end", default="2017-06",
                        help="End month YYYY-MM")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Bbox margin in degrees (default: 0.5)")
    parser.add_argument("--keep-nc", action="store_true",
                        help="Keep intermediate NetCDF files")
    args = parser.parse_args()

    # Compute bbox
    bbox = compute_bbox(args.sites, args.margin)
    log.info("Europe bbox: N%.2f W%.2f S%.2f E%.2f",
             bbox["north"], bbox["west"], bbox["south"], bbox["east"])
    log.info("Grid size: ~%d × %d points",
             int((bbox["north"] - bbox["south"]) / 0.25),
             int((bbox["east"] - bbox["west"]) / 0.25))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # CDS client
    import cdsapi
    api_key = __import__("os").environ.get("CDS_API_KEY")
    api_url = __import__("os").environ.get("CDS_API_URL")
    if api_key:
        client = cdsapi.Client(url=api_url, key=api_key, quiet=True)
    else:
        client = cdsapi.Client(quiet=True)

    # Download all months
    y0, m0 = int(args.start[:4]), int(args.start[5:7])
    y1, m1 = int(args.end[:4]), int(args.end[5:7])
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1

    log.info("Downloading %d months: %s", len(months), months)
    nc_files = []
    for y, m in months:
        pl_path, sf_path = download_month(client, y, m, bbox, args.output)
        nc_files.append((pl_path, sf_path))

    # Concat all months
    log.info("Loading and concatenating NetCDF files...")
    all_data = [nc_to_arrays(pl, sf) for pl, sf in nc_files]

    merged = {
        "times": np.concatenate([d["times"] for d in all_data]),
        "lats": all_data[0]["lats"],
        "lons": all_data[0]["lons"],
        "levels": all_data[0]["levels"],
    }
    for var in ["u", "v", "z", "t", "q", "u10", "v10", "t2m"]:
        merged[var] = np.concatenate([d[var] for d in all_data], axis=0)

    log.info("Final shape: %d times × %d levels × %d × %d",
             len(merged["times"]), len(merged["levels"]),
             len(merged["lats"]), len(merged["lons"]))

    # Write to Zarr
    import zarr
    log.info("Writing Zarr to %s", args.output)
    store = zarr.open_group(str(args.output), mode="w")
    coords = store.create_group("coords")
    coords.create_array("time", data=merged["times"].astype(np.int64))
    coords.create_array("lat", data=merged["lats"])
    coords.create_array("lon", data=merged["lons"])
    coords.create_array("level", data=merged["levels"])

    pressure = store.create_group("pressure")
    for var in ["u", "v", "z", "t", "q"]:
        pressure.create_array(var, data=merged[var])

    surface = store.create_group("surface")
    for var in ["u10", "v10", "t2m"]:
        surface.create_array(var, data=merged[var])

    log.info("Done: %s", args.output)

    # Cleanup
    if not args.keep_nc:
        for pl, sf in nc_files:
            pl.unlink(missing_ok=True)
            sf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
