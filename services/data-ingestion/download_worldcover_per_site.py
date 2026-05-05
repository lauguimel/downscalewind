"""
download_worldcover_per_site.py — Pre-download WorldCover tiles per site.

For each site in sites.csv, download a small WorldCover GeoTIFF (10 m, ESA v200)
covering a buffered bounding box around the site. Stored as
data/raw/worldcover_per_site/<site_id>.tif so that on Aqua,
generate_z0_field.py can map z0 spatially during the campaign run.

Strategy: small per-site bbox (~6 km x 6 km) → ~700 KB/site → ~600 MB total
for 820 sites. Lighter than rsyncing a Europe-wide WorldCover (~10 GB).

Usage
-----
    python download_worldcover_per_site.py \\
        --sites data/campaign/complex_terrain_v1/sites.csv \\
        --out-dir data/raw/worldcover_per_site \\
        --buffer-km 4 --workers 4

Requires:
    - earthengine-api authenticated (`earthengine authenticate`)
    - EARTHENGINE_PROJECT env var (defaults to ee-guillaumemaitrejean)
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import requests

logger = logging.getLogger("dl_wc")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def _download_geotiff(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    tmp.rename(dest)


def download_site(site_id: str, lat: float, lon: float,
                  out_dir: Path, buffer_km: float, scale_m: int) -> tuple[str, str]:
    """Return (site_id, status). status: 'ok', 'skip', or error message."""
    import ee
    dest = out_dir / f"{site_id}.tif"
    if dest.exists() and dest.stat().st_size > 1024:
        return site_id, "skip"

    # Convert km buffer to degrees (lat/lon — small enough that flat-earth is OK)
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * max(0.1, abs(__import__("math").cos(__import__("math").radians(lat)))))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    region = ee.Geometry.Rectangle(list(bbox))

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(region)
    url = wc.getDownloadURL({
        "scale": scale_m,
        "crs": "EPSG:4326",
        "region": region,
        "format": "GEO_TIFF",
    })
    _download_geotiff(url, dest)
    return site_id, "ok"


@click.command()
@click.option("--sites", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--buffer-km", default=4.0, show_default=True,
              help="Half-width of bbox around each site (km)")
@click.option("--scale-m", default=10, show_default=True,
              help="Download resolution in metres (10 = native WorldCover)")
@click.option("--workers", default=4, show_default=True)
@click.option("--limit", default=0, show_default=True,
              help="Limit to first N sites (debug)")
@click.option("--ee-project", default=None,
              help="GEE project (defaults to $EARTHENGINE_PROJECT or ee-guillaumemaitrejean)")
def main(sites, out_dir, buffer_km, scale_m, workers, limit, ee_project):
    import ee
    project = ee_project or os.environ.get("EARTHENGINE_PROJECT", "ee-guillaumemaitrejean")
    ee.Initialize(project=project)
    logger.info("GEE initialised (project=%s)", project)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read sites
    rows = []
    with open(sites) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append((r["site_id"], float(r["lat"]), float(r["lon"])))
    if limit > 0:
        rows = rows[:limit]
    logger.info("Sites to process: %d", len(rows))

    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_site, sid, lat, lon, out_dir, buffer_km, scale_m): sid
                   for sid, lat, lon in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                _, status = fut.result()
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                if i % 25 == 0 or i == len(rows):
                    logger.info("[%4d/%d] ok=%d skip=%d err=%d", i, len(rows), n_ok, n_skip, n_err)
            except Exception as e:
                n_err += 1
                logger.warning("FAIL %s: %s", sid, str(e)[:120])

    logger.info("Done: ok=%d skip=%d err=%d (out=%s)", n_ok, n_skip, n_err, out_dir)
    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
