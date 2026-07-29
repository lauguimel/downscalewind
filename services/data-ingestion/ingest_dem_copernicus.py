"""
ingest_dem_copernicus.py — Download Copernicus DSM 30 m tiles from AWS S3.

Phase H mission M_H+ Axe 1: extend DEM coverage to PT + south ES so OBS stations
in that zone become usable for `infer_at_stations.py`.

Tile naming convention (matches `inference_input._resolve_dem_path`):
    Copernicus_DSM_COG_10_N<NN>_00_<E|W><EEE>_00_DEM.tif
where N<NN> = floor(lat), <E|W><EEE> = E for lon>=0 / W otherwise with
abs(floor(lon)) zero-padded to 3 digits.

Endpoint (AWS S3 anonymous, public): the file lives at:
    https://copernicus-dem-30m.s3.amazonaws.com/
        Copernicus_DSM_COG_10_<TILE>_DEM/Copernicus_DSM_COG_10_<TILE>_DEM.tif
where <TILE> = N<NN>_00_<E|W><EEE>_00 (note: no trailing _DEM in the prefix).

Usage
-----
Download PT (bbox 36..42 N, -10..-6 E) + south ES extension into the existing
shared srtm_tiles dir:

    python services/data-ingestion/ingest_dem_copernicus.py \\
        --bbox 36,-10,42,-6 \\
        --output-dir /home/maitreje/dsw/data/raw/srtm_tiles/ \\
        --skip-existing

`--bbox` is `S,W,N,E` (degrees). Tile range is inclusive: for `S,W,N,E =
36,-10,42,-6`, downloads lat in [36..41] and lon in [-10..-7] (lat<N, lon<E)
- 6×4 = 24 tiles candidates.

Smoke (no download, just print tile list):
    python services/data-ingestion/ingest_dem_copernicus.py --bbox 38,-9,39,-8 --dry-run
"""
from __future__ import annotations

import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

logger = logging.getLogger("ingest_dem_copernicus")

# AWS S3 public mirror of Copernicus DSM 30 m (GLO-30, COG 10 m → resampled 30 m).
S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


# ─── tile naming helpers ───────────────────────────────────────────────────

def tile_name(lat_floor: int, lon_floor: int) -> str:
    """Return `N<NN>_00_<E|W><EEE>_00` (no Copernicus_DSM_COG_10_ prefix)."""
    lat_dir = "N" if lat_floor >= 0 else "S"
    lat_idx = lat_floor if lat_floor >= 0 else -lat_floor
    lon_dir = "E" if lon_floor >= 0 else "W"
    lon_idx = lon_floor if lon_floor >= 0 else -lon_floor
    return f"{lat_dir}{lat_idx:02d}_00_{lon_dir}{lon_idx:03d}_00"


def tile_filename(lat_floor: int, lon_floor: int) -> str:
    """Filename matching `inference_input._resolve_dem_path` expectations."""
    return f"Copernicus_DSM_COG_10_{tile_name(lat_floor, lon_floor)}_DEM.tif"


def tile_url(lat_floor: int, lon_floor: int) -> str:
    base = tile_name(lat_floor, lon_floor)
    return f"{S3_BASE}/Copernicus_DSM_COG_10_{base}_DEM/Copernicus_DSM_COG_10_{base}_DEM.tif"


def enumerate_tiles(south: float, west: float, north: float, east: float) -> list[tuple[int, int]]:
    """Enumerate 1°×1° tile floors covering [south, north) × [west, east).

    Convention: tile N38_W008 covers lat in [38, 39) and lon in [-8, -7).
    """
    import math
    lat_lo = int(math.floor(south))
    lat_hi = int(math.floor(north - 1e-9))   # exclusive upper bound treated correctly
    lon_lo = int(math.floor(west))
    lon_hi = int(math.floor(east - 1e-9))
    tiles: list[tuple[int, int]] = []
    for lat in range(lat_lo, lat_hi + 1):
        for lon in range(lon_lo, lon_hi + 1):
            tiles.append((lat, lon))
    return tiles


# ─── download helpers ───────────────────────────────────────────────────────

def download_tile(
    lat_floor: int, lon_floor: int, out_dir: Path,
    *, retries: int = 3, timeout: float = 60.0,
) -> tuple[Path, str]:
    """Download one tile. Returns (out_path, status) where status in
    {downloaded, skipped, not_found, error}.
    """
    fname = tile_filename(lat_floor, lon_floor)
    url = tile_url(lat_floor, lon_floor)
    out_path = out_dir / fname

    # Min size = 50 KB: filter only truly-empty placeholder files. Coastal
    # tiles with sparse land (e.g. small island / partial coast) can be
    # legitimately ~90 KB; any DSM tile with non-trivial content is ≥ 50 KB.
    MIN_BYTES = 50 * 1024
    if out_path.exists() and out_path.stat().st_size >= MIN_BYTES:
        return out_path, "skipped"

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "downscalewind/ingest_dem_copernicus"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(out_path, "wb") as f:
                # Stream by chunks to avoid memory spikes (~40 MB per tile).
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            n_bytes = out_path.stat().st_size
            if n_bytes < MIN_BYTES:
                logger.warning("  %s: suspicious size %d B — removing", fname, n_bytes)
                out_path.unlink(missing_ok=True)
                return out_path, "error"
            return out_path, "downloaded"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Tile does not exist (ocean-only or outside coverage).
                return out_path, "not_found"
            last_exc = e
            logger.warning("  %s: HTTP %d (attempt %d/%d)", fname, e.code, attempt + 1, retries)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            logger.warning("  %s: %s (attempt %d/%d)", fname, type(e).__name__, attempt + 1, retries)
        time.sleep(2.0 * (attempt + 1))

    if last_exc is not None:
        logger.error("  %s: failed after %d retries (%s)", fname, retries, last_exc)
    return out_path, "error"


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command(context_settings={"show_default": True})
@click.option("--bbox", required=True,
              help="Bounding box 'S,W,N,E' in degrees, e.g. '36,-10,42,-6' for PT.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory to write tiles into (existing shared dir works).")
@click.option("--skip-existing", is_flag=True, default=True,
              help="Skip tiles already present (default).")
@click.option("--dry-run", is_flag=True, default=False,
              help="List target tiles without downloading.")
@click.option("--retries", type=int, default=3)
@click.option("--timeout", type=float, default=120.0,
              help="HTTP request timeout in seconds.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(bbox, output_dir, skip_existing, dry_run, retries, timeout, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise click.BadParameter(f"bbox must be 'S,W,N,E' (4 values), got {bbox!r}")
    south, west, north, east = (float(p) for p in parts)
    if south >= north or west >= east:
        raise click.BadParameter("bbox: require S<N and W<E")

    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    tiles = enumerate_tiles(south, west, north, east)
    logger.info("bbox = (S=%.2f, W=%.2f, N=%.2f, E=%.2f) → %d candidate tiles",
                south, west, north, east, len(tiles))

    counts = {"downloaded": 0, "skipped": 0, "not_found": 0, "error": 0}
    for lat_f, lon_f in tiles:
        fname = tile_filename(lat_f, lon_f)
        if dry_run:
            existing = (output_dir / fname).exists() if not dry_run else False
            url = tile_url(lat_f, lon_f)
            logger.info("  [dry-run] %s %s -> %s", "EXISTS" if existing else "MISSING", fname, url)
            continue
        existing_ok = skip_existing and (output_dir / fname).exists() and (output_dir / fname).stat().st_size >= 1024 * 1024
        if existing_ok:
            counts["skipped"] += 1
            logger.info("  [skip] %s (already %d bytes)", fname, (output_dir / fname).stat().st_size)
            continue
        t0 = time.time()
        out, status = download_tile(lat_f, lon_f, output_dir, retries=retries, timeout=timeout)
        counts[status] += 1
        if status == "downloaded":
            size_mb = out.stat().st_size / 1e6
            logger.info("  [ok ] %s (%.1f MB, %.1f s)", fname, size_mb, time.time() - t0)
        elif status == "not_found":
            logger.info("  [404] %s (ocean / outside DSM coverage)", fname)
        elif status == "skipped":
            counts["skipped"] += 1
        else:
            logger.error("  [err] %s", fname)

    logger.info(
        "DONE | downloaded=%d skipped=%d not_found=%d error=%d / total=%d",
        counts["downloaded"], counts["skipped"], counts["not_found"], counts["error"], len(tiles),
    )
    if counts["error"] > 0 and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    cli()
