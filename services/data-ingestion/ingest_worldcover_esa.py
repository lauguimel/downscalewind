"""
ingest_worldcover_esa.py — Download ESA WorldCover 2021 v200 tiles from AWS S3.

Phase H' mission M_H+ Axe 4: build a global ESA WC tile library so OBS stations
(NOAA, ICOS, SYNOP, AEMET, IPMA) in FR/ES/PT can resolve z0_eff at inference time
instead of falling back to the uniform z0_eff=0.05 m placeholder which is
out-of-distribution for the surrogate v2 (trained on heterogeneous z0).

Tile naming convention (ESA WC v200):
    ESA_WorldCover_10m_2021_v200_<LAT><LON>_Map.tif
where
    <LAT> = N<NN>|S<NN>   (2-digit, lower-left corner)
    <LON> = E<EEE>|W<EEE> (3-digit, lower-left corner)
Tiles span 3°×3°. Lower-left corner snapped to multiples of 3 (lat ∈
{..., 33, 36, 39, 42, 45, 48, 51, ...}, lon ∈ {..., -12, -9, -6, -3, 0, 3, 6, ...}).

Endpoint (AWS S3 anonymous, public registry):
    https://esa-worldcover.s3.amazonaws.com/v200/2021/map/<filename>

Usage
-----
Download FR / ES / PT continental tiles (bbox S=35, W=-10, N=52, E=8):

    python services/data-ingestion/ingest_worldcover_esa.py \\
        --bbox 35,-10,52,8 \\
        --output-dir /home/maitreje/dsw/data/raw/worldcover_esa/ \\
        --skip-existing

`--bbox` is `S,W,N,E` (degrees). Tile enumeration snaps the bbox to the 3°
grid lower-left corners that intersect the bbox interior.

Dry-run (no download, just print tile list):
    python services/data-ingestion/ingest_worldcover_esa.py \\
        --bbox 35,-10,52,8 --output-dir /tmp --dry-run
"""
from __future__ import annotations

import logging
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

logger = logging.getLogger("ingest_worldcover_esa")

# AWS S3 public mirror of ESA WorldCover 2021 v200 (registry of open data).
S3_BASE = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"

# Tiles span 3°×3°.
TILE_STEP_DEG = 3


# ─── tile naming helpers ───────────────────────────────────────────────────

def _snap_to_step(value: float, step: int, *, mode: str) -> int:
    """Snap `value` (float deg) to the nearest multiple of `step` ≤ value
    (mode='floor') or ≥ value (mode='ceil_excl'). For the bbox enumeration
    we want all tiles whose lower-left corner is in [S, N) × [W, E).
    """
    if mode == "floor":
        return int(math.floor(value / step) * step)
    if mode == "ceil_excl":
        # Smallest multiple of `step` strictly greater than `value`.
        return int(math.floor(value / step) * step) + (
            step if (value % step == 0) else step
        )
    raise ValueError(f"unknown mode: {mode}")


def tile_name(lat_ll: int, lon_ll: int) -> str:
    """Return `N<NN><E|W><EEE>` where (lat_ll, lon_ll) are LOWER-LEFT corner."""
    lat_dir = "N" if lat_ll >= 0 else "S"
    lat_idx = lat_ll if lat_ll >= 0 else -lat_ll
    lon_dir = "E" if lon_ll >= 0 else "W"
    lon_idx = lon_ll if lon_ll >= 0 else -lon_ll
    return f"{lat_dir}{lat_idx:02d}{lon_dir}{lon_idx:03d}"


def tile_filename(lat_ll: int, lon_ll: int) -> str:
    """Filename matching `_resolve_wc_path` expectations."""
    return f"ESA_WorldCover_10m_2021_v200_{tile_name(lat_ll, lon_ll)}_Map.tif"


def tile_url(lat_ll: int, lon_ll: int) -> str:
    return f"{S3_BASE}/{tile_filename(lat_ll, lon_ll)}"


def enumerate_tiles(south: float, west: float, north: float, east: float) -> list[tuple[int, int]]:
    """Enumerate 3°×3° tile lower-left corners covering [south, north) × [west, east).

    Convention: tile with lower-left (lat_ll, lon_ll) covers
        lat ∈ [lat_ll, lat_ll + 3) and lon ∈ [lon_ll, lon_ll + 3).
    Returns the tiles whose footprint intersects the bbox interior.
    """
    lat_lo = _snap_to_step(south, TILE_STEP_DEG, mode="floor")
    lon_lo = _snap_to_step(west, TILE_STEP_DEG, mode="floor")
    lat_hi = int(math.floor((north - 1e-9) / TILE_STEP_DEG) * TILE_STEP_DEG)
    lon_hi = int(math.floor((east - 1e-9) / TILE_STEP_DEG) * TILE_STEP_DEG)
    tiles: list[tuple[int, int]] = []
    lat = lat_lo
    while lat <= lat_hi:
        lon = lon_lo
        while lon <= lon_hi:
            tiles.append((lat, lon))
            lon += TILE_STEP_DEG
        lat += TILE_STEP_DEG
    return tiles


# ─── download helpers ───────────────────────────────────────────────────────

def download_tile(
    lat_ll: int, lon_ll: int, out_dir: Path,
    *, retries: int = 3, timeout: float = 120.0,
) -> tuple[Path, str]:
    """Download one tile. Returns (out_path, status) where status in
    {downloaded, skipped, not_found, error}.

    Min size = 50 KB. WC tiles are highly compressed COGs but a 3°×3° tile
    with any land coverage is typically 2-130 MB. 404 = tile not in catalogue
    (ocean / outside the 2651-tile set).
    """
    fname = tile_filename(lat_ll, lon_ll)
    url = tile_url(lat_ll, lon_ll)
    out_path = out_dir / fname

    MIN_BYTES = 50 * 1024  # 50 KB floor: filter empty / placeholder files
    if out_path.exists() and out_path.stat().st_size >= MIN_BYTES:
        return out_path, "skipped"

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "downscalewind/ingest_worldcover_esa"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(out_path, "wb") as f:
                # Stream by 1 MB chunks (typical tile ~10-130 MB).
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
            if e.code in (403, 404):
                # 404 = tile does not exist (ocean / outside catalogue).
                # ESA S3 sometimes returns 403 on non-existent keys.
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
              help="Bounding box 'S,W,N,E' in degrees, e.g. '35,-10,52,8' for FR+ES+PT.")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path),
              help="Directory to write tiles into (e.g. data/raw/worldcover_esa/).")
@click.option("--skip-existing", is_flag=True, default=True,
              help="Skip tiles already present and ≥50 KB (default).")
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
    logger.info("bbox = (S=%.2f, W=%.2f, N=%.2f, E=%.2f) → %d candidate tiles (3°×3°)",
                south, west, north, east, len(tiles))

    counts = {"downloaded": 0, "skipped": 0, "not_found": 0, "error": 0}
    total_mb = 0.0
    for lat_ll, lon_ll in tiles:
        fname = tile_filename(lat_ll, lon_ll)
        if dry_run:
            existing = (output_dir / fname).exists()
            url = tile_url(lat_ll, lon_ll)
            logger.info("  [dry-run] %s %s -> %s",
                        "EXISTS" if existing else "MISSING", fname, url)
            continue
        existing_ok = (
            skip_existing
            and (output_dir / fname).exists()
            and (output_dir / fname).stat().st_size >= 50 * 1024
        )
        if existing_ok:
            counts["skipped"] += 1
            size_mb = (output_dir / fname).stat().st_size / 1e6
            logger.info("  [skip] %s (already %.1f MB)", fname, size_mb)
            total_mb += size_mb
            continue
        t0 = time.time()
        out, status = download_tile(lat_ll, lon_ll, output_dir, retries=retries, timeout=timeout)
        counts[status] += 1
        if status == "downloaded":
            size_mb = out.stat().st_size / 1e6
            total_mb += size_mb
            logger.info("  [ok ] %s (%.1f MB, %.1f s)", fname, size_mb, time.time() - t0)
        elif status == "not_found":
            logger.info("  [404] %s (ocean / outside ESA WC catalogue)", fname)
        elif status == "skipped":
            pass
        else:
            logger.error("  [err] %s", fname)

    logger.info(
        "DONE | downloaded=%d skipped=%d not_found=%d error=%d / total=%d (≈%.0f MB)",
        counts["downloaded"], counts["skipped"], counts["not_found"], counts["error"],
        len(tiles), total_mb,
    )
    if counts["error"] > 0 and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    cli()
