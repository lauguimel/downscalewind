"""
extract_v2_input_at_coords.py — Build a v2 grid.zarr/input at arbitrary (lat, lon, ts).

Bridges raw geospatial sources (DEM, WorldCover, ERA5 europe Zarr) to the
on-disk schema expected by the trained surrogate v2 (ViT base + AGL100 +
surface). One output Zarr per pairing, ready to be consumed by
`services/module2b-surrogate/src/dataset_v2_vit.py`.

Two modes:

(1) Single coord
    python extract_v2_input_at_coords.py \\
        --lat 39.7 --lon -7.7 --timestamp 2017-05-15T12:00:00 \\
        --output data/inference/test_input/grid.zarr \\
        --era5-store data/raw/era5_europe_spring2017_v2.zarr \\
        --dem data/raw/srtm_perdigao_30m.tif \\
        --worldcover data/raw/worldcover_perdigao.tif

(2) Batch from an obs_unified Zarr (one grid.zarr per (station, timestamp))
    python extract_v2_input_at_coords.py \\
        --stations-zarr data/raw/obs_unified_perdigao.zarr \\
        --era5-store data/raw/era5_europe_spring2017_v2.zarr \\
        --dem data/raw/srtm_perdigao_30m.tif \\
        --worldcover data/raw/worldcover_perdigao.tif \\
        --output-dir data/inference/stations_perdigao/ \\
        --max-stations 3 --max-timestamps 1 --smoke
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import click
import numpy as np

logger = logging.getLogger("extract_v2_input")

# Make local utils import work when the script is launched as
# `python services/module2b-surrogate/extract_v2_input_at_coords.py`.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.inference_input import (  # noqa: E402
    NI, NJ, NK,
    build_native_z,
    compute_z0_eff_from_wc,
    extract_era5_at_coords,
    extract_terrain_from_dem,
    write_input_grid_zarr,
)


# ─── Per-pairing builder ────────────────────────────────────────────────────

def build_one(
    *,
    site_id: str,
    lat: float,
    lon: float,
    timestamp_iso: str,
    era5_store: Path,
    dem: Path,
    worldcover: Path | None,
    output: Path,
    overwrite: bool = False,
    extra_meta: dict | None = None,
    max_era5_delta_h: float = 3.5,
) -> Path:
    """Build a single grid.zarr at (lat, lon, ts). Returns the output path."""
    t0 = time.time()
    logger.info("[%s @ %s] lat=%.4f lon=%.4f → %s",
                site_id, timestamp_iso, lat, lon, output)

    terrain = extract_terrain_from_dem(dem, lat, lon)
    if terrain.shape != (NI, NJ):
        raise RuntimeError(f"terrain shape {terrain.shape} != ({NI},{NJ})")

    if worldcover is not None and Path(worldcover).exists():
        z0_eff, wc_counts = compute_z0_eff_from_wc(worldcover, lat, lon)
        logger.debug("  z0_eff=%.4f m (WC classes=%s)", z0_eff, wc_counts)
    else:
        logger.warning("  WC tile missing — falling back to z0_eff=0.05 m")
        z0_eff = 0.05

    era5 = extract_era5_at_coords(
        era5_store, lat, lon, timestamp_iso,
        max_delta_seconds=max_era5_delta_h * 3600 + 1.0,
    )
    if era5.delta_seconds > 0:
        logger.debug("  ERA5 nearest %s (Δ=%.0fs)",
                     era5.actual_timestamp_iso, era5.delta_seconds)

    z_grid = build_native_z(terrain)
    out = write_input_grid_zarr(
        output,
        site_id=site_id,
        lat=lat, lon=lon,
        terrain=terrain,
        z_grid=z_grid,
        z0_eff=z0_eff,
        era5=era5,
        timestamp_iso=timestamp_iso,
        extra_meta=extra_meta,
        overwrite=overwrite,
    )
    logger.info("  wrote %s (%.2fs)", out, time.time() - t0)
    return out


# ─── Smoke verification helper ──────────────────────────────────────────────

def verify_smoke(path: Path) -> dict:
    """Open a written grid.zarr/input and return its shapes for sanity check."""
    import zarr
    g = zarr.open_group(str(path), mode="r")
    out = {
        "terrain_shape": tuple(g["input/terrain"].shape),
        "z_shape": tuple(g["coords/z"].shape),
        "x_shape": tuple(g["coords/x"].shape),
        "y_shape": tuple(g["coords/y"].shape),
        "n_pressure_levels": int(g["input/era5_pressure_levels"].shape[0]),
        "era5_3d_vars": sorted(list(g["input/era5_3d"])),
        "era5_3d_u_shape": tuple(g["input/era5_3d/u"].shape),
        "era5_surface_vars": sorted(list(g["input/era5_surface"])),
        "z0_eff": float(g["input"].attrs.get("z0_eff", float("nan"))),
        "lat": float(g["input"].attrs.get("lat", float("nan"))),
        "inflow_meta": dict(g["input/inflow_meta"].attrs),
    }
    return out


def _check_v2_compat(shapes: dict) -> list[str]:
    """Return a list of mismatch messages vs the expected v2 schema."""
    issues: list[str] = []
    if shapes["terrain_shape"] != (NI, NJ):
        issues.append(f"terrain shape {shapes['terrain_shape']} != ({NI},{NJ})")
    if shapes["z_shape"] != (NI, NJ, NK):
        issues.append(f"coords/z shape {shapes['z_shape']} != ({NI},{NJ},{NK})")
    if shapes["era5_3d_u_shape"][:2] != (3, 3):
        issues.append(f"era5_3d/u shape {shapes['era5_3d_u_shape']} not (3,3,N_p)")
    for var in ("u", "v", "T", "q"):
        if var not in shapes["era5_3d_vars"]:
            issues.append(f"era5_3d missing var: {var}")
    for var in ("t2m", "d2m", "u10", "v10"):
        if var not in shapes["era5_surface_vars"]:
            issues.append(f"era5_surface missing var: {var}")
    return issues


# ─── Stations Zarr enumeration (batch mode) ─────────────────────────────────

def iter_station_pairings(
    stations_zarr: Path,
    *,
    max_stations: int | None = None,
    max_timestamps: int | None = None,
) -> Iterable[tuple[str, float, float, float, str]]:
    """Yield (site_id, lat, lon, elev, timestamp_iso) tuples from an obs Zarr.

    For each station, samples up to `max_timestamps` non-NaN timestamps from the
    `data/wind_speed` (or first non-empty data variable) at the 10 m height. This
    keeps the smoke fast while still validating the pipeline on real coords.
    """
    import zarr
    g = zarr.open_group(str(stations_zarr), mode="r")
    station_ids = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)
                   for s in g["stations/station_id"][:]]
    lats = np.asarray(g["stations/lat"][:], dtype=np.float32)
    lons = np.asarray(g["stations/lon"][:], dtype=np.float32)
    elevs = np.asarray(g["stations/elev"][:], dtype=np.float32)
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    heights = np.asarray(g["heights/height_m"][:], dtype=np.float32)
    h10_idx = int(np.argmin(np.abs(heights - 10.0)))

    # Pick first variable that has any non-NaN entries to drive timestamp choice
    data_grp = g["data"]
    candidate_vars = [v for v in ("wind_speed", "u", "v", "t2m")
                      if v in data_grp]
    if not candidate_vars:
        raise ValueError(f"No usable data var in {stations_zarr}/data")
    var_name = candidate_vars[0]

    n_stations = len(station_ids)
    if max_stations is not None:
        n_stations = min(n_stations, max_stations)

    for s_idx in range(n_stations):
        slab = np.asarray(data_grp[var_name][:, s_idx, h10_idx], dtype=np.float32)
        valid_t_idx = np.flatnonzero(np.isfinite(slab))
        if valid_t_idx.size == 0:
            logger.warning("Station %s: no valid timestamps at h=10 m for %s",
                           station_ids[s_idx], var_name)
            continue
        if max_timestamps is not None:
            # Uniformly sample up to max_timestamps across the valid range
            step = max(1, valid_t_idx.size // max_timestamps)
            valid_t_idx = valid_t_idx[::step][:max_timestamps]
        for t_idx in valid_t_idx:
            ts_iso = str(np.array(int(times[t_idx])).astype("datetime64[ns]"))
            yield (
                station_ids[s_idx],
                float(lats[s_idx]),
                float(lons[s_idx]),
                float(elevs[s_idx]),
                ts_iso,
            )


# ─── CLI ────────────────────────────────────────────────────────────────────

@click.command(context_settings={"show_default": True})
@click.option("--lat", type=float, default=None, help="Site latitude (single mode)")
@click.option("--lon", type=float, default=None, help="Site longitude (single mode)")
@click.option("--timestamp", type=str, default=None,
              help="ISO timestamp (single mode), e.g. 2017-05-15T12:00:00")
@click.option("--site-id", type=str, default="custom_site",
              help="Site identifier (single mode)")
@click.option("--stations-zarr", type=click.Path(exists=True, path_type=Path),
              default=None, help="obs_unified Zarr for batch mode")
@click.option("--era5-store", type=click.Path(exists=True, path_type=Path),
              required=True, help="ERA5 Zarr (e.g. era5_europe_spring2017_v2.zarr)")
@click.option("--dem", type=click.Path(exists=True, path_type=Path),
              default="data/raw/srtm_perdigao_30m.tif",
              help="DEM GeoTIFF (Copernicus GLO-30 or SRTM)")
@click.option("--worldcover", type=click.Path(exists=True, path_type=Path),
              default="data/raw/worldcover_perdigao.tif",
              help="ESA WorldCover GeoTIFF for z0_eff")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Output grid.zarr path (single mode)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Output directory (batch mode); one grid.zarr per pairing")
@click.option("--max-stations", type=int, default=None,
              help="Limit number of stations (batch mode)")
@click.option("--max-timestamps", type=int, default=1,
              help="Timestamps per station (batch mode)")
@click.option("--max-era5-delta-h", type=float, default=3.5,
              help="Max hours from requested ts to nearest ERA5 time")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--smoke", is_flag=True, default=False,
              help="Print compat check on each output")
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(lat, lon, timestamp, site_id, stations_zarr, era5_store, dem, worldcover,
        output, output_dir, max_stations, max_timestamps, max_era5_delta_h,
        overwrite, smoke, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    single_mode = stations_zarr is None
    if single_mode:
        if lat is None or lon is None or timestamp is None or output is None:
            raise click.BadParameter(
                "Single mode requires --lat, --lon, --timestamp, --output."
            )
        out_path = build_one(
            site_id=site_id,
            lat=float(lat), lon=float(lon),
            timestamp_iso=timestamp,
            era5_store=era5_store,
            dem=dem,
            worldcover=worldcover,
            output=output,
            overwrite=overwrite,
            max_era5_delta_h=max_era5_delta_h,
        )
        if smoke:
            shapes = verify_smoke(out_path)
            issues = _check_v2_compat(shapes)
            logger.info("SMOKE %s: shapes=%s", out_path,
                        {k: v for k, v in shapes.items() if "shape" in k})
            if issues:
                logger.error("SMOKE FAIL: %s", issues)
                sys.exit(2)
            logger.info("SMOKE OK")
        return

    # Batch mode
    if output_dir is None:
        raise click.BadParameter("Batch mode requires --output-dir.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairings = list(iter_station_pairings(
        stations_zarr,
        max_stations=max_stations,
        max_timestamps=max_timestamps,
    ))
    if not pairings:
        logger.error("No usable pairings found in %s", stations_zarr)
        sys.exit(2)
    logger.info("Batch mode: %d pairings", len(pairings))

    n_ok = n_err = 0
    all_issues: list[str] = []
    for sid, plat, plon, pelev, pts in pairings:
        ts_tag = pts.replace(":", "").replace("-", "")[:13]
        out_path = output_dir / f"{sid}_{ts_tag}" / "grid.zarr"
        try:
            build_one(
                site_id=sid,
                lat=plat, lon=plon,
                timestamp_iso=pts,
                era5_store=era5_store,
                dem=dem,
                worldcover=worldcover,
                output=out_path,
                overwrite=overwrite,
                extra_meta={"station_elev": pelev},
                max_era5_delta_h=max_era5_delta_h,
            )
            if smoke:
                shapes = verify_smoke(out_path)
                issues = _check_v2_compat(shapes)
                if issues:
                    all_issues.extend(f"{sid}/{pts}: {i}" for i in issues)
                    logger.error("SMOKE FAIL %s: %s", sid, issues)
                    n_err += 1
                    continue
            n_ok += 1
        except Exception as exc:
            logger.exception("Failed %s @ %s: %s", sid, pts, exc)
            n_err += 1

    logger.info("Done: ok=%d, err=%d (output_dir=%s)", n_ok, n_err, output_dir)
    if smoke and all_issues:
        logger.error("Compat issues:\n  - " + "\n  - ".join(all_issues))
        sys.exit(2)
    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    cli()
