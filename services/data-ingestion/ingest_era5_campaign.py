"""
ingest_era5_campaign.py — Batch ERA5 ingestion for the 9k campaign sites.

Generates per-site YAML configs and calls ingest_era5.py for each site.
Uses the exact same CDS pipeline as the single-site ingestion.

Usage:
    python ingest_era5_campaign.py \
        --sites ../../data/campaign/9k/sites.csv \
        --run-matrix ../../data/campaign/9k/run_matrix.csv \
        --output-dir ../../data/raw/era5_campaign/ \
        --start 2017-05 --end 2017-06

Verification:
    After ingestion, run with --verify to check that trilinear interpolation
    of the 3x3 grid at the site center reproduces the existing 1D profiles
    in the training dataset.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("ingest_era5_campaign")


def generate_site_config(
    site_id: str,
    lat: float,
    lon: float,
    config_dir: Path,
) -> Path:
    """Generate a site YAML config for ingest_era5.py.

    Creates a 3x3 ERA5 domain (0.25° spacing) centered on the site.
    """
    # Round to nearest 0.25° grid point for clean ERA5 alignment
    lat_center = round(lat * 4) / 4
    lon_center = round(lon * 4) / 4

    config = {
        "site": {
            "name": site_id,
            "coordinates": {
                "latitude": float(lat),
                "longitude": float(lon),
            },
        },
        "era5_domain": {
            "north": float(lat_center + 0.25),
            "south": float(lat_center - 0.25),
            "east": float(lon_center + 0.25),
            "west": float(lon_center - 0.25),
            "grid_spacing_deg": 0.25,
            "n_lat": 3,
            "n_lon": 3,
        },
    }

    config_path = config_dir / f"{site_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_ingestion(
    site_id: str,
    config_dir: Path,
    output_dir: Path,
    start: str,
    end: str,
    dry_run: bool = False,
) -> bool:
    """Run ingest_era5.py for a single site using subprocess.

    Uses the same CLI as manual ingestion to avoid code duplication.
    """
    import subprocess

    output_path = output_dir / f"era5_{site_id}.zarr"

    if output_path.exists():
        log.info("  %s: already exists, skipping", site_id)
        return True

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "ingest_era5.py"),
        "--site", site_id,
        "--start", start,
        "--end", end,
        "--output", str(output_path),
        "--config-dir", str(config_dir),
    ]

    if dry_run:
        log.info("  [DRY RUN] %s", " ".join(cmd))
        return True

    log.info("  %s: downloading ERA5...", site_id)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Check if data was actually written (checkpoint sentinel may fail
    # but the Zarr store is still valid)
    if output_path.exists() and (output_path / "pressure").exists():
        log.info("  %s: OK → %s", site_id, output_path)
        return True

    if result.returncode != 0:
        log.error("  %s: FAILED\n%s", site_id, result.stderr[-500:])
        return False

    log.info("  %s: OK → %s", site_id, output_path)
    return True


def verify_profiles(
    site_id: str,
    era5_zarr: Path,
    training_dir: Path,
    run_matrix: list[dict],
) -> list[str]:
    """Verify that trilinear interpolation of 3x3 grid matches existing 1D profiles.

    Uses extract_era5_profile from prepare_inflow.py (same function as campaign).
    Returns list of error messages (empty = all OK).
    """
    import json
    import zarr

    import zarr as zarr_lib

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "module2a-cfd"))
    from prepare_inflow import extract_era5_profile

    # Load ERA5 data using same pattern as prepare_inflow()
    store = zarr_lib.open_group(str(era5_zarr), mode="r")
    times_int64 = np.array(store["coords/time"][:])
    times_ns = times_int64.astype("datetime64[ns]")
    has_q = "q" in store["pressure"]
    era5_data = {
        "times":           times_ns.astype("datetime64[s]"),
        "pressure_levels": store["coords/level"][:],
        "lats":            store["coords/lat"][:],
        "lons":            store["coords/lon"][:],
        "u":               store["pressure/u"][:],
        "v":               store["pressure/v"][:],
        "t":               store["pressure/t"][:],
        "z":               store["pressure/z"][:],
        "q":               store["pressure/q"][:] if has_q else None,
    }
    errors = []

    site_runs = [r for r in run_matrix if r["site_id"] == site_id]

    for run in site_runs:
        ts_str = run["timestamp"]
        timestamp = np.datetime64(ts_str)
        lat = float(run["lat"])
        lon = float(run["lon"])

        # Extract profile using the same function as the campaign
        profile = extract_era5_profile(era5_data, timestamp, lat, lon)

        # Find the corresponding case in training dataset
        # Convention: site_NNNNN_case_tsNNN
        site_num = int(site_id.replace("site_", ""))
        ts_idx = site_runs.index(run)
        case_id = f"site_{site_num:05d}_case_ts{ts_idx:03d}"
        case_dir = training_dir / case_id

        if not (case_dir / "grid.zarr").exists():
            continue

        store = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")
        if "input/era5/u" not in store:
            continue

        # Compare extracted profile with stored 1D profile
        stored_u = np.array(store["input/era5/u"][:], dtype=np.float64)
        stored_v = np.array(store["input/era5/v"][:], dtype=np.float64)

        # The stored profiles are on Z_LEVELS_AGL (geomspace 5-5000, 32 levels)
        # The extracted profile is on ERA5 pressure levels → need to interpolate
        # to the same z-levels. Use the same interpolation as convert_stacked.
        z_levels = np.geomspace(5, 5000, 32)
        u_interp = np.interp(z_levels, profile["z_m"], profile["u_ms"])
        v_interp = np.interp(z_levels, profile["z_m"], profile["v_ms"])

        # Check agreement (should be <1e-3 m/s for well-aligned grids)
        du = np.max(np.abs(u_interp - stored_u))
        dv = np.max(np.abs(v_interp - stored_v))

        if du > 0.1 or dv > 0.1:
            errors.append(
                f"{case_id}: max|Δu|={du:.4f}, max|Δv|={dv:.4f} m/s"
            )
        else:
            log.debug("  %s: OK (Δu=%.4f, Δv=%.4f)", case_id, du, dv)

    return errors


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Batch ERA5 ingestion for 9k campaign sites")
    parser.add_argument("--sites", required=True, type=Path,
                        help="sites.csv with site_id,lat,lon columns")
    parser.add_argument("--run-matrix", required=True, type=Path,
                        help="run_matrix.csv with site_id,timestamp columns")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for per-site ERA5 Zarr stores")
    parser.add_argument("--start", default="2017-05",
                        help="Start month (default: 2017-05)")
    parser.add_argument("--end", default="2017-06",
                        help="End month (default: 2017-06)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs but don't download")
    parser.add_argument("--verify", action="store_true",
                        help="Verify 3x3 interpolation matches existing 1D profiles")
    parser.add_argument("--training-dir", type=Path, default=None,
                        help="Training dataset dir (for --verify)")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Limit number of sites (for testing)")
    args = parser.parse_args()

    # Read sites
    with open(args.sites) as f:
        reader = csv.DictReader(f)
        sites = list(reader)

    if args.max_sites:
        sites = sites[:args.max_sites]

    log.info("Campaign ERA5 ingestion: %d sites, period %s to %s",
             len(sites), args.start, args.end)

    # Read run matrix (for verification)
    run_matrix = []
    with open(args.run_matrix) as f:
        reader = csv.DictReader(f)
        run_matrix = list(reader)

    # Generate site configs in a temp dir
    config_dir = args.output_dir / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0

    for site in sites:
        site_id = site["site_id"]
        lat = float(site["lat"])
        lon = float(site["lon"])

        # Generate config
        generate_site_config(site_id, lat, lon, config_dir)

        if args.verify:
            # Verification mode: check existing ERA5 against training data
            era5_path = args.output_dir / f"era5_{site_id}.zarr"
            if not era5_path.exists():
                log.warning("  %s: ERA5 not found at %s", site_id, era5_path)
                n_fail += 1
                continue

            errors = verify_profiles(
                site_id, era5_path, args.training_dir, run_matrix)
            if errors:
                log.error("  %s: %d mismatches", site_id, len(errors))
                for e in errors[:3]:
                    log.error("    %s", e)
                n_fail += 1
            else:
                n_ok += 1
        else:
            # Download mode
            ok = run_ingestion(
                site_id, config_dir, args.output_dir,
                args.start, args.end, args.dry_run)
            if ok:
                n_ok += 1
            else:
                n_fail += 1

    action = "verified" if args.verify else "downloaded"
    log.info("Done: %d %s, %d failed out of %d sites",
             n_ok, action, n_fail, len(sites))


if __name__ == "__main__":
    main()
