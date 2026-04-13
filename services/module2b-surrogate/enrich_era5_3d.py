"""
enrich_era5_3d.py — Add ERA5 3×3 grid to existing training grid.zarr files.

Reads the shared ERA5 Zarr (era5_perdigao.zarr) and for each case:
  1. Maps case_id → timestamp via run_matrix.csv
  2. Extracts the raw 3×3 ERA5 grid at that timestamp
  3. Converts geopotential → geometric height, interpolates to 32 AGL z-levels
  4. Writes input/era5_3d/{u,v,T,q} (3, 3, 32) + input/era5_3d/lat, lon

Does NOT modify targets or residuals — only adds new input arrays.

Usage:
    python enrich_era5_3d.py \
        --training-dir /path/to/training_9k \
        --era5-zarr /path/to/era5_perdigao.zarr \
        --run-matrix /path/to/run_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

G = 9.81    # m/s²
KAPPA = 0.41  # von Kármán
CMU = 0.09    # k-epsilon constant
Z0_DEFAULT = 0.05  # m, fallback roughness
Z_LEVELS_AGL = np.geomspace(5, 5000, 32).astype(np.float32)


def load_era5(era5_zarr: Path) -> dict:
    """Load ERA5 Zarr into memory."""
    import zarr

    store = zarr.open_group(str(era5_zarr), mode="r")
    times_raw = np.array(store["coords/time"][:])
    times = times_raw.astype("datetime64[ns]").astype("datetime64[s]")

    return {
        "times": times,
        "levels": np.array(store["coords/level"][:]),
        "lats": np.array(store["coords/lat"][:], dtype=np.float32),
        "lons": np.array(store["coords/lon"][:], dtype=np.float32),
        "u": np.array(store["pressure/u"][:]),      # (time, level, lat, lon)
        "v": np.array(store["pressure/v"][:]),
        "t": np.array(store["pressure/t"][:]),
        "z": np.array(store["pressure/z"][:]),       # geopotential [m²/s²]
        "q": np.array(store["pressure/q"][:]) if "q" in store["pressure"] else None,
    }


def extract_3d_at_timestamp(era5: dict, timestamp: np.datetime64) -> dict:
    """Extract 3×3 ERA5 grid at one timestamp, interpolated to AGL z-levels.

    Returns dict with arrays of shape (3, 3, 32) for each variable.
    """
    # Find nearest time
    dt = np.abs(era5["times"] - timestamp)
    t_idx = int(np.argmin(dt))

    n_lat, n_lon = len(era5["lats"]), len(era5["lons"])
    n_z = len(Z_LEVELS_AGL)

    # Geopotential height at each (level, lat, lon)
    z_geo = era5["z"][t_idx] / G  # (level, lat, lon) → meters

    # For each column (lat, lon), interpolate from pressure levels to AGL z-levels
    result = {
        "u": np.zeros((n_lat, n_lon, n_z), dtype=np.float32),
        "v": np.zeros((n_lat, n_lon, n_z), dtype=np.float32),
        "T": np.zeros((n_lat, n_lon, n_z), dtype=np.float32),
        "q": np.zeros((n_lat, n_lon, n_z), dtype=np.float32),
        "k": np.zeros((n_lat, n_lon, n_z), dtype=np.float32),
    }

    for i in range(n_lat):
        for j in range(n_lon):
            z_col = z_geo[:, i, j]  # heights at this column [m]
            order = np.argsort(z_col)
            z_sorted = z_col[order]

            # Ground reference = lowest ERA5 level
            z_ground = z_sorted[0]
            z_agl_targets = Z_LEVELS_AGL + z_ground

            for var in ["u", "v"]:
                vals = era5[var][t_idx, :, i, j][order]
                result[var][i, j, :] = np.interp(
                    z_agl_targets, z_sorted, vals,
                    left=vals[0], right=vals[-1],
                ).astype(np.float32)

            t_vals = era5["t"][t_idx, :, i, j][order]
            result["T"][i, j, :] = np.interp(
                z_agl_targets, z_sorted, t_vals,
                left=t_vals[0], right=t_vals[-1],
            ).astype(np.float32)

            if era5["q"] is not None:
                q_vals = era5["q"][t_idx, :, i, j][order]
                result["q"][i, j, :] = np.interp(
                    z_agl_targets, z_sorted, q_vals,
                    left=q_vals[0], right=q_vals[-1],
                ).astype(np.float32)

            # k (TKE): estimate u* from lowest-level wind speed + log-law
            u_low = result["u"][i, j, 0]
            v_low = result["v"][i, j, 0]
            speed_low = np.sqrt(u_low**2 + v_low**2)
            z_low = Z_LEVELS_AGL[0]  # ~5m
            u_star = max(KAPPA * speed_low / np.log(z_low / Z0_DEFAULT), 0.01)
            k_val = u_star**2 / np.sqrt(CMU)
            result["k"][i, j, :] = k_val

    return result


def build_timestamp_mapping(run_matrix_path: Path) -> dict[str, str]:
    """Build case_id → timestamp mapping from run_matrix.csv.

    Convention: site_NNNNN_case_tsMMM where MMM is the row index
    within each site's runs (0-indexed, in order of run_matrix rows).
    """
    from collections import defaultdict

    site_runs = defaultdict(list)
    with open(run_matrix_path) as f:
        for row in csv.DictReader(f):
            site_runs[row["site_id"]].append(row["timestamp"])

    mapping = {}
    for site_id, timestamps in site_runs.items():
        site_num = int(site_id.replace("site_", ""))
        for ts_idx, ts in enumerate(timestamps):
            case_id = f"site_{site_num:05d}_case_ts{ts_idx:03d}"
            mapping[case_id] = ts

    return mapping


def enrich_case(
    case_dir: Path,
    era5_3d: dict,
    lats: np.ndarray,
    lons: np.ndarray,
) -> bool:
    """Add ERA5 3D arrays to an existing grid.zarr."""
    import zarr

    grid_path = case_dir / "grid.zarr"
    if not grid_path.exists():
        return False

    store = zarr.open_group(str(grid_path), mode="r+")

    # Skip if already enriched
    if "input/era5_3d" in store:
        return True

    grp = store["input"].create_group("era5_3d")
    for var in ["u", "v", "T", "q", "k"]:
        grp.create_array(var, data=era5_3d[var])
    grp.create_array("lat", data=lats)
    grp.create_array("lon", data=lons)

    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Enrich training grid.zarr with ERA5 3×3 data")
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--era5-zarr", required=True, type=Path)
    parser.add_argument("--run-matrix", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    # Load ERA5
    logger.info("Loading ERA5 from %s", args.era5_zarr)
    era5 = load_era5(args.era5_zarr)
    logger.info("ERA5: %d times, %d levels, lats=%s, lons=%s",
                len(era5["times"]), len(era5["levels"]),
                era5["lats"], era5["lons"])

    # Build case → timestamp mapping
    mapping = build_timestamp_mapping(args.run_matrix)
    logger.info("Mapped %d cases to timestamps", len(mapping))

    # Pre-compute ERA5 3D for each unique timestamp (only 15)
    unique_ts = sorted(set(mapping.values()))
    logger.info("Extracting ERA5 3D for %d unique timestamps...", len(unique_ts))
    era5_by_ts = {}
    for ts_str in unique_ts:
        ts = np.datetime64(ts_str.replace(" ", "T"), "s")
        era5_by_ts[ts_str] = extract_3d_at_timestamp(era5, ts)
        logger.info("  %s: OK", ts_str)

    # Enrich all cases
    case_dirs = sorted(args.training_dir.iterdir())
    case_dirs = [d for d in case_dirs if d.is_dir() and d.name.startswith("site_")]
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]

    logger.info("Enriching %d cases...", len(case_dirs))
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name
        ts_str = mapping.get(case_id)
        if ts_str is None:
            n_fail += 1
            continue

        era5_3d = era5_by_ts.get(ts_str)
        if era5_3d is None:
            n_fail += 1
            continue

        ok = enrich_case(case_dir, era5_3d, era5["lats"], era5["lons"])
        if ok:
            n_ok += 1
        else:
            n_skip += 1

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d cases processed", i + 1, len(case_dirs))

    elapsed = time.time() - t0
    logger.info("Done: %d enriched, %d skipped, %d failed in %.1fs",
                n_ok, n_skip, n_fail, elapsed)


if __name__ == "__main__":
    main()
