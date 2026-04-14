"""
enrich_era5_3d.py — Add ERA5 3×3 log-law grid to existing training grid.zarr.

Approach A: run prepare_inflow at each of the 9 ERA5 grid points (not at site).
Each returns a physically reconstructed profile on 32 AGL z-levels (log-law +
Monin-Obukhov + cubic spline). Same reconstruction as the stored 1D profile.

Output: input/era5_3d/{u,v,T,q,k}  →  (3, 3, 32) per variable
        input/era5_3d/lat           →  (3,)
        input/era5_3d/lon           →  (3,)

Usage:
    python enrich_era5_3d.py \
        --training-dir /path/to/training_9k \
        --era5-zarr /path/to/era5_perdigao.zarr \
        --run-matrix /path/to/run_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

Z_LEVELS_AGL = np.geomspace(5, 5000, 32).astype(np.float32)
CMU = 0.09


def build_logflaw_library(era5_zarr: Path, timestamps: list[str],
                          era5_lats: np.ndarray,
                          era5_lons: np.ndarray) -> dict:
    """Run prepare_inflow at each of the 9 ERA5 grid points for each timestamp.

    Returns: {ts_str: {u: (3,3,32), v: (3,3,32), T: (3,3,32), q: (3,3,32), k: (3,3,32)}}
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module2a-cfd"))
    from prepare_inflow import prepare_inflow as _prepare

    library = {}
    n_z = len(Z_LEVELS_AGL)

    for ts_str in timestamps:
        profiles = {v: np.zeros((3, 3, n_z), dtype=np.float32)
                    for v in ["u", "v", "T", "q", "k"]}

        for i, lat in enumerate(era5_lats):
            for j, lon in enumerate(era5_lons):
                try:
                    result = _prepare(
                        era5_zarr=str(era5_zarr),
                        timestamp=ts_str.replace(" ", "T"),
                        site_lat=float(lat),
                        site_lon=float(lon),
                    )
                except Exception as e:
                    logger.warning("  %s @ (%.2f, %.2f): %s", ts_str, lat, lon, e)
                    continue

                z_src = np.array(result["z_levels"], dtype=np.float32)
                ux = np.array(result["ux_profile"], dtype=np.float32)
                uy = np.array(result["uy_profile"], dtype=np.float32)
                T = np.array(result.get("T_profile", [result["T_ref"]] * len(z_src)),
                             dtype=np.float32)
                q = np.array(result.get("q_profile", [0.0] * len(z_src)),
                             dtype=np.float32)

                # Interpolate to 32 AGL levels
                profiles["u"][i, j] = np.interp(Z_LEVELS_AGL, z_src, ux)
                profiles["v"][i, j] = np.interp(Z_LEVELS_AGL, z_src, uy)
                profiles["T"][i, j] = np.interp(Z_LEVELS_AGL, z_src, T)
                profiles["q"][i, j] = np.interp(Z_LEVELS_AGL, z_src, q)

                # k from u_star (constant with height in Parente formulation)
                u_star = float(result.get("u_star", 0.3))
                profiles["k"][i, j, :] = u_star**2 / np.sqrt(CMU)

        library[ts_str] = profiles
        logger.info("  %s: 9 columns OK", ts_str)

    return library


def enrich_case(case_dir: Path, era5_3d: dict,
                lats: np.ndarray, lons: np.ndarray) -> bool:
    """Add or replace input/era5_3d/ in grid.zarr."""
    import zarr
    grid_path = case_dir / "grid.zarr"
    if not grid_path.exists():
        return False
    store = zarr.open_group(str(grid_path), mode="r+")

    if "input/era5_3d" in store:
        del store["input/era5_3d"]

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
        description="Enrich training grid.zarr with log-law reconstructed ERA5 3×3")
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--era5-zarr", default=None, type=Path,
                        help="Single shared ERA5 Zarr (legacy mode)")
    parser.add_argument("--era5-dir", default=None, type=Path,
                        help="Directory with per-site era5_site_NNNNN.zarr")
    parser.add_argument("--sites-csv", default=None, type=Path,
                        help="sites.csv for per-site mode")
    parser.add_argument("--run-matrix", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    # Mode: per-site ERA5 (correct) or shared (legacy)
    per_site_mode = args.era5_dir is not None
    if per_site_mode and not args.sites_csv:
        raise ValueError("--era5-dir requires --sites-csv")

    # Get unique timestamps
    ts_by_site = {}
    with open(args.run_matrix) as f:
        for row in csv.DictReader(f):
            ts_by_site.setdefault(row["site_id"], []).append(row["timestamp"])
    all_timestamps = sorted({ts for tss in ts_by_site.values() for ts in tss})
    logger.info("%d unique timestamps, %d sites in run_matrix",
                len(all_timestamps), len(ts_by_site))

    # Per-site coordinates lookup
    site_coords = {}
    if args.sites_csv:
        with open(args.sites_csv) as f:
            for row in csv.DictReader(f):
                site_coords[row["site_id"]] = (float(row["lat"]), float(row["lon"]))

    # Build log-law libraries (one per site if per_site_mode, else shared)
    if per_site_mode:
        logger.info("Building per-site log-law libraries...")
        site_libraries = {}  # site_id → {ts_str: era5_3d}
        site_grids = {}      # site_id → (lats, lons)
        t0 = time.time()
        import zarr as zarr_lib
        for sid, (slat, slon) in site_coords.items():
            era5_path = args.era5_dir / f"era5_{sid}.zarr"
            if not era5_path.exists():
                continue
            store = zarr_lib.open_group(str(era5_path), mode="r")
            site_lats = np.array(store["coords/lat"][:], dtype=np.float32)
            site_lons = np.array(store["coords/lon"][:], dtype=np.float32)
            site_grids[sid] = (site_lats, site_lons)
            # Build log-law lib for this site (15 timestamps × 9 cols = 135 calls)
            site_libraries[sid] = build_logflaw_library(
                era5_path, all_timestamps, site_lats, site_lons)
        logger.info("Built %d site libraries in %.0fs",
                    len(site_libraries), time.time() - t0)
    else:
        # Legacy: single shared ERA5
        store = zarr.open_group(str(args.era5_zarr), mode="r")
        era5_lats = np.array(store["coords/lat"][:], dtype=np.float32)
        era5_lons = np.array(store["coords/lon"][:], dtype=np.float32)
        logger.info("Shared ERA5 grid: lats=%s lons=%s", era5_lats, era5_lons)
        t0 = time.time()
        shared_library = build_logflaw_library(
            args.era5_zarr, all_timestamps, era5_lats, era5_lons)
        logger.info("Library built in %.0fs", time.time() - t0)

    # Lookup (wind_dir, u_hub) → timestamp for sites NOT in run_matrix
    ts_lookup = {}
    for ts_idx, ts_str in enumerate(ts_by_site.get("site_00000", all_timestamps)):
        case_id = f"site_00000_case_ts{ts_idx:03d}"
        inflow_path = args.training_dir / case_id / "inflow.json"
        if inflow_path.exists():
            with open(inflow_path) as f:
                inf = json.load(f)
            key = (round(inf["wind_dir"], 1), round(inf["u_hub"], 2))
            ts_lookup[key] = ts_str

    # Enrich all cases
    case_dirs = sorted(args.training_dir.iterdir())
    case_dirs = [d for d in case_dirs if d.is_dir() and d.name.startswith("site_")]
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]

    logger.info("Enriching %d cases...", len(case_dirs))
    t0 = time.time()
    n_ok = n_fail = n_verify_ok = n_verify_fail = 0

    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name
        site_id = "_".join(case_id.split("_")[:2])
        ts_idx = int(case_id.split("_ts")[-1])

        ts_str = None
        if site_id in ts_by_site and ts_idx < len(ts_by_site[site_id]):
            ts_str = ts_by_site[site_id][ts_idx]
        else:
            inflow_path = case_dir / "inflow.json"
            if inflow_path.exists():
                with open(inflow_path) as f:
                    inf = json.load(f)
                key = (round(inf["wind_dir"], 1), round(inf["u_hub"], 2))
                ts_str = ts_lookup.get(key)

        if ts_str is None:
            n_fail += 1
            continue

        # Get the ERA5 3D for this case
        if per_site_mode:
            if site_id not in site_libraries:
                n_fail += 1
                continue
            library = site_libraries[site_id]
            site_lats, site_lons = site_grids[site_id]
        else:
            library = shared_library
            site_lats, site_lons = era5_lats, era5_lons

        era5_3d = library.get(ts_str)
        if era5_3d is None:
            n_fail += 1
            continue

        # Verify center matches stored 1D
        if args.verify:
            import zarr as zarr_lib
            store = zarr_lib.open_group(str(case_dir / "grid.zarr"), mode="r")
            if "input/era5/u" in store:
                stored_u = np.array(store["input/era5/u"][:])
                center_u = era5_3d["u"][1, 1, :]
                max_du = np.max(np.abs(center_u - stored_u))
                if max_du < 0.1:
                    n_verify_ok += 1
                else:
                    n_verify_fail += 1
                    if n_verify_fail <= 3:
                        logger.warning("  %s: center max|Δu|=%.4f",
                                       case_id, max_du)

        ok = enrich_case(case_dir, era5_3d, site_lats, site_lons)
        if ok:
            n_ok += 1

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d cases (%.1f s)",
                        i + 1, len(case_dirs), time.time() - t0)

    logger.info("Done: %d enriched, %d failed in %.0fs",
                n_ok, n_fail, time.time() - t0)
    if args.verify:
        logger.info("Verification: %d OK, %d mismatch",
                    n_verify_ok, n_verify_fail)


if __name__ == "__main__":
    main()
