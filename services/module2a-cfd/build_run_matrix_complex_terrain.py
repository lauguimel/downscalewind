"""
build_run_matrix_complex_terrain.py — Build run matrix for complex_terrain_v1.

For each site in sites.csv, selects N diverse timestamps (k-means on surface
wind speed + direction in the per-site ERA5 zarr) and writes a run_matrix.csv.

Usage
-----
    python services/module2a-cfd/build_run_matrix_complex_terrain.py \\
        --sites data/campaign/complex_terrain_v1/sites.csv \\
        --era5-dir /scratch/maitreje/dsw/era5_campaign_v3 \\
        --n-timestamps 15 \\
        --out data/campaign/complex_terrain_v1/run_matrix.csv \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def select_timestamps_for_site(era5_zarr: Path, n_ts: int,
                                seed: int = 42) -> list[str]:
    """Select n_ts diverse timestamps for a site via stratified sampling on
    wind (speed, direction). Uses a 2D bin on speed quantile × wind sector;
    picks the median of each occupied bin. No sklearn dependency.

    Samples from the CENTER cell of the 3×3 per-site ERA5 grid.
    """
    import zarr

    store = zarr.open_group(str(era5_zarr), mode="r")
    times = store["coords/time"][:]
    u10 = store["surface/u10"][:, 1, 1]
    v10 = store["surface/v10"][:, 1, 1]

    speed = np.sqrt(u10**2 + v10**2)
    direction = np.degrees(np.arctan2(-u10, -v10)) % 360

    n_ts_eff = min(n_ts, len(times))
    rng = np.random.default_rng(seed)

    # Stratify into roughly sqrt(n_ts) speed bins × sqrt(n_ts) direction bins
    n_speed = max(1, int(np.ceil(np.sqrt(n_ts_eff))))
    n_dir = max(1, int(np.ceil(n_ts_eff / n_speed)))

    # Speed bins as quantiles (avoid empty bins if distribution is skewed)
    speed_edges = np.quantile(speed, np.linspace(0, 1, n_speed + 1))
    speed_edges[-1] += 1e-6  # include max
    speed_bin = np.digitize(speed, speed_edges) - 1
    speed_bin = np.clip(speed_bin, 0, n_speed - 1)

    # Direction sectors evenly spaced
    dir_bin = (direction / (360.0 / n_dir)).astype(int) % n_dir

    # Combined bin id
    bin_id = speed_bin * n_dir + dir_bin
    # Collect occupied bins
    occupied = np.unique(bin_id)

    selected: list[int] = []
    if len(occupied) >= n_ts_eff:
        chosen_bins = rng.choice(occupied, size=n_ts_eff, replace=False)
    else:
        chosen_bins = list(occupied)
        remaining = n_ts_eff - len(chosen_bins)
        # Fill with extra picks from random occupied bins
        chosen_bins = list(chosen_bins) + list(
            rng.choice(occupied, size=remaining, replace=True))

    for b in chosen_bins:
        idxs = np.where(bin_id == b)[0]
        pick = int(rng.choice(idxs))
        selected.append(pick)

    selected_idx = sorted(set(selected))
    # Fill up to n_ts_eff if deduplication removed some (rare)
    pool = list(set(range(len(times))) - set(selected_idx))
    while len(selected_idx) < n_ts_eff and pool:
        j = int(rng.choice(pool))
        selected_idx.append(j)
        pool.remove(j)
    selected_idx = sorted(selected_idx)[:n_ts_eff]

    import pandas as pd
    ts = pd.to_datetime(times[selected_idx])
    return [t.isoformat() for t in ts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--era5-dir", required=True, type=Path,
                        help="Directory containing era5_<site_id>.zarr per site")
    parser.add_argument("--n-timestamps", type=int, default=15)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Read sites
    with open(args.sites) as f:
        sites = list(csv.DictReader(f))
    logger.info("Read %d sites", len(sites))

    # Build run matrix
    rows = []
    n_ok = n_fail = 0
    for i, site in enumerate(sites):
        site_id = site["site_id"]
        era5_path = args.era5_dir / f"era5_{site_id}.zarr"
        if not era5_path.exists():
            logger.warning("  %s: no ERA5 (%s), skipping", site_id, era5_path.name)
            n_fail += 1
            continue
        try:
            timestamps = select_timestamps_for_site(era5_path, args.n_timestamps,
                                                    seed=args.seed)
        except Exception as e:
            logger.error("  %s: timestamp selection failed: %s", site_id, e)
            n_fail += 1
            continue

        for ts in timestamps:
            rows.append({
                "run_id": f"run_{len(rows):06d}",
                "site_id": site_id,
                "timestamp": ts,
                "lat": site["lat"],
                "lon": site["lon"],
                "group": site["group"],
                "priority": "high" if site["group"] in ("D_fire", "G_paragliding") else "normal",
                "status": "pending",
            })
        n_ok += 1
        if (i + 1) % 50 == 0:
            logger.info("  progress: %d/%d sites (%d runs)", i + 1, len(sites), len(rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d runs (%d sites OK, %d failed) to %s",
                len(rows), n_ok, n_fail, args.out)


if __name__ == "__main__":
    main()
