"""
build_run_matrix.py — Join sites × timestamps into a CFD run matrix.

Reads sites.csv (from sample_sites.py) and directions.csv (from
sample_directions.py), merges on site_id, assigns run IDs and priorities,
and outputs a ready-to-run matrix.

Usage
-----
    python build_run_matrix.py \
        --sites ../../data/campaign/sites/sites.csv \
        --directions ../../data/campaign/directions/all_directions.csv \
        --output ../../data/campaign/run_matrix.csv

Dependencies: numpy, pandas
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Runtime model (empirical from TBM L2 PoC)
SEC_PER_RUN = 22.0  # 165k cells, 500 iter, 24 cores


def assign_priority(speed: float) -> str:
    """Priority based on 10m wind speed.

    - high:   moderate (4–9 m/s) — most common, best for training
    - medium: strong (≥9 m/s) — important for extremes
    - low:    weak (1–4 m/s) — near-calm, less informative
    """
    if speed < 4.0:
        return "low"
    elif speed < 9.0:
        return "high"
    else:
        return "medium"


def build_matrix(
    sites: pd.DataFrame,
    directions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sites and directions into a run matrix.

    If directions has a site_id column, merge per-site.
    Otherwise, cross-join (all sites × all directions).
    """
    # Normalise column names
    for df in [sites, directions]:
        if "lon_center" in df.columns:
            df.rename(columns={"lon_center": "lon", "lat_center": "lat"},
                      inplace=True)

    if "site_id" in directions.columns:
        # Per-site timestamps: inner join
        site_cols = ["site_id", "lat", "lon", "mean_elev", "std_elev",
                     "cluster_id"]
        site_cols = [c for c in site_cols if c in sites.columns]

        matrix = directions.merge(
            sites[site_cols], on="site_id", how="inner",
            suffixes=("", "_site"),
        )
        # Resolve conflicts
        if "cluster_id_site" in matrix.columns:
            matrix.rename(columns={
                "cluster_id": "wind_cluster_id",
                "cluster_id_site": "terrain_cluster_id",
            }, inplace=True)
    else:
        # Cross-join
        sites_cp = sites.copy()
        directions_cp = directions.copy()
        sites_cp["_key"] = 1
        directions_cp["_key"] = 1
        matrix = sites_cp.merge(directions_cp, on="_key").drop("_key", axis=1)

    # Priority
    if "speed_10m" in matrix.columns:
        matrix["priority"] = matrix["speed_10m"].apply(assign_priority)
    else:
        matrix["priority"] = "normal"

    matrix["status"] = "pending"

    # Sort by priority then site
    prio_order = {"high": 0, "medium": 1, "low": 2, "normal": 1}
    matrix["_prio"] = matrix["priority"].map(prio_order)
    matrix.sort_values(["_prio", "site_id", "timestamp"], inplace=True)
    matrix.drop(columns=["_prio"], inplace=True)
    matrix.reset_index(drop=True, inplace=True)

    # Assign run IDs
    matrix.insert(0, "run_id", [f"run_{i:06d}" for i in range(len(matrix))])

    return matrix


def print_summary(matrix: pd.DataFrame, n_cores: int = 48) -> None:
    """Print summary statistics and estimates."""
    n_runs = len(matrix)
    n_sites = matrix["site_id"].nunique() if "site_id" in matrix.columns else "?"
    per_site = n_runs / max(1, matrix["site_id"].nunique()) if "site_id" in matrix.columns else "?"

    parallel = max(1, n_cores // 24)
    wall_s = n_runs * SEC_PER_RUN / parallel
    wall_h = wall_s / 3600

    print(f"\n{'='*60}")
    print(f"RUN MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Total runs:          {n_runs:,}")
    print(f"  Unique sites:        {n_sites}")
    print(f"  Timestamps/site:     {per_site:.1f} (avg)")

    # Priority breakdown
    prio = matrix["priority"].value_counts()
    for p in ["high", "medium", "low"]:
        print(f"  {p:>8s} priority:  {prio.get(p, 0):,}")

    # Speed stats
    if "speed_10m" in matrix.columns:
        sp = matrix["speed_10m"]
        print(f"  Speed range:         {sp.min():.1f} – {sp.max():.1f} m/s")

    # Storage
    zarr_gb = n_runs * 4.5 / 1024
    archive_gb = n_runs * 7.6 / 1024
    n_sites_int = matrix["site_id"].nunique() if "site_id" in matrix.columns else n_runs
    mesh_gb = n_sites_int * 8 / 1024
    total_gb = zarr_gb + archive_gb + mesh_gb

    print(f"\n  Estimated HPC time:  {wall_h:,.1f} h ({wall_h/24:.1f} days, "
          f"{n_cores} cores)")
    print(f"  Storage — Zarr:      {zarr_gb:,.1f} GB")
    print(f"  Storage — OF arch:   {archive_gb:,.1f} GB")
    print(f"  Storage — meshes:    {mesh_gb:,.1f} GB")
    print(f"  Storage — TOTAL:     {total_gb:,.1f} GB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build CFD run matrix from sites and timestamps"
    )
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("../../data/campaign/run_matrix.csv"))
    parser.add_argument("--n-cores", type=int, default=48,
                        help="HPC cores for time estimate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sites = pd.read_csv(args.sites)
    directions = pd.read_csv(args.directions)
    logger.info("Loaded %d sites and %d direction entries", len(sites), len(directions))

    matrix = build_matrix(sites, directions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, index=False)
    logger.info("Saved %d runs → %s", len(matrix), args.output)

    print_summary(matrix, n_cores=args.n_cores)


if __name__ == "__main__":
    main()
