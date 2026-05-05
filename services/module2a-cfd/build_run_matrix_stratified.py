"""
build_run_matrix_stratified.py — Stratified timestamp selection per site.

For each site in sites.csv, loads its per-site ERA5 zarr, computes u_hub_est
(speed at ERA5 1000hPa, ~138m) for every available timestamp, stratifies into
3 wind bins (strong > 5 m/s, moderate 2-5 m/s, calm < 2 m/s), and picks
N timestamps per bin (default 5+5+5 = 15) with directional diversity.

Output: new run_matrix.csv with same schema as the original one.

Usage
-----
    # Local (sites.csv local, ERA5 zarrs on Aqua → run remotely)
    python build_run_matrix_stratified.py \\
        --sites data/campaign/complex_terrain_v1/sites.csv \\
        --era5-dir /scratch/maitreje/dsw/era5_campaign_v3 \\
        --output data/campaign/complex_terrain_v1/run_matrix_stratified.csv \\
        --n-strong 5 --n-moderate 5 --n-calm 5 \\
        --thresh-strong 5.0 --thresh-calm 2.0
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("strat")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


def select_diverse_directions(indices: np.ndarray, dirs_deg: np.ndarray,
                              n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n indices with maximally diverse wind directions (greedy)."""
    if len(indices) <= n:
        return indices
    # Greedy farthest-first traversal in directional space
    chosen = [int(rng.choice(indices))]
    remaining = list(set(indices.tolist()) - set(chosen))
    while len(chosen) < n and remaining:
        # For each remaining, compute min angular distance to chosen
        best_i, best_d = None, -1.0
        for i in remaining:
            d_chosen = np.array([abs(((dirs_deg[i] - dirs_deg[c] + 180) % 360) - 180)
                                 for c in chosen])
            d_min = float(d_chosen.min())
            if d_min > best_d:
                best_d, best_i = d_min, i
        chosen.append(best_i)
        remaining.remove(best_i)
    return np.array(sorted(chosen))


def select_for_site(site_id: str, era5_paths: list[Path], n_strong: int,
                    n_moderate: int, n_calm: int, thresh_strong: float,
                    thresh_calm: float, rng: np.random.Generator) -> list[dict]:
    """Return list of {timestamp, u_hub_est, wind_dir, bin} for selected ts.

    era5_paths can be multiple zarrs covering different periods for the same site
    — concatenated and sorted chronologically.
    """
    import zarr
    times_list, u_list, v_list = [], [], []
    for p in era5_paths:
        if not p.exists():
            continue
        g = zarr.open_group(str(p), mode="r")
        times_list.append(np.array(g["coords/time"][:]).astype("datetime64[ns]"))
        u_arr = np.array(g["pressure/u"][:, 0])
        v_arr = np.array(g["pressure/v"][:, 0])
        cy, cx = u_arr.shape[1] // 2, u_arr.shape[2] // 2
        u_list.append(u_arr[:, cy, cx])
        v_list.append(v_arr[:, cy, cx])
    if not times_list:
        return []
    times = np.concatenate(times_list)
    u_c = np.concatenate(u_list)
    v_c = np.concatenate(v_list)
    order = np.argsort(times)
    times = times[order]; u_c = u_c[order]; v_c = v_c[order]
    spd = np.hypot(u_c, v_c)  # per timestamp at central point
    dirs = (np.degrees(np.arctan2(-u_c, -v_c)) + 360.0) % 360.0

    # Bin indices
    idx_strong = np.where(spd > thresh_strong)[0]
    idx_mod = np.where((spd > thresh_calm) & (spd <= thresh_strong))[0]
    idx_calm = np.where(spd <= thresh_calm)[0]

    # Backfill if a bin is short
    n_avail = {"strong": len(idx_strong), "moderate": len(idx_mod), "calm": len(idx_calm)}
    targets = {"strong": n_strong, "moderate": n_moderate, "calm": n_calm}

    # First pass: take what we can from each bin
    chosen = {}
    for label, idx, n in [("strong", idx_strong, n_strong),
                          ("moderate", idx_mod, n_moderate),
                          ("calm", idx_calm, n_calm)]:
        if len(idx) <= n:
            chosen[label] = idx.tolist()
        else:
            chosen[label] = select_diverse_directions(idx, dirs, n, rng).tolist()

    # Backfill shortages from other bins (stronger preferred)
    total_target = n_strong + n_moderate + n_calm
    total_chosen = sum(len(v) for v in chosen.values())
    if total_chosen < total_target:
        deficit = total_target - total_chosen
        # Prefer pulling extras from strong > moderate > calm
        for label, idx, taken in [("strong", idx_strong, chosen["strong"]),
                                   ("moderate", idx_mod, chosen["moderate"]),
                                   ("calm", idx_calm, chosen["calm"])]:
            if deficit <= 0:
                break
            extras = [int(i) for i in idx if i not in taken]
            if not extras:
                continue
            extras_arr = np.array(extras)
            sel = select_diverse_directions(extras_arr, dirs,
                                            min(deficit, len(extras_arr)), rng)
            chosen[label] += sel.tolist()
            deficit -= len(sel)

    rows = []
    for label, indices in chosen.items():
        for i in indices:
            rows.append({
                "timestamp": str(times[i])[:19],
                "u_hub_est": float(spd[i]),
                "wind_dir": float(dirs[i]),
                "bin": label,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sites", required=True, type=Path,
                   help="sites.csv with columns site_id,lat,lon,group,...")
    p.add_argument("--era5-dir", required=True, type=Path, nargs="+",
                   help="One or more directories with per-site ERA5 zarrs "
                        "(era5_<site_id>.zarr). Multiple dirs are concatenated chronologically.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output run_matrix.csv")
    p.add_argument("--n-strong", type=int, default=5)
    p.add_argument("--n-moderate", type=int, default=5)
    p.add_argument("--n-calm", type=int, default=5)
    p.add_argument("--thresh-strong", type=float, default=5.0,
                   help="u_hub_est threshold for 'strong' bin (m/s)")
    p.add_argument("--thresh-calm", type=float, default=2.0,
                   help="u_hub_est threshold for 'calm' bin (m/s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N sites (debug)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load sites
    with open(args.sites) as f:
        sites = list(csv.DictReader(f))
    if args.limit > 0:
        sites = sites[:args.limit]
    logger.info("Sites to process: %d", len(sites))

    # Build run_matrix
    out_rows = []
    bin_counts = {"strong": 0, "moderate": 0, "calm": 0}
    n_skip = 0
    for i, s in enumerate(sites):
        sid = s["site_id"]
        era5_paths = [d / f"era5_{sid}.zarr" for d in args.era5_dir]
        era5_paths = [p for p in era5_paths if p.exists()]
        if not era5_paths:
            n_skip += 1
            continue
        try:
            picks = select_for_site(sid, era5_paths,
                                     args.n_strong, args.n_moderate, args.n_calm,
                                     args.thresh_strong, args.thresh_calm, rng)
        except Exception as e:
            logger.warning("Site %s failed: %s", sid, str(e)[:80])
            n_skip += 1
            continue
        for j, p in enumerate(picks):
            out_rows.append({
                "run_id": f"run_{len(out_rows):06d}",
                "site_id": sid,
                "timestamp": p["timestamp"],
                "lat": s["lat"],
                "lon": s["lon"],
                "group": s.get("group", ""),
                "priority": "high",
                "status": "pending",
                "u_hub_est": f"{p['u_hub_est']:.3f}",
                "wind_dir": f"{p['wind_dir']:.1f}",
                "bin": p["bin"],
            })
            bin_counts[p["bin"]] += 1
        if (i + 1) % 50 == 0:
            logger.info("[%d/%d] sites done, %d runs so far",
                        i + 1, len(sites), len(out_rows))

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "site_id", "timestamp", "lat", "lon", "group",
                  "priority", "status", "u_hub_est", "wind_dir", "bin"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Summary
    logger.info("=" * 60)
    logger.info("DONE: %d runs from %d sites (%d skipped)",
                len(out_rows), len(sites) - n_skip, n_skip)
    logger.info("Bin distribution:")
    for label, n in bin_counts.items():
        pct = 100 * n / max(len(out_rows), 1)
        logger.info("  %-9s : %d runs (%.1f%%)", label, n, pct)
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
