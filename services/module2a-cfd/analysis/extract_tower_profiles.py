"""
extract_tower_profiles.py — Extract CFD wind/T/q at ICOS tower locations.

Reads each site's stacked Zarr, finds the cell column closest to (0,0)
(tower is at domain centre), and extracts vertical profiles at all z_agl
levels. Outputs a CSV with one row per (site, case, height).

Usage (on Aqua):
    python services/module2a-cfd/analysis/extract_tower_profiles.py \
        --campaign-dir data/campaign/icos_fwi_v1/cases \
        --output data/campaign/icos_fwi_v1/tower_profiles.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def extract_site(site_dir: Path, site_id: str) -> pd.DataFrame:
    """Extract tower column from a site Zarr."""
    zarr_path = site_dir / f"{site_id}.zarr"
    if not zarr_path.exists():
        print(f"  {site_id}: no Zarr found, skipping")
        return pd.DataFrame()

    root = zarr.open_group(str(zarr_path), mode="r")
    x = np.array(root["coords/x"][:])
    y = np.array(root["coords/y"][:])
    z_agl = np.array(root["coords/z_agl"][:])

    # Tower is at domain centre (0, 0) — find the vertical column
    r2 = x**2 + y**2
    # Get unique z levels near the tower (cells within 100m of centre)
    near_mask = r2 < 100**2
    if near_mask.sum() == 0:
        near_mask = r2 < 500**2
    if near_mask.sum() == 0:
        print(f"  {site_id}: no cells near tower, skipping")
        return pd.DataFrame()

    # For each unique z_agl level in the near-tower cells, pick the closest cell
    near_z = z_agl[near_mask]
    near_idx = np.where(near_mask)[0]
    near_r2 = r2[near_mask]

    # Group by z level (round to 5m bins)
    z_bins = np.round(near_z / 5) * 5
    unique_z = np.unique(z_bins)
    # Keep only heights up to 300m (relevant for tower comparison)
    unique_z = unique_z[unique_z <= 300]

    best_cells = []
    for zb in unique_z:
        mask_z = z_bins == zb
        idx_in_near = np.argmin(near_r2[mask_z])
        cell_idx = near_idx[mask_z][idx_in_near]
        best_cells.append((zb, cell_idx))

    # Read fields
    U = np.array(root["U"][:])        # (n_cases, n_cells, 3)
    T = np.array(root["T"][:])        # (n_cases, n_cells)
    q = np.array(root["q"][:])        # (n_cases, n_cells)
    case_ids = np.array(root["meta/case_id"][:])
    wind_dirs = np.array(root["meta/wind_dir"][:])
    u_hubs = np.array(root["meta/u_hub"][:])

    # Try to decode case_ids if bytes
    if case_ids.dtype.kind == 'S' or case_ids.dtype.kind == 'O':
        case_ids = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in case_ids])

    n_cases = U.shape[0]
    rows = []
    for i_case in range(n_cases):
        for z_level, cell_idx in best_cells:
            u, v, w = U[i_case, cell_idx, :]
            ws = float(np.sqrt(u**2 + v**2))
            wd = float((270 - np.degrees(np.arctan2(v, u))) % 360)
            rows.append({
                "site_id": site_id,
                "case_id": str(case_ids[i_case]),
                "z_agl_m": float(z_level),
                "u": float(u),
                "v": float(v),
                "w": float(w),
                "ws": ws,
                "wd": wd,
                "T": float(T[i_case, cell_idx]),
                "q": float(q[i_case, cell_idx]),
                "wind_dir_inflow": float(wind_dirs[i_case]),
                "u_hub_inflow": float(u_hubs[i_case]),
            })

    df = pd.DataFrame(rows)
    print(f"  {site_id}: {n_cases} cases × {len(best_cells)} heights = {len(df)} rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    frames = []
    for site_dir in sorted(args.campaign_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        site_id = site_dir.name
        if site_id.startswith("."):
            continue
        df = extract_site(site_dir, site_id)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("ERROR: no profiles extracted")
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nTotal: {len(result)} rows, {result['site_id'].nunique()} sites")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
