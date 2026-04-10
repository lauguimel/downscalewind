"""
build_icos_run_matrix.py — Build a run_matrix.csv for the ICOS campaign.

Reads the downloaded ICOS tall-tower Zarr files and samples timestamps
stratified by Beaufort wind speed bins and wind direction octants.

Output format matches run_multisite_campaign.py expectations:
    run_id, site_id, timestamp, lat, lon, group, priority, status

Usage:
    python services/module2a-cfd/build_icos_run_matrix.py \\
        --icos-dir data/raw \\
        --output data/campaign/icos_fwi_v1/run_matrix.csv \\
        --per-site 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "data-ingestion"))
from ingest_icos import ICOS_STATIONS


def _beaufort_bin(ws: float) -> str:
    """Convert wind speed (m/s) to a Beaufort bin label."""
    if ws < 5.5:
        return "0-3"
    elif ws < 8.0:
        return "4"
    elif ws < 10.8:
        return "5"
    return "6+"


def _direction_octant(wd: float) -> int:
    width = 45.0
    return int(((wd + width / 2) % 360) // width)


def sample_station(
    zarr_path: Path, station_id: str, meta: dict, n_samples: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample stratified timestamps from an ICOS tall-tower Zarr."""
    root = zarr.open_group(str(zarr_path), mode="r")
    times = pd.to_datetime(np.asarray(root["coords/time"][:]).astype("datetime64[ns]"))

    # Get wind speed and direction at the highest available level
    meteo_keys = list(root["meteo"].keys()) if "meteo" in root else []
    ws_col = "ws"
    wd_col = "wd"
    if ws_col not in meteo_keys:
        print(f"  {station_id}: no 'ws' in meteo keys {meteo_keys}, skipping")
        return pd.DataFrame()

    ws = np.asarray(root["meteo/ws"][:], dtype=np.float32)
    wd = np.asarray(root["meteo/wd"][:], dtype=np.float32) if wd_col in meteo_keys else np.zeros_like(ws)

    df = pd.DataFrame({"time": times, "ws": ws, "wd": wd})
    df = df.dropna(subset=["ws"])
    df = df[df["ws"] > 0.5]  # Filter calm winds
    if df.empty:
        return pd.DataFrame()

    # Resample to daily (use noon value — one case per day max)
    df = df.set_index("time")
    df_daily = df.between_time("10:00", "14:00").resample("1D").mean().dropna()
    if df_daily.empty:
        return pd.DataFrame()

    df_daily["bin"] = df_daily["ws"].map(_beaufort_bin)
    df_daily["oct"] = df_daily["wd"].map(_direction_octant)

    # Beaufort stratified sampling — weight toward stronger winds
    weights = {"0-3": 0.10, "4": 0.30, "5": 0.30, "6+": 0.30}
    alloc = {b: max(1, round(n_samples * w)) for b, w in weights.items()}
    diff = n_samples - sum(alloc.values())
    if diff != 0:
        key = max(alloc, key=alloc.get)
        alloc[key] += diff

    picks = []
    for b, k in alloc.items():
        sub = df_daily[df_daily["bin"] == b]
        if sub.empty:
            continue
        n = min(len(sub), k)
        # Spread across direction octants
        chosen = (
            sub.groupby("oct", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, n // max(1, sub["oct"].nunique()))),
                                       random_state=int(rng.integers(2**31))))
        )
        if len(chosen) > k:
            chosen = chosen.sample(k, random_state=int(rng.integers(2**31)))
        elif len(chosen) < k and len(sub) >= k:
            extra = sub.drop(chosen.index).sample(k - len(chosen),
                                                   random_state=int(rng.integers(2**31)))
            chosen = pd.concat([chosen, extra])
        picks.append(chosen)

    if not picks:
        return pd.DataFrame()

    result = pd.concat(picks).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    if len(result) > n_samples:
        result = result.sample(n_samples, random_state=int(rng.integers(2**31)))

    return result.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Build ICOS campaign run_matrix.csv")
    ap.add_argument("--icos-dir", type=Path, default=Path("data/raw"),
                    help="Directory containing icos_*.zarr files")
    ap.add_argument("--output", type=Path,
                    default=Path("data/campaign/icos_fwi_v1/run_matrix.csv"))
    ap.add_argument("--per-site", type=int, default=30,
                    help="Timestamps per site (default: 30)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    run_id = 0

    for station_id, meta in ICOS_STATIONS.items():
        if meta.get("type") != "AS":
            continue  # Only atmosphere tall-towers for CFD validation
        zarr_name = f"icos_{station_id.replace('-', '_').lower()}.zarr"
        zarr_path = args.icos_dir / zarr_name
        if not zarr_path.exists():
            print(f"  {station_id}: {zarr_name} not found, skipping")
            continue

        print(f"  {station_id}: sampling from {zarr_name}...")
        picks = sample_station(zarr_path, station_id, meta, args.per_site, rng)
        if picks.empty:
            print(f"  {station_id}: no valid timestamps, skipping")
            continue

        for _, row in picks.iterrows():
            ts = row["time"]
            rows.append({
                "run_id": f"run_{run_id:06d}",
                "site_id": station_id,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts),
                "lat": meta["lat"],
                "lon": meta["lon"],
                "group": f"icos_{meta.get('type', 'AS')}",
                "priority": "high",
                "status": "pending",
            })
            run_id += 1

        print(f"  {station_id}: {len(picks)} timestamps selected "
              f"(ws range {picks['ws'].min():.1f}-{picks['ws'].max():.1f} m/s)")

    if not rows:
        print("ERROR: No runs generated. Check ICOS Zarr files in --icos-dir.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nRun matrix: {len(df)} cases across {df['site_id'].nunique()} sites")
    print(f"Saved to: {args.output}")
    print(f"\nPer-site breakdown:")
    print(df.groupby("site_id").size().to_string())


if __name__ == "__main__":
    main()
