"""
build_icos_campaign.py — Build the ICOS validation CFD campaign.

Reads configs/icos_campaign.yaml, samples timestamps per site with
Beaufort × direction stratification against ERA5 hourly fields, and
emits a case manifest compatible with generate_campaign.py + kraken-sim.

Output:
    data/campaign/icos_fwi_v1/
        sites.csv                # resolved site list with metadata
        run_matrix.csv           # one row per (site, timestamp) case
        campaign.yaml            # kraken-sim manifest

Usage:
    python services/module2a-cfd/build_icos_campaign.py \\
        --config configs/icos_campaign.yaml \\
        --era5 data/raw/era5_europe_2020_2023.zarr \\
        --output data/campaign/icos_fwi_v1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger

log = get_logger("build_icos_campaign")


# Beaufort speed bin edges (m/s at 10 m)
BEAUFORT_EDGES = {"0-3": (0.0, 5.5), "4": (5.5, 8.0), "5": (8.0, 10.8), "6+": (10.8, 35.0)}


@dataclass
class Sample:
    site_id: str
    lat: float
    lon: float
    site_type: str
    timestamp: pd.Timestamp
    u10: float
    v10: float
    wspd: float
    wdir: float
    beaufort_bin: str
    dir_octant: int


def _beaufort_bin(ws: float) -> str:
    for name, (lo, hi) in BEAUFORT_EDGES.items():
        if lo <= ws < hi:
            return name
    return "6+"


def _direction_octant(wdir: float, n: int = 8) -> int:
    width = 360.0 / n
    return int(((wdir + width / 2) % 360) // width)


def _sample_site_timestamps(
    era5_ds,
    lat: float,
    lon: float,
    period: tuple[str, str],
    n_samples: int,
    beaufort_weights: dict[str, float],
    n_octants: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Extract ERA5 10 m wind at site and sample N stratified timestamps."""
    import xarray as xr

    # Assume ERA5 Zarr has u10, v10 with (time, lat, lon)
    da_u = era5_ds["u10"].sel(lat=lat, lon=lon, method="nearest")
    da_v = era5_ds["v10"].sel(lat=lat, lon=lon, method="nearest")
    da_u = da_u.sel(time=slice(*period))
    da_v = da_v.sel(time=slice(*period))

    times = pd.to_datetime(da_u.time.values)
    u = da_u.values.astype(np.float32)
    v = da_v.values.astype(np.float32)
    ws = np.sqrt(u * u + v * v)
    wd = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0

    df = pd.DataFrame(
        {"time": times, "u10": u, "v10": v, "wspd": ws, "wdir": wd}
    )
    df["bin"] = df["wspd"].map(_beaufort_bin)
    df["oct"] = df["wdir"].map(lambda x: _direction_octant(x, n_octants))

    # Stratified sampling: allocate n_samples across bins, then uniform over octants
    total_weight = sum(beaufort_weights.values())
    allocations = {
        b: max(1, int(round(n_samples * w / total_weight)))
        for b, w in beaufort_weights.items()
    }
    # Trim to hit exactly n_samples
    diff = n_samples - sum(allocations.values())
    if diff != 0:
        key = max(allocations, key=allocations.get)
        allocations[key] += diff

    picks: list[pd.DataFrame] = []
    for b, k in allocations.items():
        sub = df[df["bin"] == b]
        if sub.empty:
            continue
        # Uniform coverage across octants present in this bin
        per_oct = max(1, k // max(1, sub["oct"].nunique()))
        chosen = (
            sub.groupby("oct", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_oct), random_state=int(rng.integers(2**31))))
        )
        if len(chosen) > k:
            chosen = chosen.sample(k, random_state=int(rng.integers(2**31)))
        elif len(chosen) < k and len(sub) >= k:
            extra = sub.drop(chosen.index).sample(k - len(chosen), random_state=int(rng.integers(2**31)))
            chosen = pd.concat([chosen, extra])
        picks.append(chosen)

    if not picks:
        return pd.DataFrame()
    return pd.concat(picks, ignore_index=True).sort_values("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/icos_campaign.yaml")
    ap.add_argument("--era5", required=True, help="ERA5 hourly Zarr covering all sites")
    ap.add_argument("--output", default="data/campaign/icos_fwi_v1")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["sampling"]["random_seed"])

    import xarray as xr
    log.info("Opening ERA5", extra={"path": args.era5})
    era5 = xr.open_zarr(args.era5, consolidated=False)

    sites = cfg["sites"]
    period = (cfg["sampling"]["period_start"], cfg["sampling"]["period_end"])
    n_per_site = cfg["sampling"]["timestamps_per_site"]
    bf_weights = cfg["sampling"]["beaufort_weights"]
    n_oct = cfg["sampling"]["direction_octants"]

    all_cases: list[dict] = []
    for s in sites:
        log.info("Sampling site", extra={"site": s["id"], "lat": s["lat"], "lon": s["lon"]})
        picks = _sample_site_timestamps(
            era5, s["lat"], s["lon"], period, n_per_site, bf_weights, n_oct, rng
        )
        if picks.empty:
            log.warning("No samples for site", extra={"site": s["id"]})
            continue
        for i, row in picks.iterrows():
            all_cases.append(
                {
                    "case_id": f"{s['id']}_{row['time'].strftime('%Y%m%d_%H%M')}",
                    "site_id": s["id"],
                    "site_type": s["type"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "timestamp": row["time"],
                    "u10": float(row["u10"]),
                    "v10": float(row["v10"]),
                    "wspd": float(row["wspd"]),
                    "wdir": float(row["wdir"]),
                    "beaufort_bin": row["bin"],
                    "dir_octant": int(row["oct"]),
                }
            )

    run_matrix = pd.DataFrame(all_cases)
    run_matrix_path = out / "run_matrix.csv"
    run_matrix.to_csv(run_matrix_path, index=False)
    log.info("Run matrix written", extra={"n_cases": len(run_matrix), "path": str(run_matrix_path)})

    sites_df = pd.DataFrame(sites).rename(columns={"id": "site_id"})
    sites_df.to_csv(out / "sites.csv", index=False)

    # kraken-sim campaign manifest
    manifest = {
        "campaign": cfg["campaign_name"],
        "solver": cfg["solver"],
        "stability": cfg["stability"],
        "domain": cfg["domain"],
        "solver_control": cfg["solver_control"],
        "land_surface": cfg["land_surface"],
        "inflow": cfg["inflow"],
        "cases": [
            {
                "case_id": c["case_id"],
                "site_id": c["site_id"],
                "lat": c["lat"],
                "lon": c["lon"],
                "timestamp": c["timestamp"].isoformat(),
                "inflow_u10": c["u10"],
                "inflow_v10": c["v10"],
            }
            for c in all_cases
        ],
    }
    (out / "campaign.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    log.info("Campaign manifest written", extra={"path": str(out / "campaign.yaml")})

    # Summary
    print("\n=== ICOS Campaign Summary ===")
    print(f"  Sites:  {len(sites_df)}")
    print(f"  Cases:  {len(run_matrix)}")
    print(f"  Beaufort distribution:")
    print(run_matrix["beaufort_bin"].value_counts().to_string())
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
