"""Build the FuXi-CFD EU-towers pairings parquet from local ICOS obs Zarrs.

Sites = the ICOS tall towers used in the FuXi-CFD paper EU validation that we
hold obs for: OPE Houdelaincourt (10/50 m kept; 120 m dropped, above our 100 m
AGL grid) and IPR Ispra (40/60/100 m). One row per (site, height, hour) with a
valid wind speed+direction. u/v use the meteorological from-direction
convention: u = -ws*sin(wd), v = -ws*cos(wd).

Output columns match score_fuxi_vs_ours sample schema: station_id, timestamp,
lat, lon, elev, height_obs, speed_obs, u_obs, v_obs, season, pop ("icos_eu").

Usage (local):
  python build_icos_eu_pairings.py --raw-dir ../../data/raw \
    --output ../../data/inference/icos_eu_towers.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

SITES = {
    "OPE": {"zarr": "icos_ope.zarr", "lat": 48.5619, "lon": 5.5036,
            "elev": 395.0, "heights": [10, 50]},
    "IPR": {"zarr": "icos_ipr.zarr", "lat": 45.8126, "lon": 8.6360,
            "elev": 210.0, "heights": [40, 60, 100]},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for site, spec in SITES.items():
        g = zarr.open_group(str(args.raw_dir / spec["zarr"]), mode="r")
        times = np.array(g["coords/time"]).astype("datetime64[ns]")
        for h in spec["heights"]:
            ws = np.asarray(g[f"meteo/ws_{h}m"][:], dtype=np.float64)
            wd = np.asarray(g[f"meteo/wd_{h}m"][:], dtype=np.float64)
            ok = np.isfinite(ws) & np.isfinite(wd) & (ws > 0.0)
            wd_rad = np.deg2rad(wd[ok])
            df = pd.DataFrame({
                "station_id": f"{site}_h{h:03d}",
                "timestamp": times[ok],
                "lat": spec["lat"], "lon": spec["lon"], "elev": spec["elev"],
                "height_obs": float(h),
                "speed_obs": ws[ok],
                "u_obs": -ws[ok] * np.sin(wd_rad),
                "v_obs": -ws[ok] * np.cos(wd_rad),
                "season": "jja2020", "pop": "icos_eu",
            })
            rows.append(df)
            print(f"{site} {h:>3d} m: {int(ok.sum())}/{len(ws)} valid hours, "
                  f"mean ws {np.nanmean(ws[ok]):.2f} m/s")
    out = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} pairings, {out['station_id'].nunique()} series -> {args.output}")


if __name__ == "__main__":
    main()
