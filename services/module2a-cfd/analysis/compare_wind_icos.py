"""
compare_wind_icos.py — Compare CFD wind at tower heights vs ICOS observations.

3-way comparison: ICOS obs / ERA5 / CFD at matching timestamps and heights.
Produces per-site metrics (MAE, RMSE, bias) and summary table matching
FuXi-CFD Fig. 5 format (Lin et al. 2026).

Usage:
    python services/module2a-cfd/analysis/compare_wind_icos.py \
        --tower-profiles data/campaign/icos_fwi_v1/tower_profiles.csv \
        --run-matrix data/campaign/icos_fwi_v1/run_matrix.csv \
        --icos-dir data/raw \
        --era5-dir data/raw \
        --output data/validation/wind_comparison
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_icos_obs(icos_dir: Path, site_id: str) -> pd.DataFrame:
    """Load ICOS tower observations as hourly DataFrame."""
    import zarr
    zarr_name = f"icos_{site_id.lower()}.zarr"
    zarr_path = icos_dir / zarr_name
    if not zarr_path.exists():
        return pd.DataFrame()

    root = zarr.open_group(str(zarr_path), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    meteo_keys = list(root["meteo"].keys())

    df = pd.DataFrame({"time": times})
    # Load all wind columns (ws_10m, ws_120m, etc.)
    for key in meteo_keys:
        if key.startswith(("ws_", "wd_", "T_", "RH_")) or key in ("ws", "wd", "T", "RH", "p"):
            df[key] = np.array(root[f"meteo/{key}"][:], dtype=np.float32)
    return df


def load_era5_wind(era5_dir: Path, site_id: str, lat: float, lon: float) -> pd.DataFrame:
    """Load ERA5 10m wind at site location."""
    import zarr
    zarr_path = era5_dir / f"era5_{site_id.lower()}.zarr"
    if not zarr_path.exists():
        # Try v2 format
        zarr_path = era5_dir / f"era5_{site_id.lower()}_v2.zarr"
    if not zarr_path.exists():
        return pd.DataFrame()

    root = zarr.open_group(str(zarr_path), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    # ERA5 surface: u10, v10 — pick nearest grid point (centre of 3x3)
    u10 = np.array(root["surface/u10"][:])
    v10 = np.array(root["surface/v10"][:])
    # Centre cell of 3x3 grid = index [time, 1, 1]
    if u10.ndim == 3:
        u10 = u10[:, 1, 1]
        v10 = v10[:, 1, 1]
    elif u10.ndim == 2:
        u10 = u10[:, u10.shape[1] // 2]
        v10 = v10[:, v10.shape[1] // 2]

    ws = np.sqrt(u10**2 + v10**2)
    wd = (270 - np.degrees(np.arctan2(v10, u10))) % 360
    return pd.DataFrame({"time": times, "ws_era5": ws, "wd_era5": wd,
                          "u10_era5": u10, "v10_era5": v10})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower-profiles", type=Path, required=True)
    ap.add_argument("--run-matrix", type=Path, required=True)
    ap.add_argument("--icos-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--era5-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, default=Path("data/validation/wind_comparison"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load CFD tower profiles + map case_id → timestamp
    profiles = pd.read_csv(args.tower_profiles)
    run_matrix = pd.read_csv(args.run_matrix, parse_dates=["timestamp"])

    # Build case_id → timestamp mapping per site
    ts_map = {}
    for _, row in run_matrix.iterrows():
        sid = row["site_id"]
        ts_map.setdefault(sid, []).append(row["timestamp"])

    # Add timestamp to profiles
    profiles["timestamp"] = pd.NaT
    for sid in profiles["site_id"].unique():
        site_ts = ts_map.get(sid, [])
        mask = profiles["site_id"] == sid
        case_ids = profiles.loc[mask, "case_id"].unique()
        for i, cid in enumerate(sorted(case_ids)):
            if i < len(site_ts):
                profiles.loc[mask & (profiles["case_id"] == cid), "timestamp"] = site_ts[i]

    summary_rows = []

    for site_id in sorted(profiles["site_id"].unique()):
        site_profiles = profiles[profiles["site_id"] == site_id].copy()
        site_rm = run_matrix[run_matrix["site_id"] == site_id]
        if site_rm.empty:
            continue
        lat = site_rm.iloc[0]["lat"]
        lon = site_rm.iloc[0]["lon"]

        # Load ICOS observations
        icos_df = load_icos_obs(args.icos_dir, site_id)
        if icos_df.empty:
            print(f"  {site_id}: no ICOS data, skipping")
            continue

        # Load ERA5
        era5_df = load_era5_wind(args.era5_dir, site_id, lat, lon)

        # Available ICOS wind heights
        ws_cols = [c for c in icos_df.columns if c.startswith("ws_") and c.endswith("m")]
        heights_icos = {int(c.replace("ws_", "").replace("m", "")): c for c in ws_cols}

        print(f"\n=== {site_id} ===")
        print(f"  ICOS heights: {sorted(heights_icos.keys())} m")
        print(f"  CFD heights:  {sorted(site_profiles['z_agl_m'].unique()[:10])} m ...")
        print(f"  Cases: {site_profiles['case_id'].nunique()}")

        for h_icos, ws_col in sorted(heights_icos.items()):
            wd_col = f"wd_{h_icos}m"
            # Find closest CFD height
            cfd_heights = site_profiles["z_agl_m"].unique()
            h_cfd = cfd_heights[np.argmin(np.abs(cfd_heights - h_icos))]
            if abs(h_cfd - h_icos) > 20:
                continue  # Too far apart

            cfd_at_h = site_profiles[site_profiles["z_agl_m"] == h_cfd].copy()

            # Match on timestamp (nearest hour)
            pairs = []
            for _, cfd_row in cfd_at_h.iterrows():
                ts = cfd_row["timestamp"]
                if pd.isna(ts):
                    continue
                # Find ICOS obs at this timestamp
                ts_match = icos_df.iloc[(icos_df["time"] - ts).abs().argsort()[:1]]
                if ts_match.empty:
                    continue
                obs_ws = float(ts_match[ws_col].iloc[0])
                obs_wd = float(ts_match[wd_col].iloc[0]) if wd_col in ts_match.columns else np.nan
                if np.isnan(obs_ws) or obs_ws < 0:
                    continue

                # ERA5 at this timestamp
                era5_ws = np.nan
                if not era5_df.empty:
                    era5_match = era5_df.iloc[(era5_df["time"] - ts).abs().argsort()[:1]]
                    era5_ws = float(era5_match["ws_era5"].iloc[0])

                pairs.append({
                    "site_id": site_id,
                    "timestamp": ts,
                    "height_m": h_icos,
                    "ws_obs": obs_ws,
                    "ws_cfd": float(cfd_row["ws"]),
                    "ws_era5": era5_ws,
                    "wd_obs": obs_wd,
                    "wd_cfd": float(cfd_row["wd"]),
                })

            if not pairs:
                continue
            pdf = pd.DataFrame(pairs)

            # Metrics
            for src, col in [("CFD", "ws_cfd"), ("ERA5", "ws_era5")]:
                valid = pdf[["ws_obs", col]].dropna()
                if len(valid) < 3:
                    continue
                err = valid[col] - valid["ws_obs"]
                summary_rows.append({
                    "site_id": site_id,
                    "height_m": h_icos,
                    "source": src,
                    "n": len(valid),
                    "mae": float(err.abs().mean()),
                    "rmse": float(np.sqrt((err**2).mean())),
                    "bias": float(err.mean()),
                    "corr": float(valid.corr().iloc[0, 1]) if len(valid) > 2 else np.nan,
                })

            print(f"  h={h_icos}m (CFD @{h_cfd:.0f}m): {len(pdf)} pairs")

        # Save per-site detail
        site_out = args.output / f"{site_id}_wind.csv"
        site_profiles.to_csv(site_out, index=False)

    # Summary
    if not summary_rows:
        print("\nNo valid comparisons found!")
        sys.exit(1)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    # Pretty print
    print("\n" + "=" * 70)
    print("WIND SPEED COMPARISON: MAE (m/s)")
    print("=" * 70)
    pivot = summary.pivot_table(
        index=["site_id", "height_m"], columns="source", values="mae"
    ).round(2)
    print(pivot.to_string())
    print()

    # Global means
    for src in ["ERA5", "CFD"]:
        sub = summary[summary["source"] == src]
        if sub.empty:
            continue
        print(f"  Mean MAE {src}: {sub['mae'].mean():.2f} m/s")
        print(f"  Mean RMSE {src}: {sub['rmse'].mean():.2f} m/s")
        print(f"  Mean bias {src}: {sub['bias'].mean():.2f} m/s")
    print()


if __name__ == "__main__":
    main()
