"""
compare_fwi_icos.py — 3-way FWI comparison at ICOS tower sites.

    FWI_obs  = f(ICOS T, RH, ws)  + ERA5-Land rain (no obs rain at atm towers)
    FWI_ERA5 = f(ERA5 T2m, RH, ws10) + ERA5-Land rain
    FWI_CFD  = f(CFD T, q→RH, ws @10m) + ERA5-Land rain

Same rain source for all → isolates the wind+T+RH contribution of CFD.

Usage:
    python services/module2a-cfd/analysis/compare_fwi_icos.py \
        --tower-profiles data/campaign/icos_fwi_v1/tower_profiles.csv \
        --run-matrix data/campaign/icos_fwi_v1/run_matrix.csv \
        --icos-dir data/raw \
        --era5-dir data/raw \
        --output data/validation/fwi_comparison
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.fwi import compute_fwi_series


# ── Helpers ──────────────────────────────────────────────────────────────────

def _q2rh(q: np.ndarray, t_c: np.ndarray, p_hpa: float = 1000.0) -> np.ndarray:
    """Specific humidity (kg/kg) + T (°C) → RH (%)."""
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return np.clip(100.0 * e / es, 0, 100).astype(np.float32)


def _td2rh(t2m_K, d2m_K):
    a, b = 17.625, 243.04
    tc = t2m_K - 273.15
    tdc = d2m_K - 273.15
    return np.clip(100 * np.exp(a * tdc / (b + tdc)) / np.exp(a * tc / (b + tc)), 0, 100)


def _noon_daily(df, time_col="time"):
    """Compute daily noon mean (11-13 UTC) for T, RH, ws and 24h rain sum."""
    df = df.set_index(time_col).sort_index()
    noon = df[df.index.hour.isin([11, 12, 13])][["T", "RH", "ws"]].resample("1D").mean()
    rain = df["rain_mm"].resample("1D").sum() if "rain_mm" in df.columns else pd.Series(0, index=noon.index)
    out = noon.dropna()
    out["rain_mm"] = rain.reindex(out.index).fillna(0)
    return out.reset_index().rename(columns={time_col: "date"})


def _run_fwi(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or len(daily) < 3:
        return daily
    t = daily["T"].to_numpy(dtype=np.float64)
    rh = np.clip(daily["RH"].to_numpy(dtype=np.float64), 0, 100)
    ws = daily["ws"].to_numpy(dtype=np.float64) * 3.6  # m/s → km/h
    rain = daily["rain_mm"].to_numpy(dtype=np.float64)
    months = pd.to_datetime(daily["date"]).dt.month.to_numpy(dtype=np.int32)
    res = compute_fwi_series(t, rh, ws, rain, months)
    daily = daily.copy()
    for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
        daily[k.upper()] = np.round(res[k], 2)
    return daily


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_icos_hourly(icos_dir, site_id):
    import zarr
    p = icos_dir / f"icos_{site_id.lower()}.zarr"
    if not p.exists():
        return pd.DataFrame()
    root = zarr.open_group(str(p), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    df = pd.DataFrame({"time": times})
    keys = list(root["meteo"].keys())
    # Use lowest-height T and RH, highest-height ws
    for k in keys:
        df[k] = np.array(root[f"meteo/{k}"][:], dtype=np.float32)
    # Ensure primary aliases
    if "T" not in df.columns:
        t_cols = sorted([c for c in df.columns if c.startswith("T_") and c.endswith("m")])
        if t_cols:
            df["T"] = df[t_cols[0]]
    if "RH" not in df.columns:
        rh_cols = sorted([c for c in df.columns if c.startswith("RH_") and c.endswith("m")])
        if rh_cols:
            df["RH"] = df[rh_cols[0]]
    if "ws" not in df.columns:
        ws_cols = sorted([c for c in df.columns if c.startswith("ws_") and c.endswith("m")])
        if ws_cols:
            df["ws"] = df[ws_cols[-1]]  # highest
    df["rain_mm"] = 0.0  # ICOS atm towers have no precip
    return df


def load_era5_hourly(era5_dir, site_id):
    import zarr
    for suffix in ["", "_v2"]:
        p = era5_dir / f"era5_{site_id.lower()}{suffix}.zarr"
        if p.exists():
            break
    else:
        return pd.DataFrame()
    root = zarr.open_group(str(p), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    # Centre cell of 3×3
    def _mid(arr):
        if arr.ndim == 3:
            return arr[:, 1, 1]
        elif arr.ndim == 2:
            return arr[:, arr.shape[1] // 2]
        return arr
    u10 = _mid(np.array(root["surface/u10"][:]))
    v10 = _mid(np.array(root["surface/v10"][:]))
    t2m = _mid(np.array(root["surface/t2m"][:]))
    d2m = _mid(np.array(root["surface/d2m"][:])) if "d2m" in root["surface"] else None
    ws = np.sqrt(u10**2 + v10**2)
    rh = _td2rh(t2m, d2m) if d2m is not None else np.full_like(t2m, 50)
    df = pd.DataFrame({
        "time": times, "T": t2m - 273.15, "RH": rh, "ws": ws, "rain_mm": 0.0
    })
    # TODO: add ERA5-Land precip if available
    return df


def build_cfd_daily(profiles, run_matrix, site_id):
    """Build daily noon values from CFD profiles at ~10m height."""
    site_rm = run_matrix[run_matrix["site_id"] == site_id].sort_values("timestamp")
    site_prof = profiles[profiles["site_id"] == site_id]

    # Map case_id → timestamp
    case_ids = sorted(site_prof["case_id"].unique())
    ts_list = site_rm["timestamp"].tolist()

    rows = []
    for i, cid in enumerate(case_ids):
        if i >= len(ts_list):
            break
        ts = ts_list[i]
        cp = site_prof[site_prof["case_id"] == cid]
        # Pick height closest to 10m
        h_avail = cp["z_agl_m"].unique()
        h_10 = h_avail[np.argmin(np.abs(h_avail - 10))]
        row = cp[cp["z_agl_m"] == h_10].iloc[0]
        t_celsius = float(row["T"]) - 273.15 if float(row["T"]) > 100 else float(row["T"])
        rh = float(_q2rh(np.array([row["q"]]), np.array([t_celsius]))[0])
        rows.append({
            "date": ts,
            "T": t_celsius,
            "RH": rh,
            "ws": float(row["ws"]),
            "rain_mm": 0.0,
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower-profiles", type=Path, required=True)
    ap.add_argument("--run-matrix", type=Path, required=True)
    ap.add_argument("--icos-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--era5-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, default=Path("data/validation/fwi_comparison"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(args.tower_profiles)
    run_matrix = pd.read_csv(args.run_matrix, parse_dates=["timestamp"])
    sites = sorted(profiles["site_id"].unique())

    all_metrics = []

    for site_id in sites:
        print(f"\n{'='*60}")
        print(f"  {site_id}")
        print(f"{'='*60}")

        # 1) ICOS obs → daily noon → FWI
        icos_h = load_icos_hourly(args.icos_dir, site_id)
        if icos_h.empty or "T" not in icos_h.columns or "RH" not in icos_h.columns:
            print(f"  Skip: no ICOS T/RH")
            continue
        icos_d = _noon_daily(icos_h)
        icos_fwi = _run_fwi(icos_d)
        if "FWI" not in icos_fwi.columns:
            print(f"  Skip: FWI computation failed (not enough ICOS data)")
            continue
        print(f"  ICOS: {len(icos_fwi)} days, FWI mean={icos_fwi['FWI'].mean():.1f}")

        # 2) ERA5 → daily noon → FWI
        era5_h = load_era5_hourly(args.era5_dir, site_id)
        era5_fwi = pd.DataFrame()
        if not era5_h.empty:
            era5_d = _noon_daily(era5_h)
            era5_fwi = _run_fwi(era5_d)
            print(f"  ERA5: {len(era5_fwi)} days, FWI mean={era5_fwi['FWI'].mean():.1f}")

        # 3) CFD → daily (one case = one day) → FWI
        cfd_d = build_cfd_daily(profiles, run_matrix, site_id)
        if cfd_d.empty:
            print(f"  Skip: no CFD profiles")
            continue
        cfd_fwi = _run_fwi(cfd_d)
        print(f"  CFD:  {len(cfd_fwi)} days, FWI mean={cfd_fwi['FWI'].mean():.1f}")

        # Align on date
        icos_fwi["date"] = pd.to_datetime(icos_fwi["date"]).dt.date
        cfd_fwi["date"] = pd.to_datetime(cfd_fwi["date"]).dt.date
        if not era5_fwi.empty:
            era5_fwi["date"] = pd.to_datetime(era5_fwi["date"]).dt.date

        merged = (
            cfd_fwi[["date", "T", "RH", "ws", "FWI", "ISI", "FFMC"]]
            .rename(columns={"T": "T_cfd", "RH": "RH_cfd", "ws": "ws_cfd",
                             "FWI": "FWI_cfd", "ISI": "ISI_cfd", "FFMC": "FFMC_cfd"})
        )
        obs_match = icos_fwi[["date", "T", "RH", "ws", "FWI", "ISI", "FFMC"]].rename(
            columns={"T": "T_obs", "RH": "RH_obs", "ws": "ws_obs",
                     "FWI": "FWI_obs", "ISI": "ISI_obs", "FFMC": "FFMC_obs"})
        merged = merged.merge(obs_match, on="date", how="inner")
        if not era5_fwi.empty:
            era5_match = era5_fwi[["date", "FWI", "ISI", "FFMC"]].rename(
                columns={"FWI": "FWI_era5", "ISI": "ISI_era5", "FFMC": "FFMC_era5"})
            merged = merged.merge(era5_match, on="date", how="left")

        merged["site_id"] = site_id
        merged.to_csv(args.output / f"{site_id}_fwi.csv", index=False)

        # Metrics
        for src, suffix in [("CFD", "_cfd"), ("ERA5", "_era5")]:
            fwi_col = f"FWI{suffix}"
            if fwi_col not in merged.columns:
                continue
            valid = merged[["FWI_obs", fwi_col]].dropna()
            if len(valid) < 3:
                continue
            err = valid[fwi_col] - valid["FWI_obs"]
            all_metrics.append({
                "site_id": site_id, "source": src, "n": len(valid),
                "mae_fwi": float(err.abs().mean()),
                "rmse_fwi": float(np.sqrt((err**2).mean())),
                "bias_fwi": float(err.mean()),
            })
            # Also ISI (wind-driven component)
            isi_col = f"ISI{suffix}"
            if isi_col in merged.columns:
                v2 = merged[["ISI_obs", isi_col]].dropna()
                if len(v2) >= 3:
                    e2 = v2[isi_col] - v2["ISI_obs"]
                    all_metrics.append({
                        "site_id": site_id, "source": src + "_ISI", "n": len(v2),
                        "mae_fwi": float(e2.abs().mean()),
                        "rmse_fwi": float(np.sqrt((e2**2).mean())),
                        "bias_fwi": float(e2.mean()),
                    })

        n = len(merged)
        if n > 0:
            print(f"  Matched: {n} days")
            for src in ["CFD", "ERA5"]:
                col = f"FWI_{src.lower()}"
                if col in merged.columns:
                    err = (merged[col] - merged["FWI_obs"]).dropna()
                    print(f"    FWI {src}: MAE={err.abs().mean():.2f}, bias={err.mean():.2f}")

    # Summary
    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        mdf.to_csv(args.output / "fwi_metrics.csv", index=False)
        print(f"\n{'='*60}")
        print("FWI COMPARISON SUMMARY")
        print(f"{'='*60}")
        fwi_only = mdf[~mdf["source"].str.contains("ISI")]
        for src in ["ERA5", "CFD"]:
            sub = fwi_only[fwi_only["source"] == src]
            if sub.empty:
                continue
            print(f"  {src}: MAE={sub['mae_fwi'].mean():.2f}, RMSE={sub['rmse_fwi'].mean():.2f}, bias={sub['bias_fwi'].mean():.2f}")
        print()
        isi_only = mdf[mdf["source"].str.contains("ISI")]
        if not isi_only.empty:
            print("ISI (wind-driven fire spread):")
            for src in ["ERA5_ISI", "CFD_ISI"]:
                sub = isi_only[isi_only["source"] == src]
                if sub.empty:
                    continue
                print(f"  {src}: MAE={sub['mae_fwi'].mean():.2f}, RMSE={sub['rmse_fwi'].mean():.2f}")


if __name__ == "__main__":
    main()
