"""
compare_fwi_hybrid.py — Hybrid FWI comparison with IMERG QM-corrected precipitation.

3-way comparison using CONTINUOUS daily series:
    FWI_obs   = f(ICOS T, RH, ws)  + zero rain (no obs rain at atm towers)
    FWI_ERA5  = f(ERA5 T2m, RH, ws10) + ERA5 tp (total precipitation)
    FWI_hybrid = f(ERA5 T, RH, ws_CFD_substituted) + IMERG QM-corrected rain

The hybrid approach:
    - Uses ERA5 as backbone for temporal continuity (daily series)
    - Substitutes ERA5 wind with CFD wind at timestamps where CFD is available
    - Uses QM-corrected IMERG rain instead of ERA5-Land rain (reduces drizzle bias)

This isolates the combined benefit of better wind (CFD) + better rain (IMERG QM).

Usage:
    python services/module2a-cfd/analysis/compare_fwi_hybrid.py \
        --tower-profiles data/campaign/icos_fwi_v1/tower_profiles.csv \
        --run-matrix data/campaign/icos_fwi_v1/run_matrix.csv \
        --icos-dir data/raw \
        --era5-dir data/raw \
        --imerg-csv data/raw/imerg_icos_sites.csv \
        --qm-model data/models/precip_correction/qm_stratified.npz \
        --output data/validation/fwi_hybrid
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "module3-precip"))
from shared.fwi import compute_fwi_series
from src.quantile_correction import StratifiedQMCorrector


# ── Helpers ──────────────────────────────────────────────────────────────────

ICOS_META = {
    "OPE": {"lat": 48.5619, "lon": 5.5036, "alt": 395},
    "IPR": {"lat": 45.8126, "lon": 8.6360, "alt": 210},
    "HPB": {"lat": 47.8011, "lon": 11.0246, "alt": 934},
    "PUY": {"lat": 45.7722, "lon": 2.9658, "alt": 1465},
    "SAC": {"lat": 48.7227, "lon": 2.1420, "alt": 160},
    "TRN": {"lat": 47.9647, "lon": 2.1125, "alt": 131},
}


def _q2rh(q, t_c, p_hpa=1000.0):
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return np.clip(100.0 * e / es, 0, 100)


def _td2rh(t2m_K, d2m_K):
    a, b = 17.625, 243.04
    tc, tdc = t2m_K - 273.15, d2m_K - 273.15
    return np.clip(100 * np.exp(a * tdc / (b + tdc)) / np.exp(a * tc / (b + tc)), 0, 100)


def _noon(df, time_col="time"):
    """Daily noon mean (11-13 UTC)."""
    df = df.set_index(time_col).sort_index()
    cols = [c for c in ["T", "RH", "ws"] if c in df.columns]
    noon = df[df.index.hour.isin([11, 12, 13])][cols].resample("1D").mean()
    if "rain_mm" in df.columns:
        rain = df["rain_mm"].resample("1D").sum()
    else:
        rain = pd.Series(0, index=noon.index)
    out = noon.dropna().copy()
    out["rain_mm"] = rain.reindex(out.index).fillna(0)
    return out.reset_index().rename(columns={time_col: "date"})


def _fwi(daily):
    if daily.empty or len(daily) < 5:
        return daily
    t = daily["T"].to_numpy(np.float64)
    rh = np.clip(daily["RH"].to_numpy(np.float64), 0, 100)
    ws = daily["ws"].to_numpy(np.float64) * 3.6
    rain = daily["rain_mm"].to_numpy(np.float64)
    months = pd.to_datetime(daily["date"]).dt.month.to_numpy(np.int32)
    res = compute_fwi_series(t, rh, ws, rain, months)
    out = daily.copy()
    for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
        out[k.upper()] = np.round(res[k], 2)
    return out


def load_icos_hourly(icos_dir, site_id):
    import zarr
    p = icos_dir / f"icos_{site_id.lower()}.zarr"
    if not p.exists():
        return pd.DataFrame()
    root = zarr.open_group(str(p), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    df = pd.DataFrame({"time": times})
    keys = list(root["meteo"].keys())
    for k in keys:
        df[k] = np.array(root[f"meteo/{k}"][:], dtype=np.float32)
    # Primary aliases
    if "T" not in df.columns:
        t_cols = sorted([c for c in df.columns if c.startswith("T_") and c.endswith("m")])
        if t_cols: df["T"] = df[t_cols[0]]
    if "RH" not in df.columns:
        rh_cols = sorted([c for c in df.columns if c.startswith("RH_") and c.endswith("m")])
        if rh_cols: df["RH"] = df[rh_cols[0]]
    if "ws" not in df.columns:
        ws_cols = sorted([c for c in df.columns if c.startswith("ws_") and c.endswith("m")])
        if ws_cols: df["ws"] = df[ws_cols[-1]]
    df["rain_mm"] = 0.0
    return df


def load_era5_hourly(era5_dir, site_id):
    import zarr
    for suffix in ["", "_v2"]:
        p = era5_dir / f"era5_{site_id.lower()}{suffix}.zarr"
        if p.exists(): break
    else:
        return pd.DataFrame()
    root = zarr.open_group(str(p), mode="r")
    times = pd.to_datetime(np.array(root["coords/time"][:]).astype("datetime64[ns]"))
    def _mid(a):
        if a.ndim == 3: return a[:, 1, 1]
        elif a.ndim == 2: return a[:, a.shape[1]//2]
        return a
    u10 = _mid(np.array(root["surface/u10"][:]))
    v10 = _mid(np.array(root["surface/v10"][:]))
    t2m = _mid(np.array(root["surface/t2m"][:]))
    d2m = _mid(np.array(root["surface/d2m"][:])) if "d2m" in root["surface"] else None
    ws = np.sqrt(u10**2 + v10**2)
    rh = _td2rh(t2m, d2m) if d2m is not None else np.full_like(t2m, 50)
    # ERA5 total precip (m/timestep → mm/timestep)
    tp = np.zeros_like(ws)
    if "tp" in root["surface"]:
        tp = _mid(np.array(root["surface/tp"][:]))
        tp = np.maximum(tp * 1000, 0)  # m → mm, clip negatives
    return pd.DataFrame({"time": times, "T": t2m-273.15, "RH": rh, "ws": ws, "rain_mm": tp})


def build_cfd_substituted(era5_daily, profiles, run_matrix, site_id):
    """Build hybrid daily: ERA5 backbone + CFD wind at matched timestamps."""
    site_rm = run_matrix[run_matrix["site_id"] == site_id].sort_values("timestamp")
    site_prof = profiles[profiles["site_id"] == site_id]

    hybrid = era5_daily.copy()
    hybrid["date"] = pd.to_datetime(hybrid["date"])
    hybrid["source"] = "ERA5"

    case_ids = sorted(site_prof["case_id"].unique())
    ts_list = site_rm["timestamp"].tolist()

    n_sub = 0
    for i, cid in enumerate(case_ids):
        if i >= len(ts_list): break
        ts = pd.Timestamp(ts_list[i])
        target_date = ts.normalize()

        cp = site_prof[site_prof["case_id"] == cid]
        h_avail = cp["z_agl_m"].unique()
        h_10 = h_avail[np.argmin(np.abs(h_avail - 10))]
        row = cp[cp["z_agl_m"] == h_10].iloc[0]

        t_c = float(row["T"]) - 273.15 if float(row["T"]) > 200 else float(row["T"])
        rh = float(_q2rh(np.array([row["q"]]), np.array([t_c]))[0])

        mask = hybrid["date"].dt.date == target_date.date()
        if mask.any():
            hybrid.loc[mask, "ws"] = float(row["ws"])
            hybrid.loc[mask, "T"] = t_c
            hybrid.loc[mask, "RH"] = rh
            hybrid.loc[mask, "source"] = "CFD"
            n_sub += 1

    print(f"    Substituted {n_sub}/{len(case_ids)} timestamps with CFD wind+T+RH")
    return hybrid


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower-profiles", type=Path, required=True)
    ap.add_argument("--run-matrix", type=Path, required=True)
    ap.add_argument("--icos-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--era5-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--imerg-csv", type=Path, default=None)
    ap.add_argument("--qm-model", type=Path, default=Path("data/models/precip_correction/qm_stratified.npz"))
    ap.add_argument("--output", type=Path, default=Path("data/validation/fwi_hybrid"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_csv(args.tower_profiles)
    run_matrix = pd.read_csv(args.run_matrix, parse_dates=["timestamp"])

    # Load QM corrector
    qm = None
    if args.qm_model.exists():
        qm = StratifiedQMCorrector.load(args.qm_model)
        print(f"QM model loaded: {args.qm_model}")

    # Load IMERG
    imerg_df = None
    if args.imerg_csv and args.imerg_csv.exists():
        imerg_df = pd.read_csv(args.imerg_csv, parse_dates=["date"])
        print(f"IMERG loaded: {len(imerg_df)} records")

    all_metrics = []

    for site_id in sorted(profiles["site_id"].unique()):
        meta = ICOS_META.get(site_id)
        if not meta:
            continue

        print(f"\n{'='*60}")
        print(f"  {site_id} ({meta['lat']:.2f}°N, {meta['lon']:.2f}°E, {meta['alt']}m)")
        print(f"{'='*60}")

        # Prepare IMERG QM rain for this site (same for all 3 FWI variants)
        site_rain = pd.DataFrame()
        if imerg_df is not None and qm is not None:
            si = imerg_df[imerg_df["site_id"] == site_id].copy()
            if not si.empty:
                si["date"] = pd.to_datetime(si["date"])
                months = si["date"].dt.month.to_numpy()
                rain_corr = qm.predict(si["rain_imerg"].to_numpy(), months,
                                       meta["alt"], meta["lat"], meta["lon"])
                site_rain = si[["date"]].copy()
                site_rain["rain_qm"] = rain_corr

        # 1) ICOS obs → continuous FWI (with IMERG QM rain — same source for all)
        icos_h = load_icos_hourly(args.icos_dir, site_id)
        if icos_h.empty or "T" not in icos_h or "RH" not in icos_h:
            print("  Skip: no ICOS T/RH")
            continue
        icos_d = _noon(icos_h)
        # Inject IMERG QM rain into obs daily
        if not site_rain.empty:
            icos_d["date"] = pd.to_datetime(icos_d["date"])
            icos_d = icos_d.merge(site_rain, on="date", how="left")
            icos_d["rain_mm"] = icos_d["rain_qm"].fillna(0)
            icos_d = icos_d.drop(columns=["rain_qm"])
        icos_fwi = _fwi(icos_d)
        if "FWI" not in icos_fwi.columns:
            print("  Skip: FWI failed (too few days)")
            continue
        print(f"  ICOS: {len(icos_fwi)} days, FWI mean={icos_fwi['FWI'].mean():.1f}")

        # 2) ERA5 → continuous FWI (with ERA5 precip)
        era5_h = load_era5_hourly(args.era5_dir, site_id)
        if era5_h.empty:
            print("  Skip: no ERA5")
            continue
        era5_d = _noon(era5_h)
        # Inject same IMERG QM rain
        if not site_rain.empty:
            era5_d["date"] = pd.to_datetime(era5_d["date"])
            era5_d = era5_d.merge(site_rain, on="date", how="left")
            era5_d["rain_mm"] = era5_d["rain_qm"].fillna(0)
            era5_d = era5_d.drop(columns=["rain_qm"])
        era5_fwi = _fwi(era5_d)
        if "FWI" not in era5_fwi.columns:
            print("  Skip: ERA5 FWI failed")
            continue
        print(f"  ERA5: {len(era5_fwi)} days, FWI mean={era5_fwi['FWI'].mean():.1f}")

        # 3) Hybrid: ERA5 backbone + CFD wind substitution + IMERG QM rain
        hybrid_d = build_cfd_substituted(era5_d, profiles, run_matrix, site_id)

        # Apply IMERG QM-corrected rain if available
        if imerg_df is not None and qm is not None:
            site_imerg = imerg_df[imerg_df["site_id"] == site_id].copy()
            if not site_imerg.empty:
                site_imerg["date"] = pd.to_datetime(site_imerg["date"])
                months = site_imerg["date"].dt.month.to_numpy()
                rain_raw = site_imerg["rain_imerg"].to_numpy()
                rain_corrected = qm.predict(rain_raw, months, meta["alt"], meta["lat"], meta["lon"])
                site_imerg["rain_qm"] = rain_corrected

                # Merge QM rain into hybrid daily
                hybrid_d["date"] = pd.to_datetime(hybrid_d["date"])
                merged_rain = hybrid_d.merge(
                    site_imerg[["date", "rain_qm"]], on="date", how="left"
                )
                # Use QM rain where available, else keep ERA5 rain
                has_qm = merged_rain["rain_qm"].notna()
                hybrid_d.loc[has_qm.values, "rain_mm"] = merged_rain.loc[has_qm, "rain_qm"].values
                print(f"    IMERG QM rain applied: {has_qm.sum()}/{len(hybrid_d)} days")
            else:
                print(f"    No IMERG data for {site_id}")
        else:
            print("    No QM model or IMERG — using ERA5 rain")

        hybrid_fwi = _fwi(hybrid_d)
        if "FWI" not in hybrid_fwi.columns:
            print("  Skip: hybrid FWI failed")
            continue
        n_cfd = (hybrid_d["source"] == "CFD").sum()
        print(f"  Hybrid: {len(hybrid_fwi)} days ({n_cfd} CFD-substituted), FWI mean={hybrid_fwi['FWI'].mean():.1f}")

        # Align on dates
        icos_fwi["date"] = pd.to_datetime(icos_fwi["date"]).dt.date
        era5_fwi["date"] = pd.to_datetime(era5_fwi["date"]).dt.date
        hybrid_fwi["date"] = pd.to_datetime(hybrid_fwi["date"]).dt.date

        merged = (
            icos_fwi[["date", "FWI", "ISI"]].rename(columns={"FWI": "FWI_obs", "ISI": "ISI_obs"})
            .merge(era5_fwi[["date", "FWI", "ISI"]].rename(columns={"FWI": "FWI_era5", "ISI": "ISI_era5"}), on="date", how="inner")
            .merge(hybrid_fwi[["date", "FWI", "ISI", "source"]].rename(columns={"FWI": "FWI_hybrid", "ISI": "ISI_hybrid"}), on="date", how="inner")
        )
        merged["site_id"] = site_id
        merged.to_csv(args.output / f"{site_id}_fwi_hybrid.csv", index=False)

        # Metrics
        for src, col in [("ERA5", "FWI_era5"), ("Hybrid", "FWI_hybrid")]:
            valid = merged[["FWI_obs", col]].dropna()
            if len(valid) < 3: continue
            err = valid[col] - valid["FWI_obs"]
            all_metrics.append({
                "site_id": site_id, "source": src, "n": len(valid),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "bias": float(err.mean()),
            })
            # Also on CFD-substituted days only
            cfd_days = merged[merged["source"] == "CFD"]
            if src == "Hybrid" and len(cfd_days) >= 3:
                v2 = cfd_days[["FWI_obs", col]].dropna()
                if len(v2) >= 3:
                    e2 = v2[col] - v2["FWI_obs"]
                    all_metrics.append({
                        "site_id": site_id, "source": "Hybrid_CFD_days", "n": len(v2),
                        "mae": float(e2.abs().mean()),
                        "rmse": float(np.sqrt((e2**2).mean())),
                        "bias": float(e2.mean()),
                    })

        print(f"  Matched: {len(merged)} days")
        for src, col in [("ERA5", "FWI_era5"), ("Hybrid", "FWI_hybrid")]:
            if col in merged:
                e = (merged[col] - merged["FWI_obs"]).dropna()
                print(f"    FWI {src}: MAE={e.abs().mean():.2f}, bias={e.mean():.2f}")

    # Summary
    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        mdf.to_csv(args.output / "fwi_hybrid_metrics.csv", index=False)
        print(f"\n{'='*60}")
        print("FWI HYBRID COMPARISON SUMMARY")
        print(f"{'='*60}")
        for src in ["ERA5", "Hybrid", "Hybrid_CFD_days"]:
            sub = mdf[mdf["source"] == src]
            if sub.empty: continue
            print(f"  {src:20s}: MAE={sub['mae'].mean():.2f}, RMSE={sub['rmse'].mean():.2f}, bias={sub['bias'].mean():.2f} ({sub['n'].sum()} days)")


if __name__ == "__main__":
    main()
