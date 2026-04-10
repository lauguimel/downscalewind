"""
fwi_4way_comparison.py — 4-way Fire Weather Index validation at ICOS sites.

For each ICOS station with co-located CFD/surrogate runs, compute FWI from
four different wind+precipitation sources and compare against in-situ
observations:

    (1)  ICOS_obs       — wind & T/RH/precip from the tower (ground truth)
    (2)  ERA5           — wind from ERA5 25 km, rain from ERA5-Land
    (3)  CFD + DS rain  — wind from OpenFOAM SF case at the tower lat/lon,
                          rain from the upgraded module3-precip downscaling
    (4)  Surrogate + DS — wind from FNO 9k v2 surrogate at the tower lat/lon,
                          rain from the upgraded module3-precip downscaling

Outputs (one row per (site, day)):
    data/validation/fwi_4way/{station}_fwi_4way.csv
    data/validation/fwi_4way/summary_metrics.csv     # RMSE / MAE / bias per site
    data/validation/figures/fwi_4way/{station}.png   # 4-way time-series
    data/validation/figures/fwi_4way/scatter_grid.png

Usage:
    python services/module2a-cfd/analysis/fwi_4way_comparison.py \\
        --icos-zarr-dir data/raw \\
        --era5-zarr data/raw/era5_europe_2020_2023.zarr \\
        --era5land-precip data/raw/era5land_precip_europe.zarr \\
        --cfd-tower-csv data/campaign/icos_fwi_v1/tower_profiles.csv \\
        --surrogate-tower-csv data/validation/surrogate_9k_v2/icos_tower_profiles.csv \\
        --precip-model data/models/precip_correction/v2 \\
        --output data/validation/fwi_4way
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.fwi import compute_fwi_series
from shared.logging_config import get_logger

log = get_logger("fwi_4way")


# ── FWI helper ───────────────────────────────────────────────────────────────


def _daily_noon_inputs(df: pd.DataFrame, t_col: str = "time") -> pd.DataFrame:
    """Aggregate hourly meteo to daily noon-mean (T, RH, ws) and 24h-sum precip.

    df must contain at least: time, T (degC), RH (%), ws (m/s), rain_mm.
    """
    df = df.copy()
    df[t_col] = pd.to_datetime(df[t_col], utc=True)
    df = df.set_index(t_col).sort_index()

    noon = df[df.index.hour.isin([11, 12, 13])][["T", "RH", "ws"]].resample("1D").mean()
    rain = df["rain_mm"].resample("1D").sum() if "rain_mm" in df.columns else None
    out = noon.dropna()
    if rain is not None:
        out["rain_mm"] = rain.reindex(out.index).fillna(0.0)
    else:
        out["rain_mm"] = 0.0
    return out.reset_index().rename(columns={t_col: "date"})


def _compute_fwi(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Run the FWI series on a daily noon dataframe and append index columns."""
    if df_daily.empty:
        return df_daily
    t_c = df_daily["T"].astype(np.float64).to_numpy()
    rh = np.clip(df_daily["RH"].astype(np.float64).to_numpy(), 0, 100)
    ws_kmh = df_daily["ws"].astype(np.float64).to_numpy() * 3.6
    rain = df_daily["rain_mm"].astype(np.float64).to_numpy()
    months = pd.to_datetime(df_daily["date"]).dt.month.to_numpy().astype(np.int32)
    res = compute_fwi_series(t_c, rh, ws_kmh, rain, months)
    out = df_daily.copy()
    for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
        out[k.upper()] = np.round(res[k], 2)
    return out


# ── Source builders ──────────────────────────────────────────────────────────


def _build_obs_source(icos_zarr: Path) -> pd.DataFrame:
    """Read in-situ ICOS station Zarr → hourly DataFrame."""
    import zarr
    root = zarr.open_group(str(icos_zarr), mode="r")
    times = pd.to_datetime(root["coords/time"][:].astype("datetime64[ns]"))
    cols = {"time": times}
    for key in ("T", "RH", "ws", "wd", "rain_mm"):
        if key in root["meteo"]:
            cols[key] = np.asarray(root[f"meteo/{key}"][:], dtype=np.float32)
    return pd.DataFrame(cols)


def _build_era5_source(
    era5_zarr: Path, era5land_zarr: Path, lat: float, lon: float
) -> pd.DataFrame:
    """Build hourly meteo from ERA5 + ERA5-Land at site location."""
    import xarray as xr

    ds = xr.open_zarr(era5_zarr, consolidated=False)
    p = ds.sel(lat=lat, lon=lon, method="nearest")
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(p.time.values),
            "T": p["t2m"].values - 273.15,
            "RH": _td2_rh(p["t2m"].values, p["d2m"].values) if "d2m" in p else np.nan,
            "ws": np.sqrt(p["u10"].values ** 2 + p["v10"].values ** 2),
            "wd": (270.0 - np.rad2deg(np.arctan2(p["v10"].values, p["u10"].values))) % 360,
        }
    )
    if era5land_zarr is not None and Path(era5land_zarr).exists():
        dl = xr.open_zarr(era5land_zarr, consolidated=False).sel(
            lat=lat, lon=lon, method="nearest"
        )
        rain = pd.Series(dl["tp"].values * 1000.0, index=pd.to_datetime(dl.time.values))
        df = df.set_index("time")
        df["rain_mm"] = rain.reindex(df.index).fillna(0.0)
        df = df.reset_index()
    else:
        df["rain_mm"] = 0.0
    return df


def _td2_rh(t2m_K: np.ndarray, d2m_K: np.ndarray) -> np.ndarray:
    """Convert (T, Td) → RH using Magnus formula."""
    a, b = 17.625, 243.04
    t_c = t2m_K - 273.15
    td_c = d2m_K - 273.15
    rh = 100.0 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t_c) / (b + t_c))
    return np.clip(rh, 0, 100).astype(np.float32)


def _build_cfd_source(
    cfd_tower_csv: Path, station_id: str, era5_df: pd.DataFrame, precip_corrected: pd.Series
) -> pd.DataFrame:
    """Replace ERA5 wind by CFD-sampled wind at the tower for matching timestamps."""
    cfd = pd.read_csv(cfd_tower_csv, parse_dates=["timestamp"])
    cfd = cfd[cfd["site_id"] == station_id].rename(columns={"timestamp": "time"})
    cfd["ws"] = np.sqrt(cfd["u"] ** 2 + cfd["v"] ** 2)
    cfd["wd"] = (270.0 - np.rad2deg(np.arctan2(cfd["v"], cfd["u"]))) % 360.0

    df = era5_df.set_index("time").copy()
    cfd_idx = cfd.set_index("time")
    df.loc[df.index.intersection(cfd_idx.index), "ws"] = cfd_idx["ws"]
    df.loc[df.index.intersection(cfd_idx.index), "wd"] = cfd_idx["wd"]
    df["rain_mm"] = precip_corrected.reindex(df.index).fillna(df["rain_mm"])
    return df.reset_index()


def _build_surrogate_source(
    surrogate_tower_csv: Path, station_id: str, era5_df: pd.DataFrame, precip_corrected: pd.Series
) -> pd.DataFrame:
    """Same as CFD, but reading the FNO 9k v2 surrogate predictions."""
    sur = pd.read_csv(surrogate_tower_csv, parse_dates=["timestamp"])
    sur = sur[sur["site_id"] == station_id].rename(columns={"timestamp": "time"})
    sur["ws"] = np.sqrt(sur["u"] ** 2 + sur["v"] ** 2)
    sur["wd"] = (270.0 - np.rad2deg(np.arctan2(sur["v"], sur["u"]))) % 360.0

    df = era5_df.set_index("time").copy()
    sur_idx = sur.set_index("time")
    df.loc[df.index.intersection(sur_idx.index), "ws"] = sur_idx["ws"]
    df.loc[df.index.intersection(sur_idx.index), "wd"] = sur_idx["wd"]
    if "T" in sur.columns:
        df.loc[df.index.intersection(sur_idx.index), "T"] = sur_idx["T"]
    if "q" in sur.columns:
        # Convert q (kg/kg) → RH approx via T at the same level
        df.loc[df.index.intersection(sur_idx.index), "RH"] = _q2rh(
            sur_idx["q"].values, df.loc[sur_idx.index, "T"].values
        )
    df["rain_mm"] = precip_corrected.reindex(df.index).fillna(df["rain_mm"])
    return df.reset_index()


def _q2rh(q: np.ndarray, t_c: np.ndarray, p_hpa: float = 1000.0) -> np.ndarray:
    """Specific humidity → RH (Magnus + Bolton's e_s, p in hPa)."""
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    return np.clip(100.0 * e / es, 0, 100)


# ── Precip downscaling helper ────────────────────────────────────────────────


def _load_corrected_precip(
    precip_model_dir: Path | None, lat: float, lon: float, period: tuple[str, str]
) -> pd.Series:
    """Run the upgraded precip downscaling at site location.

    Returns an hourly series (mm/h) of corrected precipitation. Falls back
    to NaN-filled series if the model directory is missing — the caller will
    then keep ERA5-Land rain as the precip source.
    """
    if precip_model_dir is None or not Path(precip_model_dir).exists():
        log.warning("No precip model — falling back to ERA5-Land rain")
        idx = pd.date_range(period[0], period[1], freq="1h", tz="UTC")
        return pd.Series(np.nan, index=idx, name="rain_mm")
    # Caller-side integration: this script accepts a pre-built CSV via
    # --precip-corrected-csv to keep dependencies light. The CSV must have
    # columns: time, station_id, rain_corrected_mm.
    raise NotImplementedError(
        "Pass --precip-corrected-csv with columns time/station_id/rain_corrected_mm"
    )


def _load_corrected_precip_csv(csv: Path, station_id: str) -> pd.Series:
    df = pd.read_csv(csv, parse_dates=["time"])
    df = df[df["station_id"] == station_id].set_index("time")
    s = df["rain_corrected_mm"].astype(float)
    s.name = "rain_mm"
    return s


# ── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icos-zarr-dir", required=True)
    ap.add_argument("--era5-zarr", required=True)
    ap.add_argument("--era5land-precip", default=None)
    ap.add_argument("--cfd-tower-csv", required=True)
    ap.add_argument("--surrogate-tower-csv", required=True)
    ap.add_argument("--precip-corrected-csv", required=True,
                    help="CSV from module3-precip predict pipeline (time, station_id, rain_corrected_mm)")
    ap.add_argument("--sites-csv", default="data/campaign/icos_fwi_v1/sites.csv")
    ap.add_argument("--output", default="data/validation/fwi_4way")
    args = ap.parse_args()

    out = Path(args.output)
    (out / "per_site").mkdir(parents=True, exist_ok=True)
    fig_dir = Path("data/validation/figures/fwi_4way")
    fig_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.sites_csv)
    summary_rows: list[dict] = []

    for _, s in sites.iterrows():
        sid = s["site_id"]
        zarr_path = Path(args.icos_zarr_dir) / f"icos_{sid.replace('-', '_').lower()}.zarr"
        if not zarr_path.exists():
            log.warning("ICOS zarr missing", extra={"site": sid, "path": str(zarr_path)})
            continue

        log.info("Processing site", extra={"site": sid})

        # 1) Observations
        obs_h = _build_obs_source(zarr_path)
        obs_d = _compute_fwi(_daily_noon_inputs(obs_h))

        # Corrected precip series for this site
        precip_corr = _load_corrected_precip_csv(Path(args.precip_corrected_csv), sid)

        # 2) ERA5
        era5_h = _build_era5_source(args.era5_zarr, args.era5land_precip, s["lat"], s["lon"])
        era5_d = _compute_fwi(_daily_noon_inputs(era5_h))

        # 3) CFD + downscaled precip
        cfd_h = _build_cfd_source(args.cfd_tower_csv, sid, era5_h, precip_corr)
        cfd_d = _compute_fwi(_daily_noon_inputs(cfd_h))

        # 4) Surrogate + downscaled precip
        sur_h = _build_surrogate_source(args.surrogate_tower_csv, sid, era5_h, precip_corr)
        sur_d = _compute_fwi(_daily_noon_inputs(sur_h))

        # Align on date
        merged = (
            obs_d[["date", "FWI"]].rename(columns={"FWI": "FWI_obs"})
            .merge(era5_d[["date", "FWI"]].rename(columns={"FWI": "FWI_era5"}), on="date", how="left")
            .merge(cfd_d[["date", "FWI"]].rename(columns={"FWI": "FWI_cfd_dsp"}), on="date", how="left")
            .merge(sur_d[["date", "FWI"]].rename(columns={"FWI": "FWI_sur_dsp"}), on="date", how="left")
        )
        merged["site_id"] = sid
        merged.to_csv(out / "per_site" / f"{sid}_fwi_4way.csv", index=False)

        # Metrics
        for src in ("era5", "cfd_dsp", "sur_dsp"):
            col = f"FWI_{src}"
            valid = merged[["FWI_obs", col]].dropna()
            if len(valid) < 5:
                continue
            err = valid[col] - valid["FWI_obs"]
            summary_rows.append(
                {
                    "site_id": sid,
                    "source": src,
                    "n": len(valid),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "mae": float(err.abs().mean()),
                    "bias": float(err.mean()),
                    "corr": float(valid.corr().iloc[0, 1]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary_metrics.csv", index=False)

    # Print pivot of RMSE per (site, source)
    if not summary.empty:
        pivot = summary.pivot(index="site_id", columns="source", values="rmse").round(2)
        print("\n=== FWI RMSE by source (lower = better) ===")
        print(pivot.to_string())
        print(f"\nMean RMSE — ERA5: {summary[summary.source=='era5'].rmse.mean():.2f}")
        print(f"Mean RMSE — CFD+DSP: {summary[summary.source=='cfd_dsp'].rmse.mean():.2f}")
        print(f"Mean RMSE — Surrogate+DSP: {summary[summary.source=='sur_dsp'].rmse.mean():.2f}")
    else:
        print("No sites had enough overlap for metrics.")


if __name__ == "__main__":
    main()
