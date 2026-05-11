"""
Build station FWI validation variants from frozen station-days.

This script is the station-level FWI join point:
  - OBS station FWI is the reference.
  - ERA5 meteorology is sampled from a Europe Zarr at the station point.
  - Rain variants come from IMERG, ERA5-Land, DownscalRain direct, and the
    IMERG-first dry-period correction.
  - Optional DownscaleWind/UVT predictions can be merged later without changing
    the station/timestamp protocol.

The output is intentionally long-format: one row per station/date/product, plus
metrics against OBS. This keeps the paper tables and plots reproducible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from shared.fwi import compute_fwi_series  # noqa: E402


@dataclass(frozen=True)
class ProductSpec:
    product: str
    meteo_source: str
    rain_source: str
    t_col: str
    rh_col: str
    wind_ms_col: str
    rain_col: str
    is_final_candidate: bool = False


def dewpoint_to_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
    return np.clip(rh, 0.0, 100.0)


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(values.astype(float) - float(value))))


def zarr_times_to_ns(times: np.ndarray) -> np.ndarray:
    if np.issubdtype(times.dtype, np.datetime64):
        return times.astype("datetime64[ns]").astype(np.int64)
    return times.astype(np.int64)


def ns_to_iso(value: int | np.integer) -> str:
    return pd.Timestamp(int(value), unit="ns").isoformat()


def sample_era5_meteo(station_days: pd.DataFrame, era5_zarr: Path) -> pd.DataFrame:
    import zarr

    g = zarr.open_group(str(era5_zarr), mode="r")
    times = np.asarray(g["coords/time"][:])
    time_ns = zarr_times_to_ns(times)
    lats = np.asarray(g["coords/lat"][:], dtype=float)
    lons = np.asarray(g["coords/lon"][:], dtype=float)

    # Cache nearest grid point per station; all selected stations are fixed.
    station_points = {}
    for row in station_days[["station_id", "lat", "lon"]].drop_duplicates("station_id").itertuples(index=False):
        station_points[str(row.station_id)] = (
            nearest_index(lats, float(row.lat)),
            nearest_index(lons, float(row.lon)),
        )

    out = []
    for row in station_days.itertuples(index=False):
        sid = str(row.station_id)
        i_lat, i_lon = station_points[sid]
        target = np.datetime64(pd.Timestamp(row.date).replace(hour=12), "ns").astype(np.int64)
        i_time = int(np.argmin(np.abs(time_ns - target)))

        t2m_k = float(g["surface/t2m"][i_time, i_lat, i_lon])
        d2m_k = float(g["surface/d2m"][i_time, i_lat, i_lon])
        u10 = float(g["surface/u10"][i_time, i_lat, i_lon])
        v10 = float(g["surface/v10"][i_time, i_lat, i_lon])
        t_c = t2m_k - 273.15
        td_c = d2m_k - 273.15
        out.append(
            {
                "station_id": sid,
                "date": pd.Timestamp(row.date).normalize(),
                "era5_time": ns_to_iso(time_ns[i_time]),
                "era5_lat": float(lats[i_lat]),
                "era5_lon": float(lons[i_lon]),
                "t_era5_c": t_c,
                "rh_era5_pct": float(dewpoint_to_rh(np.array([t_c]), np.array([td_c]))[0]),
                "wind_era5_ms": float(np.hypot(u10, v10)),
                "u10_era5_ms": u10,
                "v10_era5_ms": v10,
            }
        )
    return pd.DataFrame(out)


def add_optional_uvt(df: pd.DataFrame, uvt_path: Path | None) -> tuple[pd.DataFrame, bool]:
    if uvt_path is None or not uvt_path.exists():
        return df, False
    uvt = load_frame(uvt_path)
    uvt["date"] = pd.to_datetime(uvt["date"]).dt.normalize()
    required = {"station_id", "date", "t_downscaled_c", "rh_downscaled_pct", "wind_downscaled_ms"}
    missing = required - set(uvt.columns)
    if missing:
        raise ValueError(f"{uvt_path} missing required columns: {sorted(missing)}")
    merged = df.merge(
        uvt[list(required)],
        on=["station_id", "date"],
        how="left",
    )
    return merged, True


def product_specs(has_uvt: bool) -> list[ProductSpec]:
    specs = [
        ProductSpec(
            product="OBS_station_FWI",
            meteo_source="station",
            rain_source="station",
            t_col="t_obs_c",
            rh_col="rh_obs_pct",
            wind_ms_col="wind_obs_ms",
            rain_col="rain24_obs_mm",
        ),
        ProductSpec(
            product="ERA5_met_ObsRain_FWI",
            meteo_source="ERA5",
            rain_source="station",
            t_col="t_era5_c",
            rh_col="rh_era5_pct",
            wind_ms_col="wind_era5_ms",
            rain_col="rain24_obs_mm",
        ),
        ProductSpec(
            product="ERA5_met_IMERG_FWI",
            meteo_source="ERA5",
            rain_source="IMERG",
            t_col="t_era5_c",
            rh_col="rh_era5_pct",
            wind_ms_col="wind_era5_ms",
            rain_col="rain_imerg_center",
        ),
        ProductSpec(
            product="ERA5LandRain_FWI",
            meteo_source="ERA5",
            rain_source="ERA5-Land",
            t_col="t_era5_c",
            rh_col="rh_era5_pct",
            wind_ms_col="wind_era5_ms",
            rain_col="rain_era5land_center",
        ),
        ProductSpec(
            product="ERA5_met_DownscalRainDirect_FWI",
            meteo_source="ERA5",
            rain_source="DownscalRain direct",
            t_col="t_era5_c",
            rh_col="rh_era5_pct",
            wind_ms_col="wind_era5_ms",
            rain_col="rain_pred_mm",
        ),
        ProductSpec(
            product="ERA5_met_IMERGFireCorrected_FWI",
            meteo_source="ERA5",
            rain_source="IMERG fire-corrected",
            t_col="t_era5_c",
            rh_col="rh_era5_pct",
            wind_ms_col="wind_era5_ms",
            rain_col="rain_imerg_firecorrected_mm",
        ),
        ProductSpec(
            product="StationMet_IMERGFireCorrected_FWI",
            meteo_source="station",
            rain_source="IMERG fire-corrected",
            t_col="t_obs_c",
            rh_col="rh_obs_pct",
            wind_ms_col="wind_obs_ms",
            rain_col="rain_imerg_firecorrected_mm",
        ),
    ]
    if has_uvt:
        specs.append(
            ProductSpec(
                product="DownscaleWind_DownscalRain_FWI",
                meteo_source="DownscaleWind",
                rain_source="IMERG fire-corrected",
                t_col="t_downscaled_c",
                rh_col="rh_downscaled_pct",
                wind_ms_col="wind_downscaled_ms",
                rain_col="rain_imerg_firecorrected_mm",
                is_final_candidate=True,
            )
        )
    return specs


def compute_product(df: pd.DataFrame, spec: ProductSpec) -> pd.DataFrame:
    rows = []
    required = [spec.t_col, spec.rh_col, spec.wind_ms_col, spec.rain_col]
    valid_df = df.dropna(subset=required).copy()
    for station_id, group in valid_df.sort_values(["station_id", "date"]).groupby("station_id"):
        g = group.copy()
        with np.errstate(all="ignore"):
            fwi_out = compute_fwi_series(
                t_c=g[spec.t_col].astype(float).to_numpy(),
                rh=g[spec.rh_col].astype(float).to_numpy(),
                ws_kmh=(g[spec.wind_ms_col].astype(float).to_numpy() * 3.6),
                rain_mm=g[spec.rain_col].astype(float).clip(lower=0.0).to_numpy(),
                months=g["month"].astype(int).to_numpy(),
            )
        out = g[
            [
                "station_id",
                "date",
                "name",
                "lat",
                "lon",
                "alt_m",
                "region",
                "fwi_obs",
                "ffmc_obs",
                "dmc_obs",
                "dc_obs",
                "isi_obs",
                "bui_obs",
            ]
        ].copy()
        out = out.rename(
            columns={
                "ffmc_obs": "ffmc_obs_full_history",
                "dmc_obs": "dmc_obs_full_history",
                "dc_obs": "dc_obs_full_history",
                "isi_obs": "isi_obs_full_history",
                "bui_obs": "bui_obs_full_history",
                "fwi_obs": "fwi_obs_full_history",
            }
        )
        out["product"] = spec.product
        out["meteo_source"] = spec.meteo_source
        out["rain_source"] = spec.rain_source
        out["is_final_candidate"] = bool(spec.is_final_candidate)
        out["t_c"] = g[spec.t_col].astype(float).to_numpy()
        out["rh_pct"] = g[spec.rh_col].astype(float).to_numpy()
        out["wind_ms"] = g[spec.wind_ms_col].astype(float).to_numpy()
        out["rain24_mm"] = g[spec.rain_col].astype(float).clip(lower=0.0).to_numpy()
        for key, values in fwi_out.items():
            out[key] = values
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def metric_row(df: pd.DataFrame, product: str, subset: str, mask: np.ndarray) -> dict[str, float | int | str]:
    sub = df[(df["product"] == product) & mask].copy()
    valid = sub[["fwi_ref", "fwi"]].replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, float | int | str] = {"subset": subset, "product": product, "n": int(len(valid))}
    if valid.empty:
        row.update({"rmse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan})
        return row
    obs = valid["fwi_ref"].to_numpy(dtype=float)
    pred = valid["fwi"].to_numpy(dtype=float)
    row["rmse"] = float(np.sqrt(np.mean((pred - obs) ** 2)))
    row["mae"] = float(np.mean(np.abs(pred - obs)))
    row["bias"] = float(np.mean(pred - obs))
    row["corr"] = (
        float(np.corrcoef(obs, pred)[0, 1])
        if len(valid) > 2 and np.std(obs) > 0.0 and np.std(pred) > 0.0
        else np.nan
    )
    for thr in (5.2, 11.2, 21.3, 38.0):
        obs_hi = obs >= thr
        pred_hi = pred >= thr
        row[f"hit_rate_fwi_ge_{thr:g}"] = float(pred_hi[obs_hi].mean()) if obs_hi.any() else np.nan
        row[f"false_alarm_fwi_ge_{thr:g}"] = float(pred_hi[~obs_hi].mean()) if (~obs_hi).any() else np.nan
    return row


def build_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all": np.ones(len(long_df), dtype=bool),
        "obs_window_fwi_ge_5p2": long_df["fwi_ref"].to_numpy(dtype=float) >= 5.2,
        "obs_window_fwi_ge_11p2": long_df["fwi_ref"].to_numpy(dtype=float) >= 11.2,
        "obs_window_fwi_ge_21p3": long_df["fwi_ref"].to_numpy(dtype=float) >= 21.3,
        "obs_full_history_fwi_ge_11p2": long_df["fwi_obs_full_history"].to_numpy(dtype=float) >= 11.2,
        "obs_dry": long_df["rain_source"].notna().to_numpy() & (long_df["rain24_mm"].to_numpy(dtype=float) <= 1.0),
        "mediterranean": long_df["lat"].between(41.0, 45.5).to_numpy()
        & long_df["lon"].between(2.0, 10.5).to_numpy(),
    }
    rows = []
    for subset, mask in subsets.items():
        for product in sorted(long_df["product"].unique()):
            rows.append(metric_row(long_df, product, subset, mask))
    metrics = pd.DataFrame(rows)
    ref = metrics[(metrics["subset"] == "all") & (metrics["product"] == "ERA5LandRain_FWI")]
    if not ref.empty and np.isfinite(ref.iloc[0]["rmse"]):
        baseline_rmse = float(ref.iloc[0]["rmse"])
        metrics["skill_vs_ERA5LandRain_all_rmse"] = 1.0 - metrics["rmse"] / max(baseline_rmse, 1e-12)
    return metrics


def write_frame(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def write_report(output_dir: Path, metrics: pd.DataFrame, run_meta: dict) -> None:
    all_metrics = metrics[metrics["subset"].eq("all")].copy()
    preferred = [
        "OBS_station_FWI",
        "ERA5_met_ObsRain_FWI",
        "ERA5_met_IMERG_FWI",
        "ERA5LandRain_FWI",
        "ERA5_met_DownscalRainDirect_FWI",
        "ERA5_met_IMERGFireCorrected_FWI",
        "StationMet_IMERGFireCorrected_FWI",
        "DownscaleWind_DownscalRain_FWI",
    ]
    all_metrics["order"] = all_metrics["product"].map({p: i for i, p in enumerate(preferred)}).fillna(99)
    all_metrics = all_metrics.sort_values("order").drop(columns=["order"])
    cols = ["product", "n", "rmse", "mae", "bias", "corr", "skill_vs_ERA5LandRain_all_rmse"]
    cols = [c for c in cols if c in all_metrics.columns]
    lines = [
        "# FWI station validation baseline",
        "",
        f"- Stations: {run_meta['n_stations']}.",
        f"- Station-days: {run_meta['n_station_days']}.",
        f"- ERA5 source: `{run_meta['era5_zarr']}`.",
        f"- Downscaled UVT merged: {run_meta['has_downscaled_uvt']}.",
        "- FWI comparison convention: fire-season window recomputed from the first selected day per station.",
        "- `fwi_obs_full_history` is retained in the daily table; full-history model FWI still requires Jan-May ERA5/UVT meteo spin-up.",
        "",
        "## All station-days",
        "",
    ]
    try:
        lines.append(all_metrics[cols].to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        lines.append("```csv")
        lines.append(all_metrics[cols].to_csv(index=False))
        lines.append("```")
    if not run_meta["has_downscaled_uvt"]:
        lines.extend(
            [
                "",
                "## UVT status",
                "",
                "DownscaleWind_DownscalRain_FWI is not computed in this run because no station-level UVT inference table was provided.",
                "The frozen validation protocol and rain correction are ready; the missing step is the v2 station grid.zarr input builder plus ViT inference.",
            ]
        )
    (output_dir / "fwi_baseline_report.md").write_text("\n".join(lines) + "\n")


def run(
    station_days_path: Path,
    corrected_rain_path: Path,
    era5_zarr: Path,
    output_dir: Path,
    downscaled_uvt: Path | None = None,
) -> None:
    station_days = load_frame(station_days_path)
    corrected_rain = load_frame(corrected_rain_path)
    station_days["date"] = pd.to_datetime(station_days["date"]).dt.normalize()
    corrected_rain["date"] = pd.to_datetime(corrected_rain["date"]).dt.normalize()

    rain_cols = [
        "station_id",
        "date",
        "rain_imerg_center",
        "rain_era5land_center",
        "rain_pred_mm",
        "wet_probability",
        "rain_imerg_firebalanced_mm",
        "rain_imerg_firecorrected_mm",
        "firecorrected_gate",
    ]
    df = station_days.merge(corrected_rain[rain_cols], on=["station_id", "date"], how="left")
    era5 = sample_era5_meteo(df, era5_zarr)
    df = df.merge(era5, on=["station_id", "date"], how="left")
    df, has_uvt = add_optional_uvt(df, downscaled_uvt)

    products = [compute_product(df, spec) for spec in product_specs(has_uvt)]
    long_df = pd.concat([p for p in products if not p.empty], ignore_index=True)
    obs_ref = (
        long_df[long_df["product"].eq("OBS_station_FWI")]
        [["station_id", "date", "ffmc", "dmc", "dc", "isi", "bui", "fwi"]]
        .rename(
            columns={
                "ffmc": "ffmc_ref",
                "dmc": "dmc_ref",
                "dc": "dc_ref",
                "isi": "isi_ref",
                "bui": "bui_ref",
                "fwi": "fwi_ref",
            }
        )
    )
    long_df = long_df.merge(obs_ref, on=["station_id", "date"], how="left")
    metrics = build_metrics(long_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(era5, output_dir / "era5_station_meteo.csv")
    write_frame(long_df, output_dir / "fwi_daily_variants.csv")
    write_frame(metrics, output_dir / "fwi_baseline_metrics.csv")
    write_report(
        output_dir,
        metrics,
        {
            "n_stations": int(df["station_id"].nunique()),
            "n_station_days": int(len(df)),
            "era5_zarr": str(era5_zarr),
            "has_downscaled_uvt": bool(has_uvt),
        },
    )
    (output_dir / "fwi_baseline_run.json").write_text(
        json.dumps(
            {
                "station_days": str(station_days_path),
                "corrected_rain": str(corrected_rain_path),
                "era5_zarr": str(era5_zarr),
                "downscaled_uvt": str(downscaled_uvt) if downscaled_uvt else None,
                "n_stations": int(df["station_id"].nunique()),
                "n_station_days": int(len(df)),
                "n_product_rows": int(len(long_df)),
            },
            indent=2,
        )
    )
    print(f"stations={df['station_id'].nunique()}")
    print(f"station_days={len(df)}")
    print(f"product_rows={len(long_df)}")
    print(f"has_downscaled_uvt={has_uvt}")
    print(f"output_dir={output_dir}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-days",
        type=Path,
        default=PROJECT_ROOT / "data/validation/fwi_station_audit_2022_n20/validation_station_days.parquet",
    )
    parser.add_argument(
        "--corrected-rain",
        type=Path,
        default=PROJECT_ROOT / "data/validation/fwi_station_audit_2022_n20/rain_downscalrain_corrected.parquet",
    )
    parser.add_argument(
        "--era5-zarr",
        type=Path,
        default=PROJECT_ROOT / "data/raw/era5_europe_jja2022.zarr",
    )
    parser.add_argument("--downscaled-uvt", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/validation/fwi_station_audit_2022_n20",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.station_days, args.corrected_rain, args.era5_zarr, args.output_dir, args.downscaled_uvt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
