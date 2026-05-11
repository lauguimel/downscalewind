"""
Apply the calibrated IMERG-first fire correction to station validation days.

Inputs:
  - gridded station comparison from sample_fwi_station_rain.py
  - DownscalRain CNN predictions for the same station/date rows
  - calibration JSON produced by calibrate_imerg_fire_correction.py

The output is the station-level rain table needed by FWI validation:
IMERG raw, ERA5-Land raw, CNN direct prediction, IMERG fire-balanced, and
IMERG dry-period fire-corrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def metrics(df: pd.DataFrame, column: str, wet_threshold_mm: float) -> dict[str, float]:
    valid = df[["rain24_obs_mm", column]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"product": column, "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    obs = valid["rain24_obs_mm"].clip(lower=0.0).to_numpy()
    pred = valid[column].clip(lower=0.0).to_numpy()
    wet_obs = obs > wet_threshold_mm
    wet_pred = pred > wet_threshold_mm
    dry_obs = ~wet_obs
    corr = np.corrcoef(obs, pred)[0, 1] if len(valid) > 2 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    return {
        "product": column,
        "n": int(len(valid)),
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "mae": float(np.mean(np.abs(pred - obs))),
        "bias": float(np.mean(pred - obs)),
        "corr": float(corr) if np.isfinite(corr) else np.nan,
        "dry_false_wet_rate": float(wet_pred[dry_obs].mean()) if dry_obs.any() else np.nan,
        "wet_recall": float(wet_pred[wet_obs].mean()) if wet_obs.any() else np.nan,
    }


def correct_imerg(
    df: pd.DataFrame,
    threshold: float,
    retention: float,
    wet_threshold_mm: float,
    fire_months: tuple[int, ...],
) -> pd.Series:
    out = df["rain_imerg_center"].astype(float).clip(lower=0.0).copy()
    gate = (
        df["month"].isin(fire_months)
        & (df["rain_imerg_center"] > wet_threshold_mm)
        & (df["wet_probability"] < threshold)
    )
    out.loc[gate] = out.loc[gate] * retention
    return out.clip(lower=0.0)


def summarize(df: pd.DataFrame, wet_threshold_mm: float, dry_threshold_mm: float) -> pd.DataFrame:
    products = [
        "rain_imerg_center",
        "rain_era5land_center",
        "rain_pred_mm",
        "rain_imerg_firebalanced_mm",
        "rain_imerg_firecorrected_mm",
    ]
    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "obs_dry": df["rain24_obs_mm"] <= dry_threshold_mm,
        "imerg_or_era5_halo": (df["rain24_obs_mm"] <= dry_threshold_mm)
        & ((df["rain_imerg_center"] > wet_threshold_mm) | (df["rain_era5land_center"] > wet_threshold_mm)),
        "high_obs_fwi": df["fwi_obs"] > 12.0,
        "mediterranean": df["lat"].between(35.0, 45.5) & df["lon"].between(-10.0, 25.0),
    }
    rows = []
    for subset_name, mask in subsets.items():
        sub = df[mask].copy()
        for product in products:
            row = metrics(sub, product, wet_threshold_mm)
            row["subset"] = subset_name
            rows.append(row)
    return pd.DataFrame(rows)[
        ["subset", "product", "n", "rmse", "mae", "bias", "corr", "dry_false_wet_rate", "wet_recall"]
    ]


def write_frame(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def write_report(output_dir: Path, summary: pd.DataFrame, config: dict) -> None:
    lines = [
        "# Station IMERG fire correction",
        "",
        f"- Wet threshold: {config['wet_threshold_mm']} mm/day.",
        f"- Dry threshold: {config['dry_threshold_mm']} mm/day.",
        f"- Balanced gate: wet_probability < {config['balanced_threshold']} then IMERG x {config['balanced_retention']}.",
        f"- Dry correction gate: wet_probability < {config['dry_threshold']} then IMERG x {config['dry_retention']}.",
        "",
        "## Metrics",
        "",
    ]
    try:
        lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        lines.append("```csv")
        lines.append(summary.to_csv(index=False))
        lines.append("```")
    (output_dir / "rain_correction_report.md").write_text("\n".join(lines) + "\n")


def run(
    station_rain_path: Path,
    predictions_path: Path,
    calibration_json: Path,
    output_dir: Path,
) -> None:
    station = load_frame(station_rain_path)
    predictions = load_frame(predictions_path)
    station["date"] = pd.to_datetime(station["date"]).dt.normalize()
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
    df = station.merge(
        predictions[
            [
                "station_id",
                "date",
                "rain_pred_mm",
                "wet_probability",
                "conditional_amount_mm",
            ]
        ],
        on=["station_id", "date"],
        how="left",
    )
    df["month"] = pd.to_datetime(df["date"]).dt.month

    calibration = json.loads(calibration_json.read_text())
    wet_threshold_mm = float(calibration.get("wet_threshold_mm", 1.0))
    dry_threshold_mm = float(calibration.get("dry_threshold_mm", 1.0))
    months = tuple(int(v) for v in calibration.get("fire_months", [6, 7, 8, 9]))
    balanced = calibration["balanced_reference"]
    dry = calibration["dry_period_correction"]
    df["rain_imerg_firebalanced_mm"] = correct_imerg(
        df,
        float(balanced["threshold"]),
        float(balanced["retention"]),
        wet_threshold_mm,
        months,
    )
    df["rain_imerg_firecorrected_mm"] = correct_imerg(
        df,
        float(dry["threshold"]),
        float(dry["retention"]),
        wet_threshold_mm,
        months,
    )
    df["firecorrected_gate"] = (
        df["month"].isin(months)
        & (df["rain_imerg_center"] > wet_threshold_mm)
        & (df["wet_probability"] < float(dry["threshold"]))
    )
    df["firebalanced_gate"] = (
        df["month"].isin(months)
        & (df["rain_imerg_center"] > wet_threshold_mm)
        & (df["wet_probability"] < float(balanced["threshold"]))
    )

    summary = summarize(df, wet_threshold_mm, dry_threshold_mm)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(df, output_dir / "rain_downscalrain_corrected.csv")
    write_frame(summary, output_dir / "rain_correction_metrics.csv")
    write_report(
        output_dir,
        summary,
        {
            "wet_threshold_mm": wet_threshold_mm,
            "dry_threshold_mm": dry_threshold_mm,
            "balanced_threshold": float(balanced["threshold"]),
            "balanced_retention": float(balanced["retention"]),
            "dry_threshold": float(dry["threshold"]),
            "dry_retention": float(dry["retention"]),
        },
    )
    print(f"rows={len(df)}")
    print(f"firecorrected_gate={int(df['firecorrected_gate'].sum())}")
    print(f"output_dir={output_dir}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-rain", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=PROJECT_ROOT
        / "data/validation/downscalrain_imerg_firecorrected_gee_2022/imerg_fire_correction.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.station_rain, args.predictions, args.calibration_json, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
