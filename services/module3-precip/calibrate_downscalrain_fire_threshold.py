"""
Post-hoc fire-season calibration for DownscalRain CNN predictions.

The CNN predicts expected rain as P(wet) * conditional amount. For FWI, the
most damaging failure mode is false precipitation during dry fire windows, so
this script learns a validation-only wet-probability threshold and zeroes weak
rain predictions below that threshold. The raw CNN prediction is kept unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from src.downscalrain import precipitation_metrics

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _load_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {
        "station_id",
        "date",
        "split",
        "lat",
        "lon",
        "rain_station",
        "rain_imerg_center",
        "rain_era5land_center",
        "rain_pred_mm",
        "wet_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibration input missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["rain_station"] = df["rain_station"].clip(lower=0.0)
    df["rain_pred_mm"] = df["rain_pred_mm"].clip(lower=0.0)
    df["wet_probability"] = df["wet_probability"].clip(lower=0.0, upper=1.0)
    return df


def _fire_mediterranean_mask(df: pd.DataFrame, fire_months: tuple[int, ...]) -> pd.Series:
    return (
        df["month"].isin(fire_months)
        & df["lat"].between(35.0, 45.5)
        & df["lon"].between(-10.0, 25.0)
    )


def _safe_metrics(y_true: pd.Series, y_pred: pd.Series, wet_threshold_mm: float) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "correlation": np.nan,
            "wet_precision": np.nan,
            "wet_recall": np.nan,
            "dry_false_alarm": np.nan,
            "heavy_recall": np.nan,
        }
    return precipitation_metrics(
        y_true.to_numpy(),
        y_pred.clip(lower=0.0).to_numpy(),
        wet_threshold_mm=wet_threshold_mm,
        heavy_threshold_mm=10.0,
    )


def _score_threshold(
    df: pd.DataFrame,
    threshold: float,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    min_wet_recall: float,
) -> dict[str, Any]:
    pred = df["rain_pred_mm"].where(df["wet_probability"] >= threshold, 0.0)
    dry = df["rain_station"] <= dry_threshold_mm
    wet = df["rain_station"] > wet_threshold_mm
    halo = dry & (
        (df["rain_imerg_center"] > wet_threshold_mm)
        | (df["rain_era5land_center"] > wet_threshold_mm)
    )
    focus = halo if int(halo.sum()) >= 20 else dry

    all_metrics = _safe_metrics(df["rain_station"], pred, wet_threshold_mm)
    dry_pred = pred[dry].clip(lower=0.0)
    wet_metrics = _safe_metrics(df.loc[wet, "rain_station"], pred[wet], wet_threshold_mm)
    focus_metrics = _safe_metrics(df.loc[focus, "rain_station"], pred[focus], wet_threshold_mm)

    false_wet_rate = float((dry_pred > wet_threshold_mm).mean()) if len(dry_pred) else np.nan
    dry_mean = float(dry_pred.mean()) if len(dry_pred) else np.nan
    wet_recall = float(all_metrics["wet_recall"])
    dryguard_objective = (
        float(focus_metrics["rmse"])
        + 0.20 * float(all_metrics["mae"])
        + 4.0 * false_wet_rate
    )
    objective = (
        dryguard_objective
        + 8.0 * max(0.0, min_wet_recall - wet_recall)
    )

    return {
        "threshold": float(threshold),
        "objective": float(objective),
        "dryguard_objective": float(dryguard_objective),
        "n": int(len(df)),
        "n_dry": int(dry.sum()),
        "n_halo": int(halo.sum()),
        "n_wet": int(wet.sum()),
        "all_rmse": float(all_metrics["rmse"]),
        "all_mae": float(all_metrics["mae"]),
        "all_bias": float(all_metrics["bias"]),
        "wet_recall": wet_recall,
        "wet_mae": float(wet_metrics["mae"]),
        "focus_rmse": float(focus_metrics["rmse"]),
        "false_wet_rate_on_dry": false_wet_rate,
        "mean_pred_on_dry_mm": dry_mean,
    }


def _sweep_thresholds(
    val: pd.DataFrame,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    min_wet_recall: float,
) -> pd.DataFrame:
    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 91),
        np.quantile(val["wet_probability"].to_numpy(), np.linspace(0.05, 0.95, 19)),
    ]))
    rows = [
        _score_threshold(val, float(th), dry_threshold_mm, wet_threshold_mm, min_wet_recall)
        for th in candidates
    ]
    return pd.DataFrame(rows).sort_values(["objective", "threshold"]).reset_index(drop=True)


def _metrics_for_column(df: pd.DataFrame, column: str, wet_threshold_mm: float) -> dict[str, float]:
    return precipitation_metrics(
        df["rain_station"].to_numpy(),
        df[column].clip(lower=0.0).to_numpy(),
        wet_threshold_mm=wet_threshold_mm,
        heavy_threshold_mm=10.0,
    )


@click.command()
@click.option("--predictions", "predictions_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--calibration-json", required=True, type=click.Path())
@click.option("--threshold-table", required=True, type=click.Path())
@click.option("--dry-threshold-mm", default=1.0, show_default=True, type=float)
@click.option("--wet-threshold-mm", default=1.0, show_default=True, type=float)
@click.option("--min-wet-recall", default=0.60, show_default=True, type=float)
@click.option("--fire-months", default="6,7,8,9", show_default=True)
def main(
    predictions_path: str,
    output_path: str,
    calibration_json: str,
    threshold_table: str,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    min_wet_recall: float,
    fire_months: str,
) -> None:
    months = tuple(int(v.strip()) for v in fire_months.split(",") if v.strip())
    df = _load_frame(predictions_path)
    calib_mask = (df["split"] == "val") & _fire_mediterranean_mask(df, months)
    val = df[calib_mask].copy()
    if len(val) < 100:
        raise ValueError(f"not enough validation samples in fire calibration subset: {len(val)}")

    sweep = _sweep_thresholds(val, dry_threshold_mm, wet_threshold_mm, min_wet_recall)
    feasible = sweep[sweep["wet_recall"] >= min_wet_recall]
    if feasible.empty:
        log.warning(
            "No candidate met min wet recall %.3f; falling back to unconstrained objective.",
            min_wet_recall,
        )
        selection = sweep
    else:
        selection = feasible
    best = selection.sort_values(["objective", "threshold"]).iloc[0].to_dict()
    dryguard_best = sweep.sort_values(["dryguard_objective", "threshold"]).iloc[0].to_dict()
    threshold = float(best["threshold"])
    dryguard_threshold = float(dryguard_best["threshold"])

    out = df.copy()
    out["rain_pred_firecalibrated_mm"] = out["rain_pred_mm"].where(
        out["wet_probability"] >= threshold,
        0.0,
    )
    out["rain_pred_fireguard_mm"] = out["rain_pred_mm"].where(
        out["wet_probability"] >= dryguard_threshold,
        0.0,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    sweep_path = Path(threshold_table)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(sweep_path, index=False)

    test_fire = out[(out["split"] == "test") & _fire_mediterranean_mask(out, months)]
    summary = {
        "threshold": threshold,
        "dryguard_threshold": dryguard_threshold,
        "dry_threshold_mm": dry_threshold_mm,
        "wet_threshold_mm": wet_threshold_mm,
        "min_wet_recall": min_wet_recall,
        "constraint_satisfied": bool(best["wet_recall"] >= min_wet_recall),
        "fire_months": months,
        "calibration_subset": {
            "split": "val",
            "region": "Mediterranean 35-45.5N, -10-25E",
            "n": int(len(val)),
        },
        "best_validation_row": best,
        "dryguard_validation_row": dryguard_best,
        "test_fire_uncalibrated": _metrics_for_column(test_fire, "rain_pred_mm", wet_threshold_mm)
        if len(test_fire)
        else None,
        "test_fire_calibrated": _metrics_for_column(
            test_fire,
            "rain_pred_firecalibrated_mm",
            wet_threshold_mm,
        )
        if len(test_fire)
        else None,
        "test_fire_fireguard": _metrics_for_column(test_fire, "rain_pred_fireguard_mm", wet_threshold_mm)
        if len(test_fire)
        else None,
    }
    json_path = Path(calibration_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 76)
    print("DownscalRain fire calibration")
    print(f"  validation samples: {len(val)}")
    print(f"  balanced wet-probability threshold: {threshold:.3f}")
    print(f"  dryguard wet-probability threshold: {dryguard_threshold:.3f}")
    print(f"  threshold sweep: {sweep_path}")
    print(f"  calibrated predictions: {output}")
    print(f"  calibration JSON: {json_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
