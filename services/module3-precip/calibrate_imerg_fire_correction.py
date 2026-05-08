"""
Fire-season correction of IMERG precipitation using a learned dry gate.

This is intentionally not a replacement precipitation model. IMERG remains the
base product; the CNN wet probability is used only to identify likely false-rain
cases during the fire window and attenuate IMERG there.
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
        "wet_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IMERG correction input missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    for column in ("rain_station", "rain_imerg_center", "rain_era5land_center"):
        df[column] = df[column].clip(lower=0.0)
    df["wet_probability"] = df["wet_probability"].clip(lower=0.0, upper=1.0)
    return df


def _fire_mediterranean_mask(df: pd.DataFrame, fire_months: tuple[int, ...]) -> pd.Series:
    return (
        df["month"].isin(fire_months)
        & df["lat"].between(35.0, 45.5)
        & df["lon"].between(-10.0, 25.0)
    )


def _correct_imerg(
    df: pd.DataFrame,
    threshold: float,
    retention: float,
    wet_threshold_mm: float,
    fire_months: tuple[int, ...],
) -> pd.Series:
    out = df["rain_imerg_center"].astype(float).copy()
    fire = df["month"].isin(fire_months)
    gate = (
        fire
        & (df["rain_imerg_center"] > wet_threshold_mm)
        & (df["wet_probability"] < threshold)
    )
    out.loc[gate] = out.loc[gate] * retention
    return out.clip(lower=0.0)


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
    with np.errstate(invalid="ignore", divide="ignore"):
        return precipitation_metrics(
            y_true.to_numpy(),
            y_pred.clip(lower=0.0).to_numpy(),
            wet_threshold_mm=wet_threshold_mm,
            heavy_threshold_mm=10.0,
        )


def _score_candidate(
    df: pd.DataFrame,
    threshold: float,
    retention: float,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    fire_months: tuple[int, ...],
    target_wet_recall: float,
) -> dict[str, Any]:
    pred = _correct_imerg(df, threshold, retention, wet_threshold_mm, fire_months)
    dry = df["rain_station"] <= dry_threshold_mm
    halo = dry & (df["rain_imerg_center"] > wet_threshold_mm)
    focus = halo if int(halo.sum()) >= 20 else dry

    all_metrics = _safe_metrics(df["rain_station"], pred, wet_threshold_mm)
    focus_metrics = _safe_metrics(df.loc[focus, "rain_station"], pred[focus], wet_threshold_mm)
    dry_pred = pred[dry]
    false_wet_rate = float((dry_pred > wet_threshold_mm).mean()) if len(dry_pred) else np.nan
    dry_mean = float(dry_pred.mean()) if len(dry_pred) else np.nan
    n_corrected = int((
        df["month"].isin(fire_months)
        & (df["rain_imerg_center"] > wet_threshold_mm)
        & (df["wet_probability"] < threshold)
    ).sum())

    wet_recall = float(all_metrics["wet_recall"])
    dry_objective = (
        float(focus_metrics["rmse"])
        + 0.20 * float(all_metrics["mae"])
        + 4.0 * false_wet_rate
    )
    objective = dry_objective + 12.0 * max(0.0, target_wet_recall - wet_recall)

    return {
        "threshold": float(threshold),
        "retention": float(retention),
        "objective": float(objective),
        "dry_objective": float(dry_objective),
        "target_wet_recall": float(target_wet_recall),
        "n": int(len(df)),
        "n_dry": int(dry.sum()),
        "n_halo": int(halo.sum()),
        "n_corrected": n_corrected,
        "all_rmse": float(all_metrics["rmse"]),
        "all_mae": float(all_metrics["mae"]),
        "all_bias": float(all_metrics["bias"]),
        "wet_precision": float(all_metrics["wet_precision"]),
        "wet_recall": wet_recall,
        "focus_rmse": float(focus_metrics["rmse"]),
        "false_wet_rate_on_dry": false_wet_rate,
        "mean_pred_on_dry_mm": dry_mean,
    }


def _sweep(
    val: pd.DataFrame,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    fire_months: tuple[int, ...],
    recall_fraction: float,
) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_metrics = precipitation_metrics(
            val["rain_station"].to_numpy(),
            val["rain_imerg_center"].clip(lower=0.0).to_numpy(),
            wet_threshold_mm=wet_threshold_mm,
            heavy_threshold_mm=10.0,
        )
    target_wet_recall = float(raw_metrics["wet_recall"]) * recall_fraction
    thresholds = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 91),
        np.quantile(val["wet_probability"].to_numpy(), np.linspace(0.05, 0.95, 19)),
    ]))
    retentions = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    rows = [
        _score_candidate(
            val,
            float(threshold),
            float(retention),
            dry_threshold_mm,
            wet_threshold_mm,
            fire_months,
            target_wet_recall,
        )
        for threshold in thresholds
        for retention in retentions
    ]
    return pd.DataFrame(rows).sort_values(["objective", "threshold", "retention"]).reset_index(drop=True)


def _metrics_for_column(df: pd.DataFrame, column: str, wet_threshold_mm: float) -> dict[str, float]:
    with np.errstate(invalid="ignore", divide="ignore"):
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
@click.option("--sweep-table", required=True, type=click.Path())
@click.option("--dry-threshold-mm", default=1.0, show_default=True, type=float)
@click.option("--wet-threshold-mm", default=1.0, show_default=True, type=float)
@click.option("--wet-recall-retention", default=0.90, show_default=True, type=float)
@click.option("--fire-months", default="6,7,8,9", show_default=True)
def main(
    predictions_path: str,
    output_path: str,
    calibration_json: str,
    sweep_table: str,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    wet_recall_retention: float,
    fire_months: str,
) -> None:
    months = tuple(int(v.strip()) for v in fire_months.split(",") if v.strip())
    df = _load_frame(predictions_path)
    val_mask = (df["split"] == "val") & _fire_mediterranean_mask(df, months)
    val = df[val_mask].copy()
    if len(val) < 100:
        raise ValueError(f"not enough validation samples in IMERG correction subset: {len(val)}")

    sweep = _sweep(val, dry_threshold_mm, wet_threshold_mm, months, wet_recall_retention)
    feasible = sweep[sweep["wet_recall"] >= sweep["target_wet_recall"]]
    selection = feasible if len(feasible) else sweep
    best = selection.sort_values(["objective", "threshold", "retention"]).iloc[0].to_dict()
    guard = sweep.sort_values(["dry_objective", "threshold", "retention"]).iloc[0].to_dict()

    out = df.copy()
    out["rain_imerg_firebalanced_mm"] = _correct_imerg(
        out,
        float(best["threshold"]),
        float(best["retention"]),
        wet_threshold_mm,
        months,
    )
    out["rain_imerg_firecorrected_mm"] = _correct_imerg(
        out,
        float(guard["threshold"]),
        float(guard["retention"]),
        wet_threshold_mm,
        months,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    sweep_path = Path(sweep_table)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(sweep_path, index=False)

    test_fire = out[(out["split"] == "test") & _fire_mediterranean_mask(out, months)]
    summary = {
        "base_product": "rain_imerg_center",
        "gate": "wet_probability",
        "fire_months": months,
        "dry_threshold_mm": dry_threshold_mm,
        "wet_threshold_mm": wet_threshold_mm,
        "wet_recall_retention": wet_recall_retention,
        "calibration_subset": {
            "split": "val",
            "region": "Mediterranean 35-45.5N, -10-25E",
            "n": int(len(val)),
        },
        "balanced_reference": best,
        "dry_period_correction": guard,
        "test_fire_raw_imerg": _metrics_for_column(test_fire, "rain_imerg_center", wet_threshold_mm)
        if len(test_fire)
        else None,
        "test_fire_imerg_firebalanced": _metrics_for_column(
            test_fire,
            "rain_imerg_firebalanced_mm",
            wet_threshold_mm,
        )
        if len(test_fire)
        else None,
        "test_fire_imerg_firecorrected": _metrics_for_column(
            test_fire,
            "rain_imerg_firecorrected_mm",
            wet_threshold_mm,
        )
        if len(test_fire)
        else None,
    }
    json_path = Path(calibration_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 76)
    print("IMERG fire-season correction")
    print(f"  validation samples: {len(val)}")
    print(
        "  balanced reference: "
        f"threshold={best['threshold']:.3f} retention={best['retention']:.2f}"
    )
    print(
        "  dry-period correction: "
        f"threshold={guard['threshold']:.3f} retention={guard['retention']:.2f}"
    )
    print(f"  corrected predictions: {output}")
    print(f"  calibration JSON: {json_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
