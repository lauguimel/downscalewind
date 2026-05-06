"""
Train and validate a tabular DownscalRain baseline on cached station data.

This is the immediate real-data validation path while the CNN branch waits for
gridded IMERG/ERA5-Land/terrain patches. It uses station-grouped folds to avoid
station leakage and compares raw IMERG against a two-stage occurrence+amount
model.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))

from src.downscalrain import precipitation_metrics

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


FEATURES = [
    "rain_imerg",
    "rain_imerg_lag1",
    "rain_imerg_lag2",
    "rain_imerg_3d",
    "rain_imerg_7d",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi",
    "lat",
    "lon",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]


def _load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "rain_mm" in df.columns and "rain_station" not in df.columns:
        df = df.rename(columns={"rain_mm": "rain_station"})
    required = {"station_id", "date", "rain_station", "rain_imerg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "day_of_year" not in df:
        df["day_of_year"] = df["date"].dt.dayofyear
    if "month" not in df:
        df["month"] = df["date"].dt.month
    if "month_sin" not in df:
        df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    if "aspect_sin" not in df and "aspect" in df:
        aspect_rad = np.deg2rad(df["aspect"])
        df["aspect_sin"] = np.sin(aspect_rad)
        df["aspect_cos"] = np.cos(aspect_rad)

    df["doy_sin"] = np.sin(2.0 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos"] = np.cos(2.0 * np.pi * df["day_of_year"] / 366.0)

    g = df.groupby("station_id", sort=False)["rain_imerg"]
    df["rain_imerg_lag1"] = g.shift(1)
    df["rain_imerg_lag2"] = g.shift(2)
    df["rain_imerg_3d"] = g.rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["rain_imerg_7d"] = g.rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)

    for col in FEATURES:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rain_station", "rain_imerg", *FEATURES]).reset_index(drop=True)
    df["is_fire_season"] = df["date"].dt.month.between(5, 10)
    return df


def _fit_predict_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    wet_threshold_mm: float,
    heavy_rain_weight: float,
    rain_amount_weight: float,
    seed: int,
) -> pd.DataFrame:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    x_train = train[FEATURES].to_numpy(np.float32)
    x_test = test[FEATURES].to_numpy(np.float32)
    y_train = train["rain_station"].to_numpy(np.float32)

    wet_train = y_train > wet_threshold_mm
    occurrence = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    occurrence.fit(x_train, wet_train.astype(int))

    if wet_train.sum() < 10:
        raise ValueError("not enough wet days in training fold")
    wet_amounts = y_train[wet_train]
    sample_weight = np.ones_like(wet_amounts, dtype=np.float32)
    sample_weight *= np.where(wet_amounts > 10.0, float(heavy_rain_weight), 1.0)
    if rain_amount_weight > 0:
        sample_weight *= np.clip(1.0 + float(rain_amount_weight) * wet_amounts, 1.0, 3.0)
    amount = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    amount.fit(x_train[wet_train], np.log1p(wet_amounts), sample_weight=sample_weight)

    wet_prob = occurrence.predict_proba(x_test)[:, 1]
    amount_mm = np.expm1(amount.predict(x_test))
    pred = np.clip(wet_prob * amount_mm, 0.0, None)

    keep_cols = [
        "station_id",
        "date",
        "lat",
        "lon",
        "rain_station",
        "rain_imerg",
        "elevation",
        "slope",
        "tpi",
        "month",
        "is_fire_season",
    ]
    out = test[keep_cols].copy()
    out["rain_pred_tabular"] = pred
    out["wet_probability"] = wet_prob
    out["conditional_amount_mm"] = np.clip(amount_mm, 0.0, None)
    return out


def _metric_rows(pred: pd.DataFrame, subset_name: str, group_name: str = "global") -> list[dict[str, Any]]:
    rows = []
    for model_name, pred_col in [
        ("raw_imerg", "rain_imerg"),
        ("tabular_occ_amount_balanced", "rain_pred_tabular"),
    ]:
        metrics = precipitation_metrics(
            pred["rain_station"].to_numpy(),
            pred[pred_col].to_numpy(),
            wet_threshold_mm=1.0,
            heavy_threshold_mm=10.0,
        )
        rows.append({
            "subset": subset_name,
            "group": group_name,
            "model": model_name,
            "n": int(len(pred)),
            **metrics,
        })
    return rows


def _train_final(
    df: pd.DataFrame,
    wet_threshold_mm: float,
    heavy_rain_weight: float,
    rain_amount_weight: float,
    seed: int,
) -> tuple[Any, Any]:
    x = df[FEATURES].to_numpy(np.float32)
    y = df["rain_station"].to_numpy(np.float32)
    wet = y > wet_threshold_mm
    occurrence = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    occurrence.fit(x, wet.astype(int))
    wet_amounts = y[wet]
    sample_weight = np.ones_like(wet_amounts, dtype=np.float32)
    sample_weight *= np.where(wet_amounts > 10.0, float(heavy_rain_weight), 1.0)
    if rain_amount_weight > 0:
        sample_weight *= np.clip(1.0 + float(rain_amount_weight) * wet_amounts, 1.0, 3.0)
    amount = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    amount.fit(x[wet], np.log1p(wet_amounts), sample_weight=sample_weight)
    return occurrence, amount


def _add_grouped_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(_metric_rows(pred, "all"))
    rows.extend(_metric_rows(pred[pred["is_fire_season"]], "fire_season"))

    bins = [-np.inf, 200.0, 800.0, np.inf]
    labels = ["elev_low_lt200m", "elev_mid_200_800m", "elev_high_gt800m"]
    pred = pred.copy()
    pred["elevation_band"] = pd.cut(pred["elevation"], bins=bins, labels=labels)
    for band, group in pred.groupby("elevation_band", observed=True):
        if len(group):
            rows.extend(_metric_rows(group, "all", str(band)))
            fire = group[group["is_fire_season"]]
            if len(fire):
                rows.extend(_metric_rows(fire, "fire_season", str(band)))

    for month, group in pred.groupby("month"):
        if len(group):
            rows.extend(_metric_rows(group, f"month_{int(month):02d}", "monthly"))

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--dataset",
    "dataset_path",
    default="data/raw/precip_correction_cache/dataset_2022.parquet",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
    default="data/validation/downscalrain",
    type=click.Path(),
)
@click.option("--folds", default=5, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--wet-threshold-mm", default=0.2, show_default=True, type=float)
@click.option("--heavy-rain-weight", default=4.0, show_default=True, type=float)
@click.option("--rain-amount-weight", default=0.015, show_default=True, type=float)
@click.option("--save-predictions", is_flag=True, default=False)
@click.option("--model-dir", default=None, type=click.Path())
def main(
    dataset_path: str,
    output_dir: str,
    folds: int,
    seed: int,
    wet_threshold_mm: float,
    heavy_rain_weight: float,
    rain_amount_weight: float,
    save_predictions: bool,
    model_dir: str | None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _add_features(_load_dataset(dataset_path))
    groups = df["station_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)

    predictions = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups), start=1):
        pred = _fit_predict_fold(
            df,
            train_idx,
            test_idx,
            wet_threshold_mm,
            heavy_rain_weight,
            rain_amount_weight,
            seed + fold,
        )
        pred["fold"] = fold
        predictions.append(pred)
        for row in _metric_rows(pred, f"fold_{fold}_all"):
            row["fold"] = fold
            fold_rows.append(row)
        fire = pred[pred["is_fire_season"]]
        if len(fire):
            for row in _metric_rows(fire, f"fold_{fold}_fire_season"):
                row["fold"] = fold
                fold_rows.append(row)
        log.info("Fold %d complete: %d held-out rows", fold, len(pred))

    all_pred = pd.concat(predictions, ignore_index=True)
    by_fold = pd.DataFrame(fold_rows)
    summary = _add_grouped_metrics(all_pred)

    metrics_path = out_dir / "downscalrain_tabular_metrics.csv"
    fold_path = out_dir / "downscalrain_tabular_fold_metrics.csv"
    summary.to_csv(metrics_path, index=False)
    by_fold.to_csv(fold_path, index=False)

    if save_predictions:
        all_pred.to_parquet(out_dir / "downscalrain_tabular_predictions.parquet", index=False)

    if model_dir:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        occurrence, amount = _train_final(
            df,
            wet_threshold_mm,
            heavy_rain_weight,
            rain_amount_weight,
            seed,
        )
        dump(occurrence, model_path / "occurrence.joblib")
        dump(amount, model_path / "amount.joblib")
        (model_path / "feature_columns.json").write_text(json.dumps(FEATURES, indent=2))
        (model_path / "model_card.json").write_text(json.dumps({
            "model": "HistGradientBoosting occurrence + wet-day log-amount",
            "dataset": str(dataset_path),
            "n_samples": int(len(df)),
            "n_stations": int(df["station_id"].nunique()),
            "wet_threshold_mm_training": float(wet_threshold_mm),
            "heavy_rain_weight": float(heavy_rain_weight),
            "rain_amount_weight": float(rain_amount_weight),
            "features": FEATURES,
        }, indent=2))
        log.info("Saved final tabular model to %s", model_path)

    payload = {
        "dataset": str(dataset_path),
        "n_samples": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "folds": int(folds),
        "wet_threshold_mm_training": float(wet_threshold_mm),
        "heavy_rain_weight": float(heavy_rain_weight),
        "rain_amount_weight": float(rain_amount_weight),
        "features": FEATURES,
        "metrics": summary.to_dict(orient="records"),
    }
    (out_dir / "downscalrain_tabular_summary.json").write_text(json.dumps(payload, indent=2))

    click.echo("\n" + "=" * 72)
    click.echo("DownscalRain tabular station-group validation")
    click.echo(f"  samples:  {len(df)}")
    click.echo(f"  stations: {df['station_id'].nunique()}")
    click.echo(f"  metrics:  {metrics_path}")
    main_rows = summary[(summary["group"] == "global") & (summary["subset"].isin(["all", "fire_season"]))]
    for row in main_rows.to_dict(orient="records"):
        click.echo(
            f"  {row['subset']:12s} {row['model']:18s} "
            f"RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} "
            f"bias={row['bias']:.3f} wet_recall={row['wet_recall']:.3f}"
        )
    click.echo("=" * 72)


if __name__ == "__main__":
    main()
