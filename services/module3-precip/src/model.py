"""XGBoost precipitation correction model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from .dataset import split_spatial

logger = logging.getLogger(__name__)

# Large-capacity XGBoost — for the upgraded v2 precip correction trained on
# 1500+ stations with IMERG + ERA5-Land + terrain + climatology features.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "max_depth": 10,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_alpha": 0.3,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


FEATURE_COLUMNS_V1 = [
    "rain_imerg", "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi", "lat", "lon", "month_sin", "month_cos",
]

FEATURE_COLUMNS_V2 = [
    # Rain anchors — IMERG + ERA5-Land + temporal context (key for BUI)
    "rain_imerg", "rain_era5land",
    "rain_imerg_lag1", "rain_imerg_lag2",
    "rain_imerg_3d", "rain_imerg_7d", "rain_era5land_3d",
    # Terrain (multi-radius TPI + distance to coast)
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "tpi_1km", "tpi_5km", "tpi_10km", "dist_coast_km",
    # Wind-orography interaction from ERA5
    "u10", "v10", "wspd10", "upslope_flow",
    # Climatology conditioning (IMERG 10-yr monthly mean + anomaly)
    "clim_rain_month", "clim_rain_ratio",
    # Location + time (day-of-year is finer than month for Mediterranean)
    "lat", "lon", "doy_sin", "doy_cos", "month_sin", "month_cos",
]


def get_feature_columns(version: str = "v1") -> list[str]:
    """Return the feature column list for the requested model version.

    v1 — original 10-feature set (IMERG + terrain + month) — compatible with
         services/module3-precip/train.py out of the box.
    v2 — upgraded 27-feature set (adds ERA5-Land, IMERG lags, multi-radius
         TPI, wind-orography, and climatology anomaly). Requires dataset.py
         to build the extra columns — use when training the large model.
    """
    if version == "v2":
        return list(FEATURE_COLUMNS_V2)
    return list(FEATURE_COLUMNS_V1)


def train_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with spatial cross-validation and collect metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with feature columns and 'rain_station' target.
    n_folds : int
        Number of spatial CV folds.
    params : dict, optional
        XGBoost parameters. Defaults to DEFAULT_PARAMS.

    Returns
    -------
    dict
        cv_metrics (per-fold), mean_rmse, std_rmse, mean_bias,
        feature_importance (averaged across folds).
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    features = get_feature_columns()
    target = "rain_station"
    splits = split_spatial(df, n_folds=n_folds)

    cv_metrics: list[dict[str, float]] = []
    importance_sum: dict[str, float] = {f: 0.0 for f in features}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)

        # Regression metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        bias = float(np.mean(y_pred - y_test))
        corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0.0

        # Wet-day detection skill (rain > 1 mm)
        wet_true = (y_test > 1.0).astype(int)
        wet_pred = (y_pred > 1.0).astype(int)
        precision = float(precision_score(wet_true, wet_pred, zero_division=0))
        recall = float(recall_score(wet_true, wet_pred, zero_division=0))

        fold_metrics = {
            "fold": fold_idx,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "correlation": corr,
            "wet_precision": precision,
            "wet_recall": recall,
        }
        cv_metrics.append(fold_metrics)
        logger.info("Fold %d: RMSE=%.3f, MAE=%.3f, bias=%.3f", fold_idx, rmse, mae, bias)

        # Accumulate feature importance
        imp = model.feature_importances_
        for fi, fname in enumerate(features):
            importance_sum[fname] += float(imp[fi])

    # Aggregate
    rmses = [m["rmse"] for m in cv_metrics]
    biases = [m["bias"] for m in cv_metrics]
    feature_importance = {k: v / n_folds for k, v in importance_sum.items()}

    return {
        "cv_metrics": cv_metrics,
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_bias": float(np.mean(biases)),
        "feature_importance": feature_importance,
    }


def train_final(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> xgb.XGBRegressor:
    """Train on all data and return the fitted model.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data.
    params : dict, optional
        XGBoost parameters. Defaults to DEFAULT_PARAMS.

    Returns
    -------
    xgb.XGBRegressor
        Trained model.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    features = get_feature_columns()
    target = "rain_station"

    model = xgb.XGBRegressor(**params)
    model.fit(df[features], df[target], verbose=False)
    logger.info("Final model trained on %d samples", len(df))

    return model


def save_model(
    model: xgb.XGBRegressor,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save model, metrics, and feature columns to output directory.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    metrics : dict
        CV metrics dict from train_cv.
    output_dir : str or Path
        Directory to save artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model as JSON
    model_path = output_dir / "model.json"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Feature columns
    features_path = output_dir / "feature_columns.json"
    with open(features_path, "w") as f:
        json.dump(get_feature_columns(), f, indent=2)
    logger.info("Feature columns saved to %s", features_path)


def load_model(model_dir: str | Path) -> xgb.XGBRegressor:
    """Load a saved XGBoost model from directory.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model.json.

    Returns
    -------
    xgb.XGBRegressor
        Loaded model.
    """
    model_path = Path(model_dir) / "model.json"
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model


def predict_from_features(
    model: xgb.XGBRegressor,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict corrected precipitation from a pre-built feature frame.

    The caller is responsible for computing the engineered features listed
    in ``get_feature_columns()``. Use this for production pipelines where
    features are assembled from gridded inputs (module3_precip/predict.py).
    """
    missing = set(get_feature_columns()) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns for prediction: {sorted(missing)}")
    return np.clip(model.predict(features[get_feature_columns()]), 0.0, None)


# Legacy scalar-argument predict() helper removed in v2 — use
# predict_from_features(model, features_df) with engineered features
# assembled by services/module3-precip/predict.py.
