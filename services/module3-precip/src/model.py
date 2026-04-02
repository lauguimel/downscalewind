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

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "objective": "reg:squarederror",
    "random_state": 42,
}


def get_feature_columns() -> list[str]:
    """Return the list of feature column names for the model."""
    return [
        "rain_imerg",
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "tpi",
        "lat",
        "lon",
        "month_sin",
        "month_cos",
    ]


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


def predict(
    model: xgb.XGBRegressor,
    rain_imerg: float | np.ndarray,
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    elevation: float | np.ndarray,
    slope: float | np.ndarray,
    aspect: float | np.ndarray,
    tpi: float | np.ndarray,
    month: int | np.ndarray,
) -> float | np.ndarray:
    """Predict corrected precipitation.

    All inputs can be scalars or arrays of the same length.

    Parameters
    ----------
    model : xgb.XGBRegressor
        Trained model.
    rain_imerg, lat, lon, elevation, slope, aspect, tpi : float or array
        Input features.
    month : int or array
        Month (1-12) for temporal encoding.

    Returns
    -------
    float or np.ndarray
        Corrected precipitation.
    """
    rain_imerg = np.atleast_1d(np.asarray(rain_imerg, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    elevation = np.atleast_1d(np.asarray(elevation, dtype=np.float64))
    slope = np.atleast_1d(np.asarray(slope, dtype=np.float64))
    aspect = np.atleast_1d(np.asarray(aspect, dtype=np.float64))
    tpi = np.atleast_1d(np.asarray(tpi, dtype=np.float64))
    month = np.atleast_1d(np.asarray(month, dtype=np.float64))

    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    month_sin = np.sin(2.0 * np.pi * month / 12.0)
    month_cos = np.cos(2.0 * np.pi * month / 12.0)

    X = pd.DataFrame(
        {
            "rain_imerg": rain_imerg,
            "elevation": elevation,
            "slope": slope,
            "aspect_sin": aspect_sin,
            "aspect_cos": aspect_cos,
            "tpi": tpi,
            "lat": lat,
            "lon": lon,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
    )

    result = model.predict(X)

    if len(result) == 1:
        return float(result[0])
    return result
