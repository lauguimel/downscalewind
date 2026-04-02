"""Merge station observations, IMERG, and terrain into a training-ready DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_dataset(
    station_data: pd.DataFrame,
    imerg_data: pd.DataFrame,
    terrain_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge station obs, IMERG, and terrain into a clean training DataFrame.

    Parameters
    ----------
    station_data : pd.DataFrame
        Columns must include: station_id, date, rain_station.
    imerg_data : pd.DataFrame
        Columns must include: station_id, date, rain_imerg.
    terrain_data : pd.DataFrame
        Columns must include: station_id, elevation, slope, aspect, tpi.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with temporal and aspect encodings, ready for model.
    """
    # Inner join station + IMERG on (station_id, date)
    df = station_data.merge(imerg_data, on=["station_id", "date"], how="inner")

    # Left join terrain on station_id
    df = df.merge(terrain_data, on="station_id", how="left")

    # Temporal features
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    df["day_of_year"] = dt.dt.dayofyear

    # Aspect encoding (circular)
    aspect_rad = np.deg2rad(df["aspect"])
    df["aspect_sin"] = np.sin(aspect_rad)
    df["aspect_cos"] = np.cos(aspect_rad)

    # Rain ratio for analysis (not a model feature)
    df["rain_ratio"] = df["rain_station"] / np.maximum(df["rain_imerg"], 0.1)

    # Drop rows with NaN in key columns
    key_cols = [
        "rain_station",
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
    df = df.dropna(subset=key_cols).reset_index(drop=True)

    return df


def split_spatial(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Spatial cross-validation splits grouped by station_id.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'station_id' column.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed (used to shuffle stations before grouping).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.
    """
    # Shuffle station assignment for reproducibility
    rng = np.random.default_rng(seed)
    unique_stations = df["station_id"].unique()
    rng.shuffle(unique_stations)
    station_rank = {sid: i for i, sid in enumerate(unique_stations)}
    groups = df["station_id"].map(station_rank).values

    gkf = GroupKFold(n_splits=n_folds)
    splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in gkf.split(df, groups=groups)
    ]
    return splits
