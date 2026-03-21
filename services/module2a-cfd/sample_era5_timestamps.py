"""
sample_era5_timestamps.py — Select diverse ERA5 timestamps for SF PoC campaign.

Uses k-means clustering on (speed, direction, vertical shear) to select
N representative timestamps from the ERA5 Zarr archive.

Usage
-----
    cd services/module2a-cfd
    python sample_era5_timestamps.py \
        --era5   ../../data/raw/era5_perdigao.zarr \
        --n      25 \
        --output ../../data/campaign/sf_poc/timestamps.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_era5_wind(era5_path: str | Path) -> pd.DataFrame:
    """Load ERA5 wind data from Zarr and compute features for clustering.

    Returns a DataFrame with columns:
        datetime, u10, v10, speed_10m, direction_deg,
        dir_sin, dir_cos, shear (speed_850hPa - speed_10m)
    """
    import zarr

    store = zarr.open_group(str(era5_path), mode="r")

    # Time axis
    times_ns = np.array(store["coords/time"][:])
    datetimes = times_ns.astype("datetime64[ns]")

    # Levels
    levels = np.array(store["coords/level"][:])

    # Central grid point (1,1) in the 3×3 grid
    u10 = np.array(store["surface/u10"][:, 1, 1])
    v10 = np.array(store["surface/v10"][:, 1, 1])

    # Pressure-level winds at 850 hPa for shear calculation
    idx_850 = int(np.argmin(np.abs(levels - 850.0)))
    u850 = np.array(store["pressure/u"][:, idx_850, 1, 1])
    v850 = np.array(store["pressure/v"][:, idx_850, 1, 1])

    speed_10m = np.sqrt(u10**2 + v10**2)
    speed_850 = np.sqrt(u850**2 + v850**2)

    # Meteorological direction (wind FROM, degrees clockwise from N)
    direction_deg = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0

    df = pd.DataFrame({
        "datetime": datetimes,
        "u10": u10,
        "v10": v10,
        "speed_10m": speed_10m,
        "direction_deg": direction_deg,
        "dir_sin": np.sin(np.radians(direction_deg)),
        "dir_cos": np.cos(np.radians(direction_deg)),
        "shear": speed_850 - speed_10m,
        "speed_850": speed_850,
    })
    return df


def select_diverse_timestamps(
    df: pd.DataFrame,
    n_clusters: int = 25,
    min_speed_ms: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select N diverse timestamps using k-means on wind features.

    Parameters
    ----------
    df : DataFrame with columns speed_10m, dir_sin, dir_cos, shear
    n_clusters : number of clusters (= timestamps to select)
    min_speed_ms : exclude near-calm conditions (SF unreliable < 1 m/s)
    random_state : reproducibility seed

    Returns
    -------
    DataFrame with selected timestamps + cluster_id column
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Filter out near-calm conditions
    mask = df["speed_10m"] >= min_speed_ms
    df_valid = df[mask].copy()
    logger.info(
        "Filtered %d/%d timestamps (speed >= %.1f m/s)",
        len(df_valid), len(df), min_speed_ms,
    )

    if len(df_valid) < n_clusters:
        logger.warning(
            "Only %d valid timestamps, reducing n_clusters to %d",
            len(df_valid), len(df_valid),
        )
        n_clusters = len(df_valid)

    # Clustering features: speed, direction (sin/cos), vertical shear
    features = df_valid[["speed_10m", "dir_sin", "dir_cos", "shear"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    df_valid = df_valid.copy()
    df_valid["cluster_id"] = labels

    # Select medoid (closest to cluster centre) for each cluster
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_features = features_scaled[cluster_mask]
        centre = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - centre, axis=1)
        medoid_local = np.argmin(distances)
        # Map back to df_valid index
        cluster_df_indices = df_valid.index[cluster_mask]
        selected_indices.append(cluster_df_indices[medoid_local])

    selected = df_valid.loc[selected_indices].sort_values("datetime").reset_index(drop=True)
    return selected


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Select diverse ERA5 timestamps for SF PoC")
    parser.add_argument("--era5", required=True, help="Path to ERA5 Zarr archive")
    parser.add_argument("--n", type=int, default=25, help="Number of timestamps to select")
    parser.add_argument("--min-speed", type=float, default=1.0, help="Min 10m wind speed [m/s]")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = load_era5_wind(args.era5)
    logger.info(
        "ERA5: %d timestamps, speed_10m %.1f–%.1f m/s, %s → %s",
        len(df), df["speed_10m"].min(), df["speed_10m"].max(),
        df["datetime"].iloc[0], df["datetime"].iloc[-1],
    )

    selected = select_diverse_timestamps(df, n_clusters=args.n, min_speed_ms=args.min_speed)
    logger.info("Selected %d timestamps:", len(selected))
    for _, row in selected.iterrows():
        logger.info(
            "  %s  speed=%.1f m/s  dir=%.0f°  shear=%.1f m/s  cluster=%d",
            row["datetime"], row["speed_10m"], row["direction_deg"],
            row["shear"], row["cluster_id"],
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
