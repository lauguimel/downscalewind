"""
train_precip_correction.py — Train a precipitation bias-correction model.

Corrects IMERG satellite precipitation using terrain features and
ground truth from GHCN-D / synop stations. Produces a model that
can estimate station-quality daily precipitation anywhere in Europe.

Pipeline:
  1. Download GHCN-D daily precipitation for ~2000 European stations
  2. Download IMERG satellite precipitation at each station location (via GEE)
  3. Extract terrain features from SRTM (altitude, slope, aspect, TPI)
  4. Train XGBoost: features=(IMERG, terrain, season, location) → target=station precip
  5. Validate on held-out stations (spatial cross-validation)

Usage:
    python train_precip_correction.py --year 2022 --output data/models/precip_correction/
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger

log = get_logger("precip_correction")


# ── GHCN-D download ─────────────────────────────────────────────────────────

def download_ghcnd_inventory(bbox: tuple = (-10, 35, 30, 60)) -> pd.DataFrame:
    """Download GHCN-D station inventory, filter to bbox with PRCP."""
    log.info("Downloading GHCN-D inventory...")
    url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    stations = []
    for line in r.text.strip().split("\n"):
        sid = line[:11].strip()
        lat = float(line[12:20])
        lon = float(line[21:30])
        elem = line[31:35].strip()
        first = int(line[36:40])
        last = int(line[41:45])
        if (elem == "PRCP" and bbox[1] <= lat <= bbox[3]
                and bbox[0] <= lon <= bbox[2] and last >= 2022):
            stations.append({"id": sid, "lat": lat, "lon": lon,
                             "first": first, "last": last})

    df = pd.DataFrame(stations).drop_duplicates(subset="id")
    log.info("Found %d stations with PRCP in bbox", len(df))
    return df


def download_ghcnd_precip(station_id: str, year: int) -> pd.DataFrame | None:
    """Download daily PRCP for one station from GHCN-D."""
    url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station_id}.csv.gz"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.read_csv(
            io.BytesIO(r.content), compression="gzip",
            names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"],
            dtype={"date": str},
        )
        prcp = df[(df["element"] == "PRCP") & (df["date"].str.startswith(str(year)))]
        if len(prcp) == 0:
            return None
        prcp = prcp.copy()
        prcp["date"] = pd.to_datetime(prcp["date"], format="%Y%m%d")
        prcp["rain_mm"] = prcp["value"] / 10.0  # tenths of mm → mm
        # Quality flag: exclude flagged data
        prcp = prcp[prcp["qflag"].isna() | (prcp["qflag"] == " ")]
        return prcp[["date", "rain_mm"]].set_index("date")
    except Exception:
        return None


def download_ghcnd_batch(stations: pd.DataFrame, year: int,
                         max_stations: int = 500) -> pd.DataFrame:
    """Download PRCP for multiple stations, return long-format DataFrame."""
    records = []
    n = min(len(stations), max_stations)
    log.info("Downloading GHCN-D PRCP for %d stations, year %d", n, year)

    for i, (_, row) in enumerate(stations.head(n).iterrows()):
        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] %s", i + 1, n, row["id"])
        df = download_ghcnd_precip(row["id"], year)
        if df is not None and len(df) > 300:  # at least 300 days
            for date, rain in df["rain_mm"].items():
                records.append({
                    "station_id": row["id"],
                    "date": date,
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "rain_station": rain,
                })
        time.sleep(0.05)  # be nice to NOAA

    df = pd.DataFrame(records)
    log.info("Downloaded %d station-days from %d stations",
             len(df), df["station_id"].nunique())
    return df


# ── IMERG download via GEE ───────────────────────────────────────────────────

def download_imerg_at_stations(stations: pd.DataFrame, year: int) -> pd.DataFrame:
    """Download IMERG daily precipitation at station locations via GEE."""
    import ee
    ee.Initialize(project="ee-guillaumemaitrejean")

    log.info("Downloading IMERG at %d station locations for %d", len(stations), year)

    # Create station points
    unique_stations = stations[["station_id", "lat", "lon"]].drop_duplicates(subset="station_id")

    # Process in monthly chunks to avoid GEE timeouts
    all_results = []

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{month + 1:02d}-01"

        log.info("  IMERG %s to %s", start, end)

        # For each station, get monthly IMERG
        for _, st in unique_stations.iterrows():
            point = ee.Geometry.Point(float(st["lon"]), float(st["lat"]))

            col = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
                   .filterDate(start, end)
                   .select(["precipitation"]))

            # Get all half-hourly values, aggregate to daily
            data = col.getRegion(point, scale=11132).getInfo()
            if len(data) <= 1:
                continue

            header = data[0]
            daily = {}
            for row in data[1:]:
                d = dict(zip(header, row))
                dt = pd.Timestamp(d["time"], unit="ms")
                day = dt.strftime("%Y-%m-%d")
                precip = max(d.get("precipitation", 0) or 0, 0)
                daily.setdefault(day, []).append(precip * 0.5)  # mm/hr × 0.5h

            for day, vals in daily.items():
                all_results.append({
                    "station_id": st["station_id"],
                    "date": pd.Timestamp(day),
                    "rain_imerg": sum(vals),
                })

    df = pd.DataFrame(all_results)
    log.info("IMERG: %d station-days", len(df))
    return df


# ── Terrain features ─────────────────────────────────────────────────────────

def extract_terrain_features(stations: pd.DataFrame) -> pd.DataFrame:
    """Extract terrain features at station locations via GEE SRTM."""
    import ee
    ee.Initialize(project="ee-guillaumemaitrejean")

    log.info("Extracting terrain features for %d stations", len(stations))

    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)

    unique = stations[["station_id", "lat", "lon"]].drop_duplicates(subset="station_id")
    results = []

    # Batch query
    for i, (_, st) in enumerate(unique.iterrows()):
        point = ee.Geometry.Point(float(st["lon"]), float(st["lat"]))
        region = point.buffer(5000)  # 5km radius for TPI

        elev = srtm.reduceRegion(ee.Reducer.first(), point, 30).getInfo()
        elev_mean = srtm.reduceRegion(ee.Reducer.mean(), region, 90).getInfo()
        slp = slope.reduceRegion(ee.Reducer.first(), point, 30).getInfo()
        asp = aspect.reduceRegion(ee.Reducer.first(), point, 30).getInfo()

        results.append({
            "station_id": st["station_id"],
            "elevation": elev.get("elevation", 0),
            "elevation_mean_5km": elev_mean.get("elevation", 0),
            "slope": slp.get("slope", 0),
            "aspect": asp.get("aspect", 0),
            "tpi": (elev.get("elevation", 0) or 0) - (elev_mean.get("elevation", 0) or 0),
        })

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] terrain features", i + 1, len(unique))

    return pd.DataFrame(results)


# ── Train XGBoost ────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, output_dir: Path):
    """Train XGBoost precip correction model with spatial cross-validation."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_squared_error
    except ImportError:
        log.error("xgboost and scikit-learn required: pip install xgboost scikit-learn")
        return

    feature_cols = [
        "rain_imerg", "elevation", "slope", "aspect_sin", "aspect_cos",
        "tpi", "lat", "lon", "month", "month_sin", "month_cos",
    ]

    # Feature engineering
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect"]))

    df = df.dropna(subset=feature_cols + ["rain_station"])

    X = df[feature_cols].values
    y = df["rain_station"].values
    groups = df["station_id"].values

    log.info("Training XGBoost: %d samples, %d features, %d stations",
             len(X), len(feature_cols), len(np.unique(groups)))

    # Spatial 5-fold CV (each fold = different stations)
    gkf = GroupKFold(n_splits=5)
    rmses = []
    biases = []
    models = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1,
            objective="reg:squarederror", random_state=42,
        )
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[test_idx], y[test_idx])],
                  verbose=False)
        y_pred = model.predict(X[test_idx])

        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        bias = np.mean(y_pred - y[test_idx])
        rmses.append(rmse)
        biases.append(bias)
        models.append(model)

        n_test_stations = len(np.unique(groups[test_idx]))
        log.info("  Fold %d: RMSE=%.2f mm, bias=%+.2f mm (%d test stations)",
                 fold, rmse, bias, n_test_stations)

    log.info("Mean RMSE: %.2f ± %.2f mm", np.mean(rmses), np.std(rmses))
    log.info("Mean bias: %+.2f mm", np.mean(biases))

    # Retrain on all data
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1,
        objective="reg:squarederror", random_state=42,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    log.info("Feature importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        log.info("  %s: %.3f", feat, imp)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(output_dir / "precip_correction_xgb.json"))

    metrics = {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_bias_mean": float(np.mean(biases)),
        "n_samples": len(X),
        "n_stations": int(len(np.unique(groups))),
        "n_features": len(feature_cols),
        "feature_importance": {k: float(v) for k, v in importances.items()},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model saved to %s", output_dir)
    return final_model, metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--year", type=int, default=2022, help="Year to train on")
@click.option("--max-stations", type=int, default=500,
              help="Max stations to download (for testing)")
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("data/models/precip_correction"),
              help="Output directory for model and metrics")
@click.option("--skip-download", is_flag=True, help="Skip download, use cached data")
def main(year: int, max_stations: int, output: Path, skip_download: bool):
    """Train precipitation bias-correction model (IMERG → station quality)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    cache_dir = Path("data/raw/precip_correction_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        # Step 1: Get station inventory
        stations = download_ghcnd_inventory()
        stations.to_csv(cache_dir / "ghcnd_stations.csv", index=False)

        # Step 2: Download station precip
        station_data = download_ghcnd_batch(stations, year, max_stations=max_stations)
        station_data.to_parquet(cache_dir / f"ghcnd_precip_{year}.parquet")

        # Step 3: Download IMERG at station locations
        imerg_data = download_imerg_at_stations(station_data, year)
        imerg_data.to_parquet(cache_dir / f"imerg_at_stations_{year}.parquet")

        # Step 4: Terrain features
        terrain = extract_terrain_features(station_data)
        terrain.to_parquet(cache_dir / "terrain_features.parquet")
    else:
        station_data = pd.read_parquet(cache_dir / f"ghcnd_precip_{year}.parquet")
        imerg_data = pd.read_parquet(cache_dir / f"imerg_at_stations_{year}.parquet")
        terrain = pd.read_parquet(cache_dir / "terrain_features.parquet")

    # Merge
    df = station_data.merge(imerg_data, on=["station_id", "date"], how="inner")
    df = df.merge(terrain, on="station_id", how="left")

    log.info("Merged dataset: %d rows, %d stations",
             len(df), df["station_id"].nunique())

    # Step 5: Train
    model, metrics = train_model(df, output)

    print(f"\nDone. Model: {output}")
    print(f"  CV RMSE: {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f} mm")
    print(f"  Stations: {metrics['n_stations']}")


if __name__ == "__main__":
    main()
