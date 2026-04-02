"""
predict.py -- Predict bias-corrected precipitation at any location.

Apply a trained model to correct IMERG satellite precipitation using
terrain-aware bias correction.

Usage:
    python predict.py --model ../../data/models/precip_correction/ \\
        --lat 43.74 --lon 3.60 --start 2022-06-01 --end 2022-08-31 \\
        --output corrected_precip.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.imerg import download_imerg_batch
from src.terrain import extract_terrain_batch
from src.model import load_model

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@click.command()
@click.option(
    "--model",
    "model_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model directory.",
)
@click.option("--lat", required=True, type=float, help="Target latitude (degrees).")
@click.option("--lon", required=True, type=float, help="Target longitude (degrees).")
@click.option(
    "--start",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--gee-project",
    default=None,
    type=str,
    help="GEE project ID (default: from model config).",
)
@click.option(
    "--output",
    default="corrected_precip.csv",
    type=click.Path(),
    help="Output CSV path.",
)
def main(
    model_dir: str,
    lat: float,
    lon: float,
    start,
    end,
    gee_project: str | None,
    output: str,
) -> None:
    """Predict bias-corrected precipitation at a given location."""
    model_path = Path(model_dir)

    # ── Step 1: Load model ───────────────────────────────────────────────────
    logger.info("Loading model from %s", model_path)
    model, model_cfg = load_model(model_path)

    if gee_project is None:
        gee_project = model_cfg.get("data", {}).get("gee_project", "ee-guillaumemaitrejean")

    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    logger.info(
        "Predicting at (%.4f, %.4f) from %s to %s",
        lat, lon, start_date, end_date,
    )

    # ── Step 2: Download IMERG at target location ────────────────────────────
    logger.info("Downloading IMERG data via GEE...")
    target_df = pd.DataFrame({"station_id": ["target"], "lat": [lat], "lon": [lon]})
    imerg_df = download_imerg_batch(
        stations=target_df,
        year=None,  # use start/end dates instead
        gee_project=gee_project,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("IMERG: %d daily records downloaded", len(imerg_df))

    # ── Step 3: Extract terrain features ─────────────────────────────────────
    logger.info("Extracting terrain features via GEE...")
    terrain_df = extract_terrain_batch(
        stations_df=target_df,
        gee_project=gee_project,
    )

    # ── Step 4: Build feature matrix ─────────────────────────────────────────
    # Merge terrain (static) into daily IMERG records
    features_df = imerg_df.merge(terrain_df, on="station_id", how="left")

    # Add temporal features
    features_df["day_of_year"] = pd.to_datetime(features_df["date"]).dt.dayofyear
    features_df["month"] = pd.to_datetime(features_df["date"]).dt.month

    # Identify feature columns (exclude metadata)
    meta_cols = {"station_id", "date", "lat", "lon"}
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    logger.info("Features: %s", feature_cols)

    # ── Step 5: Predict ──────────────────────────────────────────────────────
    logger.info("Running prediction...")
    X = features_df[feature_cols].values
    rain_corrected = model.predict(X)

    # Ensure non-negative precipitation
    rain_corrected = np.maximum(rain_corrected, 0.0)

    # ── Step 6: Save results ─────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "date": features_df["date"].values,
        "rain_imerg": features_df["rain_imerg"].values if "rain_imerg" in features_df.columns else np.nan,
        "rain_corrected": rain_corrected,
    })
    result_df.to_csv(output, index=False)
    logger.info("Results saved to %s", output)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_days = len(result_df)
    total_imerg = result_df["rain_imerg"].sum()
    total_corrected = result_df["rain_corrected"].sum()
    n_wet_days = (result_df["rain_corrected"] > 0.1).sum()

    click.echo("\n" + "=" * 60)
    click.echo(f"Prediction complete: {n_days} days")
    click.echo(f"  Location:        ({lat:.4f}, {lon:.4f})")
    click.echo(f"  Period:          {start_date} to {end_date}")
    click.echo(f"  Total IMERG:     {total_imerg:.1f} mm")
    click.echo(f"  Total corrected: {total_corrected:.1f} mm")
    click.echo(f"  Bias (corr-raw): {total_corrected - total_imerg:+.1f} mm")
    click.echo(f"  Wet days (>0.1): {n_wet_days}")
    click.echo(f"  Output:          {output}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
