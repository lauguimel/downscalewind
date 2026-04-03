"""
train.py -- Train precipitation bias-correction model (IMERG -> station quality).

Downloads station observations, IMERG satellite data, terrain features,
then trains XGBoost with spatial cross-validation.

Usage:
    cd services/module3-precip
    python train.py --config configs/training.yaml
    python train.py --config configs/training.yaml --skip-download  # use cached data
    python train.py --config configs/training.yaml --max-stations 50  # quick test
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stations import load_all_stations
from src.imerg import download_imerg_batch
from src.terrain import extract_terrain_batch
from src.dataset import build_dataset
from src.model import train_cv, train_final, save_model

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _cache_path(cache_dir: Path, name: str) -> Path:
    """Build a cache file path and ensure the directory exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    type=click.Path(exists=True),
    help="Path to training YAML config.",
)
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip downloads, use cached data only.",
)
@click.option(
    "--max-stations",
    type=int,
    default=None,
    help="Override config max_stations (for quick testing).",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Override config year.",
)
def main(
    config_path: str,
    skip_download: bool,
    max_stations: int | None,
    year: int | None,
) -> None:
    """Train precipitation bias-correction model (IMERG -> station quality)."""
    t0 = time.perf_counter()
    cfg = _load_config(config_path)

    # CLI overrides
    if max_stations is not None:
        cfg["data"]["max_stations"] = max_stations
    if year is not None:
        cfg["data"]["year"] = year

    train_year = cfg["data"]["year"]
    cache_dir = Path(cfg["output"]["cache_dir"])
    output_dir = Path(cfg["output"]["dir"])
    gee_project = cfg["data"]["gee_project"]

    logger.info("Training config: year=%d, max_stations=%s", train_year, cfg["data"]["max_stations"])

    # ── Step 1: Station observations ─────────────────────────────────────────
    stations_cache = _cache_path(cache_dir, f"stations_precip_{train_year}.parquet")

    if skip_download and stations_cache.exists():
        import pandas as pd

        logger.info("Loading cached station data: %s", stations_cache)
        stations_df = pd.read_parquet(stations_cache)
    else:
        logger.info("Downloading station observations for %d...", train_year)
        stations_df = load_all_stations(
            year=train_year,
            bbox=cfg["data"]["bbox"],
            sources=cfg["stations"],
            max_stations=cfg["data"]["max_stations"],
            min_days=cfg["data"]["min_days_per_station"],
        )
        stations_df.to_parquet(stations_cache)
        logger.info("Saved %d station records to %s", len(stations_df), stations_cache)

    n_stations = stations_df["station_id"].nunique()
    logger.info("Stations loaded: %d unique stations, %d daily records", n_stations, len(stations_df))

    # ── Step 2: IMERG at station locations ───────────────────────────────────
    imerg_cache = _cache_path(cache_dir, f"imerg_at_stations_{train_year}.parquet")

    if skip_download and imerg_cache.exists():
        import pandas as pd

        logger.info("Loading cached IMERG data: %s", imerg_cache)
        imerg_df = pd.read_parquet(imerg_cache)
    else:
        logger.info("Downloading IMERG at %d station locations...", n_stations)
        unique_locs = stations_df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
        imerg_df = download_imerg_batch(
            stations_df=unique_locs,
            year=train_year,
            gee_project=gee_project,
        )
        imerg_df.to_parquet(imerg_cache)
        logger.info("Saved IMERG data to %s", imerg_cache)

    # ── Step 3: Terrain features ─────────────────────────────────────────────
    terrain_cache = _cache_path(cache_dir, "terrain_features.parquet")

    if skip_download and terrain_cache.exists():
        import pandas as pd

        logger.info("Loading cached terrain features: %s", terrain_cache)
        terrain_df = pd.read_parquet(terrain_cache)
    else:
        logger.info("Extracting terrain features via GEE...")
        unique_locs = stations_df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
        terrain_df = extract_terrain_batch(
            stations_df=unique_locs,
            gee_project=gee_project,
            cache_path=terrain_cache,
        )
        terrain_df.to_parquet(terrain_cache)
        logger.info("Saved terrain features to %s", terrain_cache)

    # ── Step 4: Build dataset ────────────────────────────────────────────────
    dataset_cache = _cache_path(cache_dir, f"dataset_{train_year}.parquet")

    logger.info("Building merged dataset...")
    dataset_df = build_dataset(
        station_data=stations_df,
        imerg_data=imerg_df,
        terrain_data=terrain_df,
    )
    dataset_df.to_parquet(dataset_cache)
    logger.info("Dataset: %d samples, %d features", len(dataset_df), dataset_df.shape[1])

    # ── Step 5: Spatial cross-validation ─────────────────────────────────────
    logger.info("Running %d-fold spatial cross-validation...", cfg["model"]["cv_folds"])
    cv_results = train_cv(
        df=dataset_df,
        n_folds=cfg["model"]["cv_folds"],
        params=cfg["model"],
    )

    logger.info(
        "CV results: RMSE=%.2f mm/day, bias=%.2f mm/day",
        cv_results.get("mean_rmse", 0),
        cv_results.get("mean_bias", 0),
    )

    # ── Step 6: Train final model ────────────────────────────────────────────
    logger.info("Training final model on all data...")
    model = train_final(
        df=dataset_df,
        params=cfg["model"],
    )

    # ── Step 7: Save model + metrics ─────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        model=model,
        metrics=cv_results,
        output_dir=output_dir,
    )
    logger.info("Model saved to %s", output_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    click.echo("\n" + "=" * 60)
    click.echo(f"Training complete in {elapsed:.0f}s")
    click.echo(f"  Stations:    {n_stations}")
    click.echo(f"  Samples:     {len(dataset_df)}")
    click.echo(f"  CV RMSE:     {cv_results['mean_rmse']:.2f} +/- {cv_results['std_rmse']:.2f} mm/day")
    click.echo(f"  CV bias:     {cv_results['mean_bias']:.2f} mm/day")
    click.echo(f"  Model saved: {output_dir}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
