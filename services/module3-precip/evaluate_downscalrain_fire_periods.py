"""
Targeted DownscalRain validation for fire-weather precipitation use cases.

The global daily-rain score mixes winter/spring rainfall regimes with the FWI
use case. This script focuses on dry-season and false-rain "halo" situations:
station is dry, but the gridded product predicts rain.
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

sys.path.insert(0, str(Path(__file__).parent))

from src.downscalrain import precipitation_metrics

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

MODELS = [
    ("raw_imerg_center", "rain_imerg_center"),
    ("raw_era5land_center", "rain_era5land_center"),
    ("downscalrain_cnn", "rain_pred_mm"),
]


def _load_frame(predictions_path: str | Path, station_table: str | Path) -> pd.DataFrame:
    pred = pd.read_parquet(predictions_path).copy()
    stations = pd.read_parquet(station_table)[["station_id", "lat", "lon"]].drop_duplicates()
    df = pred.merge(stations, on="station_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["rain_station"] = df["rain_station"].clip(lower=0.0)
    return df


def _add_puechabon_distance(df: pd.DataFrame) -> pd.DataFrame:
    # ICOS FR-Pue / Puéchabon approximate tower coordinates.
    lat0, lon0 = 43.741, 3.595
    radius_earth_km = 6371.0
    dlat = np.deg2rad(df["lat"] - lat0)
    dlon = np.deg2rad(df["lon"] - lon0)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.deg2rad(lat0)) * np.cos(np.deg2rad(df["lat"])) * np.sin(dlon / 2.0) ** 2
    )
    out = df.copy()
    out["dist_puechabon_km"] = 2.0 * radius_earth_km * np.arcsin(np.sqrt(a))
    return out


def _model_metrics(
    df: pd.DataFrame,
    subset: str,
    group: str,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
    note: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dry = df["rain_station"] <= dry_threshold_mm
    for model, column in MODELS:
        valid = df[["rain_station", column]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        metrics = precipitation_metrics(
            valid["rain_station"].to_numpy(),
            valid[column].clip(lower=0.0).to_numpy(),
            wet_threshold_mm=wet_threshold_mm,
            heavy_threshold_mm=10.0,
        )
        dry_valid = valid["rain_station"] <= dry_threshold_mm
        if dry_valid.any():
            dry_pred = valid.loc[dry_valid, column].clip(lower=0.0)
            false_wet_rate = float((dry_pred > wet_threshold_mm).mean())
            dry_pred_mean = float(dry_pred.mean())
            dry_pred_sum = float(dry_pred.sum())
        else:
            false_wet_rate = np.nan
            dry_pred_mean = np.nan
            dry_pred_sum = np.nan

        rows.append({
            "subset": subset,
            "group": group,
            "model": model,
            "n": int(len(valid)),
            "n_dry": int(dry.sum()),
            "false_wet_rate_on_dry": false_wet_rate,
            "mean_pred_on_dry_mm": dry_pred_mean,
            "sum_pred_on_dry_mm": dry_pred_sum,
            "note": note,
            **metrics,
        })
    return rows


def _subset_definitions(df: pd.DataFrame, dry_threshold_mm: float, wet_threshold_mm: float) -> list[tuple[str, str, pd.Series, str]]:
    test = df["split"] == "test"
    jja = df["month"].isin([6, 7, 8])
    jjas = df["month"].isin([6, 7, 8, 9])
    med = df["lat"].between(35.0, 45.5) & df["lon"].between(-10.0, 25.0)
    dry = df["rain_station"] <= dry_threshold_mm
    imerg_halo = dry & (df["rain_imerg_center"] > wet_threshold_mm)
    era5_halo = dry & (df["rain_era5land_center"] > wet_threshold_mm)
    pue100 = df["dist_puechabon_km"] <= 100.0
    pue150 = df["dist_puechabon_km"] <= 150.0

    return [
        ("test_jjas_mediterranean", "fire_window", test & jjas & med, "Held-out stations, Mediterranean 35-45.5N, Jun-Sep."),
        ("test_jja_mediterranean", "fire_window", test & jja & med, "Held-out stations, Mediterranean 35-45.5N, Jun-Aug."),
        ("test_jjas_mediterranean_dry", "dry_fire_window", test & jjas & med & dry, "Station dry days inside the held-out Mediterranean fire window."),
        ("test_jjas_mediterranean_imerg_halo", "halo", test & jjas & med & imerg_halo, "Station dry, IMERG wet: direct false-rain halo subset."),
        ("test_jjas_mediterranean_era5_halo", "halo", test & jjas & med & era5_halo, "Station dry, ERA5-Land wet: direct false-rain halo subset."),
        ("test_puechabon_150km_jjas", "puechabon", test & pue150 & jjas, "Held-out stations within 150 km of Puéchabon, Jun-Sep."),
        ("test_puechabon_150km_jjas_dry", "puechabon", test & pue150 & jjas & dry, "Held-out dry station-days within 150 km of Puéchabon, Jun-Sep."),
        ("puechabon_100km_jjas_all", "puechabon_diagnostic", pue100 & jjas, "Diagnostic only: nearest stations are train/val, not held-out."),
        ("puechabon_100km_jjas_imerg_halo_all", "puechabon_diagnostic", pue100 & jjas & imerg_halo, "Diagnostic only: Puéchabon-near IMERG false-rain halo days."),
        ("puechabon_100km_jjas_era5_halo_all", "puechabon_diagnostic", pue100 & jjas & era5_halo, "Diagnostic only: Puéchabon-near ERA5-Land false-rain halo days."),
    ]


def _station_table(df: pd.DataFrame, output_dir: Path) -> None:
    stations = (
        df[df["dist_puechabon_km"] <= 250.0][["station_id", "lat", "lon", "split", "dist_puechabon_km"]]
        .drop_duplicates()
        .sort_values("dist_puechabon_km")
    )
    stations.to_csv(output_dir / "puechabon_nearby_stations.csv", index=False)


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['model']} | {row['n']} | {row['n_dry']} | {row['rmse']:.3f} | {row['mae']:.3f} | "
        f"{row['bias']:.3f} | {row['correlation']:.3f} | {row['false_wet_rate_on_dry']:.3f} | "
        f"{row['mean_pred_on_dry_mm']:.3f} | {row['wet_recall']:.3f} | {row['heavy_recall']:.3f} |"
    )


def _write_report(metrics: pd.DataFrame, output_dir: Path, dry_threshold_mm: float, wet_threshold_mm: float) -> None:
    lines = [
        "# DownscalRain Fire-Period Precipitation Validation",
        "",
        f"Dry threshold: station rain <= {dry_threshold_mm:g} mm/day.",
        f"Wet threshold: model rain > {wet_threshold_mm:g} mm/day.",
        "",
        "These subsets target FWI failure modes: dry-season false precipitation and gridded-product halos.",
        "",
    ]

    order = [
        "test_jjas_mediterranean",
        "test_jjas_mediterranean_dry",
        "test_jjas_mediterranean_imerg_halo",
        "test_jjas_mediterranean_era5_halo",
        "test_puechabon_150km_jjas",
        "test_puechabon_150km_jjas_dry",
        "puechabon_100km_jjas_all",
        "puechabon_100km_jjas_imerg_halo_all",
        "puechabon_100km_jjas_era5_halo_all",
    ]
    for subset in order:
        block = metrics[metrics["subset"] == subset]
        if block.empty:
            continue
        note = str(block["note"].iloc[0])
        lines.extend([
            f"## {subset}",
            "",
            note,
            "",
            "| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in block.to_dict(orient="records"):
            lines.append(_format_row(row))
        lines.append("")

    (output_dir / "fire_period_report.md").write_text("\n".join(lines))


@click.command()
@click.option(
    "--predictions",
    "predictions_path",
    default="data/validation/downscalrain_cnn_gee_2022/downscalrain_cnn_predictions_enriched.parquet",
    type=click.Path(exists=True),
)
@click.option(
    "--station-table",
    default="data/raw/precip_correction_cache/dataset_2022.parquet",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
    default="data/validation/downscalrain_cnn_gee_2022",
    type=click.Path(),
)
@click.option("--dry-threshold-mm", default=1.0, show_default=True, type=float)
@click.option("--wet-threshold-mm", default=1.0, show_default=True, type=float)
def main(
    predictions_path: str,
    station_table: str,
    output_dir: str,
    dry_threshold_mm: float,
    wet_threshold_mm: float,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _add_puechabon_distance(_load_frame(predictions_path, station_table))

    rows: list[dict[str, Any]] = []
    for subset, group, mask, note in _subset_definitions(df, dry_threshold_mm, wet_threshold_mm):
        subset_df = df[mask].copy()
        if subset_df.empty:
            log.warning("Skipping empty subset %s", subset)
            continue
        rows.extend(_model_metrics(subset_df, subset, group, dry_threshold_mm, wet_threshold_mm, note))

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "fire_period_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    _station_table(df, out_dir)
    _write_report(metrics, out_dir, dry_threshold_mm, wet_threshold_mm)

    print("\n" + "=" * 76)
    print("DownscalRain fire-period validation")
    print(f"  metrics: {metrics_path}")
    key = metrics[metrics["subset"].isin([
        "test_jjas_mediterranean",
        "test_jjas_mediterranean_imerg_halo",
        "puechabon_100km_jjas_imerg_halo_all",
    ])]
    for row in key.to_dict(orient="records"):
        print(
            f"  {row['subset']:38s} {row['model']:20s} "
            f"RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} "
            f"false_wet/dry={row['false_wet_rate_on_dry']:.3f} "
            f"dry_mean={row['mean_pred_on_dry_mm']:.3f}"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()
