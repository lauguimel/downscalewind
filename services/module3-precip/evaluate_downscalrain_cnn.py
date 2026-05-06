"""
Evaluate DownscalRain CNN predictions against station rain and raw gridded inputs.

Example:
    cd services/module3-precip
    python evaluate_downscalrain_cnn.py \
        --dataset ../../data/processed/downscalrain/patches_gee_2022 \
        --predictions ../../data/validation/downscalrain_cnn_gee_2022/predictions.parquet \
        --split-manifest ../../data/models/downscalrain_cnn_gee_2022/split_manifest.json \
        --output-dir ../../data/validation/downscalrain_cnn_gee_2022
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
from src.patch_dataset import RainPatchDataset, load_split_manifest

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _center_channel(ds: RainPatchDataset, channel: str) -> np.ndarray:
    if channel not in ds.channels:
        return np.full(len(ds.rain), np.nan, dtype=np.float32)
    idx = ds.channels.index(channel)
    h = ds.patches.shape[2] // 2
    w = ds.patches.shape[3] // 2
    return np.asarray(ds.patches[:, idx, h, w], dtype=np.float32)


def _meta_column(ds: RainPatchDataset, name: str) -> np.ndarray:
    if name not in ds.meta_columns:
        return np.full(len(ds.rain), np.nan, dtype=np.float32)
    idx = ds.meta_columns.index(name)
    return np.asarray(ds.meta[:, idx], dtype=np.float32)


def _dataset_frame(dataset_path: str | Path, split_manifest: str | Path | None) -> pd.DataFrame:
    ds = RainPatchDataset(dataset_path, mmap=True)
    df = pd.DataFrame({
        "row_index": np.arange(len(ds.rain), dtype=np.int64),
        "station_id": ds.station_ids.astype(str),
        "date": pd.to_datetime(ds.dates.astype(str)),
        "lat": _meta_column(ds, "lat"),
        "lon": _meta_column(ds, "lon"),
        "rain_station": np.asarray(ds.rain, dtype=np.float32),
        "rain_imerg_center": _center_channel(ds, "imerg_d0"),
        "rain_era5land_center": _center_channel(ds, "era5land_d0"),
        "elevation": _meta_column(ds, "elevation"),
    })
    df["month"] = df["date"].dt.month
    df["is_fire_season"] = df["month"].between(5, 10)
    df["split"] = "all"

    if split_manifest:
        split = load_split_manifest(split_manifest)
        df["split"] = "unassigned"
        for name, indices in split.items():
            df.loc[np.asarray(indices, dtype=np.int64), "split"] = name

    return df


def _load_predictions(path: str | Path) -> pd.DataFrame:
    pred = pd.read_parquet(path)
    required = {"station_id", "date", "rain_pred_mm"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    columns = ["station_id", "date", "rain_pred_mm"]
    for optional in ("wet_probability", "conditional_amount_mm"):
        if optional in pred.columns:
            columns.append(optional)
    out = pred[columns].copy()
    out["station_id"] = out["station_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _metric_rows(df: pd.DataFrame, subset: str, group: str = "global") -> list[dict[str, Any]]:
    rows = []
    models = [
        ("raw_imerg_center", "rain_imerg_center"),
        ("raw_era5land_center", "rain_era5land_center"),
        ("downscalrain_cnn", "rain_pred_mm"),
    ]
    for model, column in models:
        valid = df[["rain_station", column]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        metrics = precipitation_metrics(
            valid["rain_station"].to_numpy(),
            valid[column].clip(lower=0.0).to_numpy(),
            wet_threshold_mm=1.0,
            heavy_threshold_mm=10.0,
        )
        rows.append({
            "subset": subset,
            "group": group,
            "model": model,
            "n": int(len(valid)),
            **metrics,
        })
    return rows


def _summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_name, subset in [("all", df), ("test", df[df["split"] == "test"])]:
        if subset.empty:
            continue
        rows.extend(_metric_rows(subset, subset_name))
        fire = subset[subset["is_fire_season"]]
        if len(fire):
            rows.extend(_metric_rows(fire, f"{subset_name}_fire_season"))

        bins = [-np.inf, 200.0, 800.0, np.inf]
        labels = ["elev_low_lt200m", "elev_mid_200_800m", "elev_high_gt800m"]
        grouped = subset.copy()
        grouped["elevation_band"] = pd.cut(grouped["elevation"], bins=bins, labels=labels)
        for band, group in grouped.groupby("elevation_band", observed=True):
            if len(group):
                rows.extend(_metric_rows(group, subset_name, str(band)))

    test = df[df["split"] == "test"]
    if len(test):
        for month, group in test.groupby("month"):
            rows.extend(_metric_rows(group, f"test_month_{int(month):02d}", "monthly"))

    return pd.DataFrame(rows)


def _write_report(metrics: pd.DataFrame, path: Path) -> None:
    lines = ["# DownscalRain CNN validation", ""]
    main = metrics[(metrics["group"] == "global") & (metrics["subset"].isin(["all", "test", "test_fire_season"]))]
    for subset in ["all", "test", "test_fire_season"]:
        block = main[main["subset"] == subset]
        if block.empty:
            continue
        lines.append(f"## {subset}")
        lines.append("")
        lines.append("| model | n | RMSE | MAE | bias | corr | wet recall | heavy recall |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in block.to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {row['n']} | {row['rmse']:.3f} | {row['mae']:.3f} | "
                f"{row['bias']:.3f} | {row['correlation']:.3f} | {row['wet_recall']:.3f} | "
                f"{row['heavy_recall']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


@click.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--predictions", "predictions_path", required=True, type=click.Path(exists=True))
@click.option("--split-manifest", default=None, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
def main(dataset_path: str, predictions_path: str, split_manifest: str | None, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _dataset_frame(dataset_path, split_manifest)
    pred = _load_predictions(predictions_path)
    df = base.merge(pred, on=["station_id", "date"], how="left", validate="one_to_one")
    missing = int(df["rain_pred_mm"].isna().sum())
    if missing:
        raise ValueError(f"{missing} dataset rows have no CNN prediction")

    metrics = _summary_metrics(df)
    metrics_path = out_dir / "downscalrain_cnn_metrics.csv"
    enriched_path = out_dir / "downscalrain_cnn_predictions_enriched.parquet"
    report_path = out_dir / "report.md"
    metrics.to_csv(metrics_path, index=False)
    df.to_parquet(enriched_path, index=False)
    _write_report(metrics, report_path)

    payload = {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "n_samples": int(len(df)),
        "n_stations": int(df["station_id"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "metrics": metrics.to_dict(orient="records"),
    }
    (out_dir / "downscalrain_cnn_summary.json").write_text(json.dumps(payload, indent=2))

    click.echo("\n" + "=" * 72)
    click.echo("DownscalRain CNN validation")
    click.echo(f"  samples: {len(df)}")
    click.echo(f"  stations: {df['station_id'].nunique()}")
    click.echo(f"  metrics: {metrics_path}")
    main_rows = metrics[(metrics["group"] == "global") & (metrics["subset"] == "test")]
    if main_rows.empty:
        main_rows = metrics[(metrics["group"] == "global") & (metrics["subset"] == "all")]
    for row in main_rows.to_dict(orient="records"):
        click.echo(
            f"  {row['model']:20s} RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} "
            f"bias={row['bias']:.3f} wet_recall={row['wet_recall']:.3f}"
        )
    click.echo("=" * 72)


if __name__ == "__main__":
    main()
