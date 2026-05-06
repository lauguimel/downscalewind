"""
Plot DownscalRain tabular validation metrics.

Example:
    python services/module3-precip/plot_downscalrain_metrics.py \
        --metrics data/validation/downscalrain/downscalrain_tabular_metrics.csv \
        --predictions data/validation/downscalrain/downscalrain_tabular_predictions.parquet \
        --output-dir data/validation/downscalrain/figures
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "raw_imerg": "IMERG raw",
    "tabular_occ_amount": "DownscalRain tabular",
    "tabular_occ_amount_balanced": "DownscalRain tabular",
}

MODEL_COLORS = {
    "raw_imerg": "#777777",
    "tabular_occ_amount": "#0B6E69",
    "tabular_occ_amount_balanced": "#0B6E69",
}

MODEL_ORDER = ["raw_imerg", "tabular_occ_amount_balanced"]


def _setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_global(metrics: pd.DataFrame, out_dir: Path) -> None:
    subset = metrics[(metrics["group"] == "global") & (metrics["subset"].isin(["all", "fire_season"]))]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, metric, ylabel in zip(
        axes,
        ["rmse", "mae", "wet_recall"],
        ["RMSE (mm/day)", "MAE (mm/day)", "Wet-day recall"],
    ):
        x = np.arange(2)
        width = 0.34
        for j, model in enumerate(MODEL_ORDER):
            values = []
            for s in ["all", "fire_season"]:
                row = subset[(subset["subset"] == s) & (subset["model"] == model)].iloc[0]
                values.append(float(row[metric]))
            ax.bar(x + (j - 0.5) * width, values, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_xticks(x, ["All year", "Fire season"])
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper right")
    _save(fig, out_dir / "downscalrain_global_metrics.png")


def plot_elevation(metrics: pd.DataFrame, out_dir: Path) -> None:
    bands = ["elev_low_lt200m", "elev_mid_200_800m", "elev_high_gt800m"]
    labels = ["<200 m", "200-800 m", ">800 m"]
    subset = metrics[(metrics["subset"] == "all") & (metrics["group"].isin(bands))]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, metric, ylabel in zip(axes, ["rmse", "mae"], ["RMSE (mm/day)", "MAE (mm/day)"]):
        x = np.arange(len(bands))
        width = 0.34
        for j, model in enumerate(MODEL_ORDER):
            values = [
                float(subset[(subset["group"] == b) & (subset["model"] == model)].iloc[0][metric])
                for b in bands
            ]
            ax.bar(x + (j - 0.5) * width, values, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper right")
    _save(fig, out_dir / "downscalrain_elevation_metrics.png")


def plot_monthly(metrics: pd.DataFrame, out_dir: Path) -> None:
    subset = metrics[metrics["subset"].str.startswith("month_") & (metrics["group"] == "monthly")].copy()
    subset["month"] = subset["subset"].str[-2:].astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for model in MODEL_ORDER:
        rows = subset[subset["model"] == model].sort_values("month")
        ax.plot(rows["month"], rows["rmse"], marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE (mm/day)")
    ax.set_xticks(range(1, 13))
    ax.legend(loc="upper right")
    _save(fig, out_dir / "downscalrain_monthly_rmse.png")


def plot_scatter(pred: pd.DataFrame, out_dir: Path, max_points: int = 60000) -> None:
    if len(pred) > max_points:
        pred = pred.sample(max_points, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True)
    vmax = float(np.nanpercentile(pred[["rain_station", "rain_imerg", "rain_pred_tabular"]].to_numpy(), 99.5))
    vmax = max(vmax, 15.0)
    for ax, col, title in [
        (axes[0], "rain_imerg", "IMERG raw"),
        (axes[1], "rain_pred_tabular", "DownscalRain tabular"),
    ]:
        ax.hexbin(
            pred["rain_station"],
            pred[col],
            gridsize=55,
            extent=(0, vmax, 0, vmax),
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.plot([0, vmax], [0, vmax], color="black", lw=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Observed rain (mm/day)")
    axes[0].set_ylabel("Predicted rain (mm/day)")
    _save(fig, out_dir / "downscalrain_scatter_hexbin.png")


@click.command()
@click.option("--metrics", "metrics_path", required=True, type=click.Path(exists=True))
@click.option("--predictions", "predictions_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
def main(metrics_path: str, predictions_path: str, output_dir: str) -> None:
    _setup()
    out_dir = Path(output_dir)
    metrics = pd.read_csv(metrics_path)
    pred = pd.read_parquet(predictions_path)
    plot_global(metrics, out_dir)
    plot_elevation(metrics, out_dir)
    plot_monthly(metrics, out_dir)
    plot_scatter(pred, out_dir)
    click.echo(f"Wrote DownscalRain figures to {out_dir}")


if __name__ == "__main__":
    main()
