"""
Run a trained DownscalRain CNN on a patch dataset.

Example:
    cd services/module3-precip
    python predict_downscalrain.py \
        --checkpoint ../../data/models/downscalrain_cnn_v1/best.pt \
        --dataset ../../data/processed/downscalrain/patches_v1 \
        --output ../../data/processed/downscalrain/predictions.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.downscalrain import DownscalRainCNN, model_config_from_checkpoint, predict_rain_mm
from src.patch_dataset import PatchDatasetStats, RainPatchDataset

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict(model: DownscalRainCNN, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        patch = batch["patch"].to(device)
        meta = batch["meta"].to(device)
        outputs = model(patch, meta if model.meta_dim else None)
        rain_pred = predict_rain_mm(outputs).detach().cpu().numpy()
        wet_prob = torch.sigmoid(outputs["wet_logit"]).detach().cpu().numpy()
        amount_mm = torch.expm1(torch.nn.functional.softplus(outputs["log_amount"])).detach().cpu().numpy()
        rain_true = batch["rain"].numpy()
        for i in range(len(rain_pred)):
            rows.append(
                {
                    "station_id": batch["station_id"][i],
                    "date": batch["date"][i],
                    "rain_true_mm": float(rain_true[i]),
                    "rain_pred_mm": float(max(rain_pred[i], 0.0)),
                    "wet_probability": float(wet_prob[i]),
                    "conditional_amount_mm": float(max(amount_mm[i], 0.0)),
                }
            )
    return pd.DataFrame(rows)


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--stats", "stats_path", default=None, type=click.Path(exists=True))
@click.option("--batch-size", default=256, show_default=True, type=int)
@click.option("--device", "device_name", default="auto", show_default=True)
def main(
    checkpoint: str,
    dataset_path: str,
    output_path: str,
    stats_path: str | None,
    batch_size: int,
    device_name: str,
) -> None:
    device = _device(device_name)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = PatchDatasetStats.load(stats_path) if stats_path else None
    if stats is None:
        default_stats = Path(checkpoint).parent / "norm_stats.json"
        if default_stats.exists():
            stats = PatchDatasetStats.load(default_stats)
            log.info("Loaded stats from %s", default_stats)
    if stats is None:
        raise ValueError("normalization stats are required; pass --stats or keep norm_stats.json next to checkpoint")

    dataset = RainPatchDataset(dataset_path, stats=stats)
    model = DownscalRainCNN(**model_config_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(ckpt["model_state"])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    df = _predict(model, loader, device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
    log.info("Wrote %d predictions to %s", len(df), output)


if __name__ == "__main__":
    main()
