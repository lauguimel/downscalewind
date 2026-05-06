from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from src.downscalrain import DownscalRainCNN, downscalrain_loss, precipitation_metrics, predict_rain_mm
from src.patch_dataset import RainPatchDataset, compute_stats, station_group_split, write_patch_dataset


def test_patch_dataset_stats_and_station_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 18
    patches = rng.normal(size=(n, 3, 8, 8)).astype(np.float32)
    rain = rng.gamma(shape=1.5, scale=2.0, size=n).astype(np.float32)
    meta = rng.normal(size=(n, 2)).astype(np.float32)
    station_ids = np.array([f"S{i // 3:02d}" for i in range(n)], dtype=object)
    dates = np.array([f"2022-01-{i + 1:02d}" for i in range(n)], dtype=object)

    write_patch_dataset(
        tmp_path,
        patches=patches,
        rain=rain,
        meta=meta,
        station_ids=station_ids,
        dates=dates,
        channels=["imerg", "era5land", "elevation"],
        meta_columns=["lat", "lon"],
    )

    ds = RainPatchDataset(tmp_path)
    split = station_group_split(ds.station_ids, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert set(split) == {"train", "val", "test"}
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == n

    stats = compute_stats(ds, split["train"])
    train = ds.subset(split["train"], stats=stats)
    item = train[0]
    assert item["patch"].shape == (3, 8, 8)
    assert item["meta"].shape == (2,)
    assert torch.isfinite(item["patch"]).all()


def test_downscalrain_forward_loss_and_metrics() -> None:
    torch.manual_seed(0)
    model = DownscalRainCNN(in_channels=4, meta_dim=2, width=8, depths=(1, 1), dropout=0.0)
    patch = torch.randn(5, 4, 16, 16)
    meta = torch.randn(5, 2)
    rain = torch.tensor([0.0, 0.3, 1.2, 12.0, 4.5])

    out = model(patch, meta)
    pred = predict_rain_mm(out)
    loss, parts = downscalrain_loss(out, rain)

    assert pred.shape == (5,)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    assert parts["loss"] > 0

    metrics = precipitation_metrics(rain.numpy(), pred.detach().numpy())
    assert {"rmse", "mae", "bias", "wet_recall", "heavy_recall"} <= set(metrics)
