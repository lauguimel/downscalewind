"""
Train the DownscalRain CNN patch-to-point precipitation model.

Example:
    cd services/module3-precip
    python train_downscalrain.py --config configs/downscalrain_cnn.yaml
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.downscalrain import (
    DownscalRainCNN,
    DownscalRainLossConfig,
    downscalrain_loss,
    precipitation_metrics,
    predict_rain_mm,
)
from src.patch_dataset import (
    RainPatchDataset,
    compute_stats,
    load_split_manifest,
    save_split_manifest,
    station_group_split,
)

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _worker_count(cfg: dict[str, Any]) -> int:
    if "num_workers" in cfg:
        return int(cfg["num_workers"])
    return 0


def _train_one_epoch(
    model: DownscalRainCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: DownscalRainLossConfig,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        patch = batch["patch"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        rain = batch["rain"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(patch, meta if model.meta_dim else None)
        loss, parts = downscalrain_loss(outputs, rain, loss_cfg)
        loss.backward()
        optimizer.step()

        bs = int(rain.shape[0])
        n += bs
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * bs

    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def _evaluate(
    model: DownscalRainCNN,
    loader: DataLoader,
    device: torch.device,
    wet_threshold_mm: float,
    heavy_threshold_mm: float,
) -> dict[str, float]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch in loader:
        patch = batch["patch"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        rain = batch["rain"].to(device, non_blocking=True)
        pred = predict_rain_mm(model(patch, meta if model.meta_dim else None))
        y_true.append(rain.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
    return precipitation_metrics(
        np.concatenate(y_true),
        np.clip(np.concatenate(y_pred), 0.0, None),
        wet_threshold_mm=wet_threshold_mm,
        heavy_threshold_mm=heavy_threshold_mm,
    )


def _save_checkpoint(
    path: Path,
    model: DownscalRainCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
    dataset: RainPatchDataset,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
        "in_channels": model.in_channels,
        "meta_dim": model.meta_dim,
        "width": model.width,
        "depths": model.depths,
        "dropout": float(cfg["model"].get("dropout", 0.05)),
        "channels": dataset.channels,
        "meta_columns": dataset.meta_columns,
    }
    torch.save(ckpt, path)


@click.command()
@click.option("--config", "config_path", default="configs/downscalrain_cnn.yaml", type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", default=None, type=click.Path(exists=True))
@click.option("--output", "output_dir", default=None, type=click.Path())
@click.option("--device", "device_name", default=None, help="auto, cpu, cuda, mps")
@click.option("--epochs", default=None, type=int)
def main(
    config_path: str,
    dataset_path: str | None,
    output_dir: str | None,
    device_name: str | None,
    epochs: int | None,
) -> None:
    t0 = time.perf_counter()
    cfg = _load_config(config_path)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    loss_cfg_raw = cfg.get("loss", {})

    if dataset_path is not None:
        cfg["patch_dataset"]["output_dir"] = dataset_path
    if output_dir is not None:
        cfg["output"]["dir"] = output_dir
    if device_name is not None:
        train_cfg["device"] = device_name
    if epochs is not None:
        train_cfg["epochs"] = epochs

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / "config.yaml")

    base_ds = RainPatchDataset(cfg["patch_dataset"]["output_dir"], mmap=True)
    split_path = Path(cfg["output"].get("split_manifest", out_dir / "split_manifest.json"))
    if split_path.exists():
        split = load_split_manifest(split_path)
        log.info("Loaded split manifest: %s", split_path)
    else:
        split = station_group_split(
            base_ds.station_ids,
            val_fraction=float(train_cfg.get("val_fraction", 0.15)),
            test_fraction=float(train_cfg.get("test_fraction", 0.15)),
            seed=int(train_cfg.get("seed", 42)),
        )
        save_split_manifest(split, split_path)
        log.info("Wrote split manifest: %s", split_path)

    stats_path = out_dir / "norm_stats.json"
    stats = compute_stats(base_ds, split["train"])
    stats.save(stats_path)
    log.info("Wrote normalization stats: %s", stats_path)

    train_ds = base_ds.subset(split["train"], stats=stats)
    val_ds = base_ds.subset(split["val"], stats=stats)
    test_ds = base_ds.subset(split["test"], stats=stats)
    log.info(
        "Dataset: train=%d val=%d test=%d channels=%d meta=%d patch=%s",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        base_ds.n_channels,
        base_ds.meta_dim,
        base_ds.patch_size,
    )

    generator = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)))
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 128)),
        "num_workers": _worker_count(train_cfg),
        "pin_memory": bool(train_cfg.get("pin_memory", False)),
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = _device(str(train_cfg.get("device", "auto")))
    model = DownscalRainCNN(
        in_channels=base_ds.n_channels,
        meta_dim=base_ds.meta_dim,
        width=int(model_cfg.get("width", 32)),
        depths=tuple(int(v) for v in model_cfg.get("depths", [2, 2, 2])),
        dropout=float(model_cfg.get("dropout", 0.05)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg.get("epochs", 50))),
    )
    loss_cfg = DownscalRainLossConfig(**loss_cfg_raw)

    history_path = out_dir / "history.csv"
    best_metric = float("inf")
    best_metrics: dict[str, float] = {}
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_occurrence_loss",
                "train_amount_loss",
                "train_dry_amount_loss",
                "val_rmse",
                "val_mae",
                "val_bias",
                "val_correlation",
                "val_wet_precision",
                "val_wet_recall",
                "val_dry_false_alarm",
                "val_heavy_recall",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
            train_parts = _train_one_epoch(model, train_loader, optimizer, device, loss_cfg)
            val_metrics = _evaluate(
                model,
                val_loader,
                device,
                wet_threshold_mm=float(loss_cfg.wet_threshold_mm),
                heavy_threshold_mm=float(loss_cfg.heavy_rain_threshold_mm),
            )
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_parts.get("loss", 0.0),
                "train_occurrence_loss": train_parts.get("occurrence_loss", 0.0),
                "train_amount_loss": train_parts.get("amount_loss", 0.0),
                "train_dry_amount_loss": train_parts.get("dry_amount_loss", 0.0),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()

            log.info(
                "Epoch %03d: loss=%.4f val_rmse=%.3f val_mae=%.3f wet_recall=%.3f",
                epoch,
                row["train_loss"],
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["wet_recall"],
            )
            if val_metrics["rmse"] < best_metric:
                best_metric = val_metrics["rmse"]
                best_metrics = dict(val_metrics)
                _save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, val_metrics, base_ds)

    test_metrics = _evaluate(
        model,
        test_loader,
        device,
        wet_threshold_mm=float(loss_cfg.wet_threshold_mm),
        heavy_threshold_mm=float(loss_cfg.heavy_rain_threshold_mm),
    )
    metrics = {
        "best_val": best_metrics,
        "test_last": test_metrics,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_checkpoint(out_dir / "last.pt", model, optimizer, int(train_cfg.get("epochs", 50)), cfg, test_metrics, base_ds)

    click.echo("\n" + "=" * 64)
    click.echo("DownscalRain CNN training complete")
    click.echo(f"  output:        {out_dir}")
    click.echo(f"  best val RMSE: {best_metric:.3f} mm/day")
    click.echo(f"  test RMSE:     {test_metrics['rmse']:.3f} mm/day")
    click.echo(f"  test MAE:      {test_metrics['mae']:.3f} mm/day")
    click.echo("=" * 64)


if __name__ == "__main__":
    main()
