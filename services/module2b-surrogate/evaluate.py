"""
evaluate.py — Evaluate trained models on the test split.

Computes metrics for each model and generates comparison plots.

Usage
-----
    cd services/module2b-surrogate
    python evaluate.py \
        --model     mlp \
        --weights   ../../data/models/module2b_poc/mlp/best_model.pt \
        --data-dir  ../../data/cfd-database/sf_poc/ \
        --dataset   ../../data/cfd-database/sf_poc/dataset.yaml \
        --output    ../../data/validation/sf_poc/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def evaluate_mlp(weights_path: str, data_dir: str, dataset_yaml: str) -> dict:
    """Evaluate MLP model on test split."""
    from src.dataset_sf import SFUnstructuredDataset
    from src.model_mlp import MLPSpeedup

    device = torch.device("cpu")  # evaluation on CPU for reproducibility
    model = MLPSpeedup()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = SFUnstructuredDataset(data_dir, dataset_yaml, split="test")

    all_pred = []
    all_true = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            pred = model(sample["node_features"], sample["global_features"])
            all_pred.append(pred.numpy())
            all_true.append(sample["target"].numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    return _compute_metrics(pred, true, "mlp")


def evaluate_unet(weights_path: str, data_dir: str, dataset_yaml: str) -> dict:
    """Evaluate U-Net model on test split."""
    from src.dataset_sf import SFGridDataset
    from src.model_unet3d import UNet3D

    device = torch.device("cpu")
    model = UNet3D(in_channels=3, out_channels=3, base_features=32)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = SFGridDataset(data_dir, dataset_yaml, split="test")

    all_pred = []
    all_true = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            pred = model(sample["input"].unsqueeze(0))
            all_pred.append(pred.squeeze(0).numpy())
            all_true.append(sample["target"].numpy())

    pred = np.concatenate([p.reshape(3, -1).T for p in all_pred], axis=0)
    true = np.concatenate([t.reshape(3, -1).T for t in all_true], axis=0)
    return _compute_metrics(pred, true, "unet")


def evaluate_gnn(weights_path: str, data_dir: str, dataset_yaml: str) -> dict:
    """Evaluate GNN model on test split."""
    from src.dataset_sf import SFUnstructuredDataset
    from src.model_gnn import GNNSpeedup

    device = torch.device("cpu")
    model = GNNSpeedup()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = SFUnstructuredDataset(data_dir, dataset_yaml, split="test")

    all_pred = []
    all_true = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            node_feat = sample["node_features"]
            pos = node_feat[:, :3]
            pred = model(node_feat, sample["global_features"], pos=pos)
            all_pred.append(pred.numpy())
            all_true.append(sample["target"].numpy())

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    return _compute_metrics(pred, true, "gnn")


def _compute_metrics(pred: np.ndarray, true: np.ndarray, model_name: str) -> dict:
    """Compute RMSE, MAE, correlation for u, v, w and wind speed."""
    results = {"model": model_name}

    # Per-component metrics
    for comp, name in enumerate(["u", "v", "w"]):
        p, t = pred[:, comp], true[:, comp]
        rmse = np.sqrt(np.mean((p - t) ** 2))
        mae = np.mean(np.abs(p - t))
        corr = np.corrcoef(p, t)[0, 1] if np.std(t) > 1e-8 else 0.0
        results[f"rmse_{name}"] = float(rmse)
        results[f"mae_{name}"] = float(mae)
        results[f"corr_{name}"] = float(corr)

    # Wind speed metrics
    speed_pred = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + pred[:, 2]**2)
    speed_true = np.sqrt(true[:, 0]**2 + true[:, 1]**2 + true[:, 2]**2)
    results["rmse_speed"] = float(np.sqrt(np.mean((speed_pred - speed_true) ** 2)))
    results["mae_speed"] = float(np.mean(np.abs(speed_pred - speed_true)))
    results["corr_speed"] = float(np.corrcoef(speed_pred, speed_true)[0, 1])

    # Baseline: predict ERA5 inflow everywhere (speed_true with no terrain effect)
    # Approximate: mean speed over all cells
    baseline_speed = np.mean(speed_true)
    rmse_baseline = np.sqrt(np.mean((baseline_speed - speed_true) ** 2))
    results["rmse_baseline"] = float(rmse_baseline)
    results["skill_score"] = float(1.0 - results["rmse_speed"] / max(rmse_baseline, 1e-8))

    logger.info(
        "%s: RMSE_speed=%.3f m/s, corr=%.3f, skill_score=%.3f (vs baseline RMSE=%.3f)",
        model_name, results["rmse_speed"], results["corr_speed"],
        results["skill_score"], rmse_baseline,
    )
    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate wind downscaling model")
    parser.add_argument("--model", required=True, choices=["mlp", "unet", "gnn"])
    parser.add_argument("--weights", required=True, help="Path to best_model.pt")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="Output directory for metrics + plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "mlp":
        metrics = evaluate_mlp(args.weights, args.data_dir, args.dataset)
    elif args.model == "unet":
        metrics = evaluate_unet(args.weights, args.data_dir, args.dataset)
    elif args.model == "gnn":
        metrics = evaluate_gnn(args.weights, args.data_dir, args.dataset)

    # Save metrics
    metrics_path = output_dir / f"metrics_{args.model}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)


if __name__ == "__main__":
    main()
