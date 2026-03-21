"""
assemble_training_zarr.py — Assemble exported cases into a training dataset.

Reads individual case Zarr stores from export_sf_dataset.py and creates:
  1. A unified index (case_ids, splits, inflow metadata)
  2. Normalization statistics (mean/std per variable on train split)

Usage
-----
    cd services/module2a-cfd
    python assemble_training_zarr.py \
        --input  ../../data/cfd-database/sf_poc/ \
        --output ../../data/cfd-database/sf_poc/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def assemble_dataset(
    input_dir: Path,
    train_frac: float = 0.72,
    val_frac: float = 0.12,
    seed: int = 42,
) -> dict:
    """Scan exported cases and create dataset index with train/val/test splits.

    Splits are by timestamp (not by cell) to avoid data leakage:
    same terrain at different times stays within one split.
    """
    case_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("ts_") and (d / "unstructured.zarr").exists()
    ])

    if not case_dirs:
        raise FileNotFoundError(f"No exported cases found in {input_dir}")

    logger.info("Found %d exported cases", len(case_dirs))

    # Collect metadata
    records = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        inflow_path = case_dir / "inflow.json"

        meta = {"case_id": case_id}
        if inflow_path.exists():
            with open(inflow_path) as f:
                inflow = json.load(f)
            meta["u_hub"] = inflow.get("u_hub", 0)
            meta["wind_dir"] = inflow.get("wind_dir", 0)
            meta["T_ref"] = inflow.get("T_ref", 0)
            meta["Ri_b"] = inflow.get("Ri_b", 0)

        # Read n_cells from unstructured Zarr
        import zarr
        store = zarr.open_group(str(case_dir / "unstructured.zarr"), mode="r")
        meta["n_cells"] = int(store.attrs.get("n_cells", len(store["x"])))

        records.append(meta)

    df = pd.DataFrame(records)

    # Split by case index (random, reproducible)
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = np.full(n, "test", dtype="U5")
    splits[indices[:n_train]] = "train"
    splits[indices[n_train:n_train + n_val]] = "val"
    df["split"] = splits

    n_train_actual = (df["split"] == "train").sum()
    n_val_actual = (df["split"] == "val").sum()
    n_test_actual = (df["split"] == "test").sum()
    logger.info("Splits: %d train, %d val, %d test", n_train_actual, n_val_actual, n_test_actual)

    return df


def compute_norm_stats(input_dir: Path, df: pd.DataFrame) -> dict:
    """Compute mean/std for each variable on the train split (Welford online)."""
    import zarr

    train_cases = df[df["split"] == "train"]["case_id"].tolist()

    # Accumulate statistics with Welford's algorithm
    n_total = 0
    sums = {}
    sum_sqs = {}
    variables = ["x", "y", "z", "z_agl", "elev", "k", "nut"]
    vector_vars = ["U"]

    for case_id in train_cases:
        store = zarr.open_group(str(input_dir / case_id / "unstructured.zarr"), mode="r")
        n = len(store["x"])
        n_total += n

        for var in variables:
            if var in store:
                data = np.array(store[var][:], dtype=np.float64)
                sums[var] = sums.get(var, 0.0) + data.sum()
                sum_sqs[var] = sum_sqs.get(var, 0.0) + (data**2).sum()

        # Velocity components
        U = np.array(store["U"][:], dtype=np.float64)
        for comp, name in enumerate(["u", "v", "w"]):
            col = U[:, comp]
            sums[name] = sums.get(name, 0.0) + col.sum()
            sum_sqs[name] = sum_sqs.get(name, 0.0) + (col**2).sum()

    stats = {}
    for var in list(sums.keys()):
        mean = sums[var] / n_total
        var_val = sum_sqs[var] / n_total - mean**2
        std = max(np.sqrt(max(var_val, 0.0)), 1e-8)
        stats[var] = {"mean": float(mean), "std": float(std)}

    logger.info("Norm stats computed over %d cells from %d train cases", n_total, len(train_cases))
    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Assemble SF PoC training dataset")
    parser.add_argument("--input", required=True, help="Directory with exported cases")
    parser.add_argument("--output", required=True, help="Output dataset YAML path")
    parser.add_argument("--train-frac", type=float, default=0.72)
    parser.add_argument("--val-frac", type=float, default=0.12)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Assemble index
    df = assemble_dataset(input_dir, args.train_frac, args.val_frac)

    # Compute normalization stats
    norm_stats = compute_norm_stats(input_dir, df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "n_cases": len(df),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "data_dir": str(input_dir),
        "norm_stats": norm_stats,
        "cases": df.to_dict("records"),
    }
    with open(output_path, "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    # Also save the index as CSV
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    logger.info("Dataset assembled: %s (%d cases)", output_path, len(df))
    logger.info("Index CSV: %s", csv_path)


if __name__ == "__main__":
    main()
