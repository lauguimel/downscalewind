"""
compute_norm_stats_v2.py — Welford mean/std of v2 grid.zarr fields on TRAIN split.

Reads cases listed in `dataset_v2_splits.yaml` (train) from `<data_dir>/<site>_<case>/grid.zarr`
and writes `dataset_v2_norm.yaml` next to the splits.

Usage
-----
    python compute_norm_stats_v2.py \\
        --data-dir   /scratch/maitreje/dsw/training_v2 \\
        --splits-yaml /scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_splits.yaml \\
        --output     /scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_norm.yaml \\
        --max-cases 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import zarr

logger = logging.getLogger(__name__)


class Welford:
    """Streaming mean / std on flat arrays (per-channel)."""
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: np.ndarray) -> None:
        x = x.ravel().astype(np.float64)
        n_new = x.size
        if n_new == 0:
            return
        delta = x.mean() - self.mean
        new_mean = self.mean + delta * n_new / (self.n + n_new)
        # Use sum of squared diffs from running mean
        m2_new = ((x - x.mean()) ** 2).sum()
        self.M2 += m2_new + delta ** 2 * self.n * n_new / (self.n + n_new)
        self.mean = new_mean
        self.n += n_new

    @property
    def std(self) -> float:
        return float(np.sqrt(self.M2 / max(self.n - 1, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits-yaml", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    with open(args.splits_yaml) as f:
        splits = yaml.safe_load(f)
    train_sites = splits.get("train", [])

    cases: list[Path] = []
    for sid in train_sites:
        for case_dir in sorted(args.data_dir.glob(f"{sid}_case_ts*")):
            if (case_dir / "grid.zarr").exists():
                cases.append(case_dir)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    logger.info("Welford on %d train cases", len(cases))

    fields = {
        "U_x": Welford(), "U_y": Welford(), "U_z": Welford(),
        "T":   Welford(), "q":   Welford(),
        "terrain": Welford(), "z": Welford(), "agl": Welford(),
        "era5_u": Welford(), "era5_v": Welford(),
        "era5_T": Welford(), "era5_q": Welford(),
        "t2m": Welford(), "d2m": Welford(),
        "u10": Welford(), "v10": Welford(),
        "pressure": Welford(),
    }

    for k, case_dir in enumerate(cases):
        if k % 100 == 0:
            logger.info("[%d/%d] %s", k + 1, len(cases), case_dir.name)
        try:
            g = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")
            U = np.asarray(g["target/U"][:], dtype=np.float32)
            fields["U_x"].update(U[..., 0])
            fields["U_y"].update(U[..., 1])
            fields["U_z"].update(U[..., 2])
            fields["T"].update(np.asarray(g["target/T"][:], dtype=np.float32))
            fields["q"].update(np.asarray(g["target/q"][:], dtype=np.float32))

            terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
            z = np.asarray(g["coords/z"][:], dtype=np.float32)
            agl = z - terrain[:, :, None]
            fields["terrain"].update(terrain)
            fields["z"].update(z)
            fields["agl"].update(agl)

            for v in ("u", "v", "T", "q"):
                arr = np.asarray(g[f"input/era5_3d/{v}"][:], dtype=np.float32)
                fields[f"era5_{v}"].update(arr)
            for v in ("t2m", "d2m", "u10", "v10"):
                arr = np.asarray(g[f"input/era5_surface/{v}"][:], dtype=np.float32)
                fields[v].update(arr)
            plev = np.asarray(g["input/era5_pressure_levels"][:], dtype=np.float32)
            fields["pressure"].update(plev)
        except Exception as e:
            logger.warning("skip %s: %s", case_dir.name, e)
            continue

    out = {
        "schema_version": "v2.0",
        "n_train_cases": len(cases),
        "stats": {
            name: {"mean": float(w.mean), "std": float(w.std), "n": int(w.n)}
            for name, w in fields.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote %s", args.output)
    for name, w in fields.items():
        print(f"  {name:10s} mean={w.mean:+.4g}  std={w.std:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
