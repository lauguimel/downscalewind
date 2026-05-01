"""
train_v2.py — Baseline FNO3D training on the campaign-v2 native-grid dataset.

Reads `WindV2Dataset` (180×180×40 grid) and trains an `FNO3D` to predict
(u, v, w, T, q) from (terrain, AGL, z, z0, lat, ERA5 1D + surface, pressure).

Usage on Aqua (GPU node)
------------------------
    qsub configs/hpc/train_v2_baseline.pbs

Or locally for debugging
------------------------
    python train_v2.py \\
        --data-dir /scratch/maitreje/dsw/training_v2 \\
        --splits-yaml /scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_splits.yaml \\
        --norm-yaml   /scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_norm.yaml \\
        --epochs 5 --batch-size 2 --device cuda
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.dataset_v2 import WindV2Dataset, DEFAULT_NORM
from src.model_fno3d import FNO3D

logger = logging.getLogger(__name__)


def _load_norm_overrides(path: Path | None) -> dict:
    """Map dataset_v2_norm.yaml stats to DEFAULT_NORM keys (mean/std → offset/scale)."""
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    s = raw["stats"]
    n: dict = {}
    if "U_x" in s: n["U_uv_scale"] = max(s["U_x"]["std"], 1e-3)
    if "U_z" in s: n["U_w_scale"] = max(s["U_z"]["std"], 1e-3)
    if "T" in s:   n["T_offset"], n["T_scale"] = s["T"]["mean"], max(s["T"]["std"], 1e-3)
    if "q" in s:   n["q_scale"] = max(s["q"]["std"], 1e-6)
    if "terrain" in s: n["terrain_scale"] = max(s["terrain"]["std"], 1.0)
    if "z" in s:       n["z_scale"]       = max(s["z"]["std"], 1.0)
    if "era5_u" in s:  n["era5_u_scale"]  = max(s["era5_u"]["std"], 1.0)
    if "era5_v" in s:  n["era5_v_scale"]  = max(s["era5_v"]["std"], 1.0)
    if "era5_T" in s:  n["era5_T_offset"], n["era5_T_scale"] = (
        s["era5_T"]["mean"], max(s["era5_T"]["std"], 1.0))
    if "era5_q" in s:  n["era5_q_scale"]  = max(s["era5_q"]["std"], 1e-6)
    if "t2m" in s:     n["t2m_offset"], n["t2m_scale"] = s["t2m"]["mean"], max(s["t2m"]["std"], 1.0)
    if "d2m" in s:     n["d2m_offset"], n["d2m_scale"] = s["d2m"]["mean"], max(s["d2m"]["std"], 1.0)
    if "u10" in s:     n["u10_scale"]     = max(s["u10"]["std"], 1.0)
    if "v10" in s:     n["v10_scale"]     = max(s["v10"]["std"], 1.0)
    if "pressure" in s:
        n["pressure_offset"], n["pressure_scale"] = (
            s["pressure"]["mean"], max(s["pressure"]["std"], 1.0))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-yaml", type=Path, required=True)
    ap.add_argument("--norm-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/models/surrogate_v2"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, nargs=3, default=(16, 16, 8))
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-cases", type=int, default=None,
                    help="Subsample N train cases (smoke test).")
    ap.add_argument("--max-val-cases", type=int, default=None,
                    help="Subsample N val cases (smoke test).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    norm = {**DEFAULT_NORM, **_load_norm_overrides(args.norm_yaml)}
    logger.info("Norm: %s", {k: round(v, 4) for k, v in norm.items()})

    train_ds = WindV2Dataset(args.data_dir, args.splits_yaml, "train", norm=norm)
    val_ds = WindV2Dataset(args.data_dir, args.splits_yaml, "val", norm=norm)

    if args.max_train_cases is not None:
        train_ds.cases = train_ds.cases[: args.max_train_cases]
        logger.info("Truncated train to %d cases (smoke).", len(train_ds))
    if args.max_val_cases is not None:
        val_ds.cases = val_ds.cases[: args.max_val_cases]
        logger.info("Truncated val to %d cases (smoke).", len(val_ds))

    # Detect input channel count from a sample
    sample_inp, _, _ = train_ds[0]
    c_in = sample_inp.shape[0]
    logger.info("Input channels: %d  | grid (Ny, Nx, Nz) = %s", c_in, tuple(sample_inp.shape[1:]))

    model = FNO3D(
        in_channels=c_in, out_channels=5,
        width=args.width, modes=tuple(args.modes), n_layers=args.n_layers,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("FNO3D params: %.2f M", n_params / 1e6)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for step, (inp, tgt, _) in enumerate(train_loader):
            inp = inp.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)
            pred = model(inp)
            loss = F.mse_loss(pred, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            if step % 50 == 0:
                logger.info("ep %d step %d/%d  loss=%.5f", ep, step, len(train_loader), loss.item())
        train_loss /= max(len(train_loader), 1)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, _ in val_loader:
                inp = inp.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                pred = model(inp)
                val_loss += F.mse_loss(pred, tgt).item()
        val_loss /= max(len(val_loader), 1)
        wall = time.time() - t0
        logger.info("EP %d  train=%.5f  val=%.5f  (%.0fs)", ep, train_loss, val_loss, wall)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "wall_s": wall})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": ep, "val_loss": val_loss, "config": vars(args)},
                       args.out_dir / "best.pt")
            logger.info("  ↳ saved best.pt")

    (args.out_dir / "history.yaml").write_text(yaml.dump(history))
    logger.info("done. best val_loss=%.5f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
