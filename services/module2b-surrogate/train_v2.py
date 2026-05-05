"""
train_v2.py — Baseline FNO3D training on the campaign-v2 native-grid dataset.

Reads `WindV2Dataset` (180×180×40 grid) and trains an `FNO3D` to predict
(u, v, w, T, q) from (terrain, AGL, z, z0, lat, ERA5 1D + surface, pressure).
Optional flags support centred normalisation, residual ERA5 targets,
terrain slopes, S4 loss and AGL-weighted pointwise terms.

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
import math
import time
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.dataset_v2 import WindV2Dataset, DEFAULT_NORM
from src.losses_v2 import mse_loss, total_loss
from src.model_fno3d import FNO3D

logger = logging.getLogger(__name__)


def _load_norm_overrides(path: Path | None) -> dict:
    """Map dataset_v2_norm.yaml stats to DEFAULT_NORM keys (mean/std → offset/scale)."""
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    s = raw["stats"]
    n: dict = {}
    if "U_x" in s:
        n["U_x_offset"] = s["U_x"]["mean"]
        n["U_uv_scale"] = max(s["U_x"]["std"], 1e-3)
    if "U_y" in s:
        n["U_y_offset"] = s["U_y"]["mean"]
        n["U_uv_scale"] = max(n.get("U_uv_scale", 0.0), s["U_y"]["std"], 1e-3)
    if "U_z" in s:
        n["U_z_offset"] = s["U_z"]["mean"]
        n["U_w_scale"] = max(s["U_z"]["std"], 1e-3)
    if "T" in s:   n["T_offset"], n["T_scale"] = s["T"]["mean"], max(s["T"]["std"], 1e-3)
    if "q" in s:   n["q_offset"], n["q_scale"] = s["q"]["mean"], max(s["q"]["std"], 1e-6)
    if "terrain" in s: n["terrain_scale"] = max(s["terrain"]["std"], 1.0)
    if "z" in s:       n["z_scale"]       = max(s["z"]["std"], 1.0)
    if "agl" in s:     n["agl_scale"]     = max(s["agl"]["std"], 1.0)
    if "era5_u" in s:
        n["era5_u_offset"] = s["era5_u"]["mean"]
        n["era5_u_scale"] = max(s["era5_u"]["std"], 1.0)
    if "era5_v" in s:
        n["era5_v_offset"] = s["era5_v"]["mean"]
        n["era5_v_scale"] = max(s["era5_v"]["std"], 1.0)
    if "era5_T" in s:  n["era5_T_offset"], n["era5_T_scale"] = (
        s["era5_T"]["mean"], max(s["era5_T"]["std"], 1.0))
    if "era5_q" in s:
        n["era5_q_offset"] = s["era5_q"]["mean"]
        n["era5_q_scale"] = max(s["era5_q"]["std"], 1e-6)
    if "t2m" in s:     n["t2m_offset"], n["t2m_scale"] = s["t2m"]["mean"], max(s["t2m"]["std"], 1.0)
    if "d2m" in s:     n["d2m_offset"], n["d2m_scale"] = s["d2m"]["mean"], max(s["d2m"]["std"], 1.0)
    if "u10" in s:
        n["u10_offset"] = s["u10"]["mean"]
        n["u10_scale"] = max(s["u10"]["std"], 1.0)
    if "v10" in s:
        n["v10_offset"] = s["v10"]["mean"]
        n["v10_scale"] = max(s["v10"]["std"], 1.0)
    if "pressure" in s:
        n["pressure_offset"], n["pressure_scale"] = (
            s["pressure"]["mean"], max(s["pressure"]["std"], 1.0))
    return n


def make_warmup_cosine(optim, warmup_epochs: int, total_epochs: int):
    def lr_fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optim, lr_fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-yaml", type=Path, required=True)
    ap.add_argument("--norm-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/models/surrogate_v2"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, nargs=3, default=(16, 16, 8))
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--loss-type", default="mse", choices=["mse", "charbonnier", "s4"])
    ap.add_argument("--w-amp", type=float, default=0.1)
    ap.add_argument("--w-div", type=float, default=0.05)
    ap.add_argument("--include-slopes", action="store_true",
                    help="Add terrain slope_x/slope_y channels broadcast on the 3D grid.")
    ap.add_argument("--use-residual", action="store_true",
                    help="Train on CFD minus ERA5-lifted baseline instead of absolute fields.")
    ap.add_argument("--agl-weight-alpha", type=float, default=0.0,
                    help="Near-ground loss boost: weight=1+alpha*exp(-AGL/H).")
    ap.add_argument("--agl-weight-height", type=float, default=300.0,
                    help="E-folding height H in metres for AGL loss weighting.")
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
    use_weight = args.agl_weight_alpha > 0.0

    ds_kwargs = dict(
        norm=norm,
        include_slopes=args.include_slopes,
        use_residual=args.use_residual,
        return_weight=use_weight,
        agl_weight_alpha=args.agl_weight_alpha,
        agl_weight_height=args.agl_weight_height,
    )
    train_ds = WindV2Dataset(args.data_dir, args.splits_yaml, "train", **ds_kwargs)
    val_ds = WindV2Dataset(args.data_dir, args.splits_yaml, "val", **ds_kwargs)

    if args.max_train_cases is not None:
        train_ds.cases = train_ds.cases[: args.max_train_cases]
        logger.info("Truncated train to %d cases (smoke).", len(train_ds))
    if args.max_val_cases is not None:
        val_ds.cases = val_ds.cases[: args.max_val_cases]
        logger.info("Truncated val to %d cases (smoke).", len(val_ds))

    # Detect input channel count from a sample
    sample = train_ds[0]
    sample_inp = sample[0]
    c_in = sample_inp.shape[0]
    logger.info("Input channels: %d  | grid (Ny, Nx, Nz) = %s", c_in, tuple(sample_inp.shape[1:]))
    logger.info("options: loss=%s residual=%s slopes=%s agl_weight_alpha=%.2f H=%.1f",
                args.loss_type, args.use_residual, args.include_slopes,
                args.agl_weight_alpha, args.agl_weight_height)

    model = FNO3D(
        in_channels=c_in, out_channels=5,
        width=args.width, modes=tuple(args.modes), n_layers=args.n_layers,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("FNO3D params: %.2f M", n_params / 1e6)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        sched = make_warmup_cosine(optim, args.warmup_epochs, args.epochs)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    def unpack_batch(batch):
        if use_weight:
            inp_b, tgt_b, weight_b, _ = batch
            return inp_b, tgt_b, weight_b
        inp_b, tgt_b, _ = batch
        return inp_b, tgt_b, None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        loss_components_acc: dict[str, float] = {}
        for step, batch in enumerate(train_loader):
            inp, tgt, weight = unpack_batch(batch)
            inp = inp.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)
            if weight is not None:
                weight = weight.to(args.device, non_blocking=True)
            pred = model(inp)
            loss, comp = total_loss(pred, tgt, args.loss_type, weight=weight,
                                    w_amp=args.w_amp, w_div=args.w_div)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            for k, v in comp.items():
                loss_components_acc[k] = loss_components_acc.get(k, 0.0) + v
            if step % 50 == 0:
                logger.info("ep %d step %d/%d loss=%.5f comp=%s",
                            ep, step, len(train_loader), loss.item(),
                            {k: round(v, 5) for k, v in comp.items()})
        train_loss /= max(len(train_loader), 1)
        for k in loss_components_acc:
            loss_components_acc[k] /= max(len(train_loader), 1)
        sched.step()

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mse_agl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt, weight = unpack_batch(batch)
                inp = inp.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                if weight is not None:
                    weight = weight.to(args.device, non_blocking=True)
                pred = model(inp)
                l, _ = total_loss(pred, tgt, args.loss_type, weight=weight,
                                  w_amp=args.w_amp, w_div=args.w_div)
                val_loss += l.item()
                val_mse += mse_loss(pred, tgt).item()
                val_mse_agl += mse_loss(pred, tgt, weight).item() if weight is not None else 0.0
        val_loss /= max(len(val_loader), 1)
        val_mse /= max(len(val_loader), 1)
        if use_weight:
            val_mse_agl /= max(len(val_loader), 1)
        wall = time.time() - t0
        logger.info("EP %d train=%.5f val=%.5f val_mse=%.5f val_mse_agl=%.5f comp=%s lr=%.2e (%.0fs)",
                    ep, train_loss, val_loss, val_mse, val_mse_agl,
                    loss_components_acc, optim.param_groups[0]["lr"], wall)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mse": val_mse,
                        "val_mse_agl": val_mse_agl if use_weight else None,
                        "lr": optim.param_groups[0]["lr"], "wall_s": wall,
                        **{f"train_{k}": v for k, v in loss_components_acc.items()}})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": ep, "val_loss": val_loss,
                        "val_mse": val_mse, "config": vars(args)},
                       args.out_dir / "best.pt")
            logger.info("  ↳ saved best.pt")

    (args.out_dir / "history.yaml").write_text(yaml.dump(history))
    logger.info("done. best val_loss=%.5f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
