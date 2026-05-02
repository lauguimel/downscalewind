"""
train_v2_vit.py — Train TerrainViT_V2_S3 on the campaign-v2 native-grid dataset.

Supports:
  --loss-type {mse, charbonnier, s4}
      mse:         simple F.mse_loss
      charbonnier: sqrt((p-t)^2 + eps^2) — robust to outliers (FuXi-CFD)
      s4:          charbonnier + amplitude(spectral) + divergence(physics)
  --resume <path>   reload model from checkpoint (continue training)
  --warmup-epochs   linear warmup before cosine decay (S4 recipe)
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.dataset_v2 import DEFAULT_NORM
from src.dataset_v2_vit import WindV2DatasetViT
from src.model_vit_v2 import build_vit_v2

logger = logging.getLogger(__name__)


# ── Loss components ──────────────────────────────────────────────────────────

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral amplitude loss on z-mean 2D fields."""
    pred_2d = pred.mean(dim=-1)
    target_2d = target.mean(dim=-1)
    pred_amp = torch.fft.rfft2(pred_2d, dim=(-2, -1)).abs()
    target_amp = torch.fft.rfft2(target_2d, dim=(-2, -1)).abs()
    return (pred_amp - target_amp).abs().mean()


def divergence_loss(pred: torch.Tensor, dx: float = 33.333) -> torch.Tensor:
    """Soft div-free penalty assuming uniform vertical spacing (cheap approx)."""
    u, v, w = pred[:, 0], pred[:, 1], pred[:, 2]
    du_dx = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dx)
    dv_dy = (v[:, 2:, :, :] - v[:, :-2, :, :]) / (2.0 * dx)
    dw_dz = (w[:, :, :, 2:] - w[:, :, :, :-2]) / 2.0   # spacing absorbed in scale
    ny = min(du_dx.shape[1], dv_dy.shape[1])
    nx = min(du_dx.shape[2], dv_dy.shape[2])
    nz = min(du_dx.shape[3], dw_dz.shape[3])
    div = (du_dx[:, 1:ny+1, :nx, 1:nz+1]
           + dv_dy[:, :ny, 1:nx+1, 1:nz+1]
           + dw_dz[:, 1:ny+1, 1:nx+1, :nz])
    return div.pow(2).mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor, kind: str,
               w_amp: float = 0.1, w_div: float = 0.05) -> tuple[torch.Tensor, dict]:
    if kind == "mse":
        l = F.mse_loss(pred, target)
        return l, {"mse": l.item()}
    if kind == "charbonnier":
        l = charbonnier_loss(pred, target)
        return l, {"char": l.item()}
    if kind == "s4":
        l_c = charbonnier_loss(pred, target)
        l_a = amplitude_loss(pred, target)
        l_d = divergence_loss(pred)
        l = l_c + w_amp * l_a + w_div * l_d
        return l, {"char": l_c.item(), "amp": l_a.item(), "div": l_d.item()}
    raise ValueError(f"unknown loss-type {kind}")


# ── Norm overrides (Welford → DEFAULT_NORM keys) ─────────────────────────────

def _load_norm_overrides(path):
    if path is None or not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text())
    s = raw["stats"]
    n = {}
    if "U_x" in s: n["U_uv_scale"] = max(s["U_x"]["std"], 1e-3)
    if "U_z" in s: n["U_w_scale"] = max(s["U_z"]["std"], 1e-3)
    if "T" in s:   n["T_offset"], n["T_scale"] = s["T"]["mean"], max(s["T"]["std"], 1e-3)
    if "q" in s:   n["q_scale"] = max(s["q"]["std"], 1e-6)
    if "terrain" in s: n["terrain_scale"] = max(s["terrain"]["std"], 1.0)
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


# ── LR schedule (S4 recipe) ──────────────────────────────────────────────────

def make_warmup_cosine(optim, warmup_epochs: int, total_epochs: int):
    def lr_fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        # cosine 1.0 → 0.05 over [warmup, total)
        progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optim, lr_fn)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-yaml", type=Path, required=True)
    ap.add_argument("--norm-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/models/surrogate_v2_vit"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--preset", default="base", choices=["small", "base", "large"])
    ap.add_argument("--loss-type", default="mse", choices=["mse", "charbonnier", "s4"])
    ap.add_argument("--w-amp", type=float, default=0.1)
    ap.add_argument("--w-div", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-cases", type=int, default=None)
    ap.add_argument("--max-val-cases", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to a best.pt to resume from (loads model weights only).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    norm = {**DEFAULT_NORM, **_load_norm_overrides(args.norm_yaml)}

    train_ds = WindV2DatasetViT(args.data_dir, args.splits_yaml, "train", norm=norm)
    val_ds = WindV2DatasetViT(args.data_dir, args.splits_yaml, "val", norm=norm)
    if args.max_train_cases is not None:
        train_ds.cases = train_ds.cases[: args.max_train_cases]
    if args.max_val_cases is not None:
        val_ds.cases = val_ds.cases[: args.max_val_cases]

    terrain, era5, target, _ = train_ds[0]
    era5_dim = era5.shape[0]
    nz = target.shape[-1]
    logger.info("terrain %s | era5 %s | target %s", terrain.shape, era5.shape, target.shape)

    model = build_vit_v2(preset=args.preset, era5_input_dim=era5_dim, nz=nz).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("ViT params: %.2f M (preset=%s)", n_params / 1e6, args.preset)

    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        try:
            model.load_state_dict(ck["model"])
            logger.info("Resumed from %s (epoch=%d, val_loss=%.5f)",
                        args.resume, ck.get("epoch", -1), ck.get("val_loss", float("nan")))
        except Exception as e:
            logger.warning("Resume FAILED (%s) — starting fresh.", e)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        sched = make_warmup_cosine(optim, args.warmup_epochs, args.epochs)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val, history = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        loss_components_acc: dict[str, float] = {}
        for step, batch in enumerate(train_loader):
            terrain, era5, tgt, _ = batch
            terrain = terrain.to(args.device, non_blocking=True)
            era5 = era5.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)
            pred = model(terrain, era5)
            loss, comp = total_loss(pred, tgt, args.loss_type,
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
        with torch.no_grad():
            for batch in val_loader:
                terrain, era5, tgt, _ = batch
                terrain = terrain.to(args.device, non_blocking=True)
                era5 = era5.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                pred = model(terrain, era5)
                l, _ = total_loss(pred, tgt, args.loss_type,
                                  w_amp=args.w_amp, w_div=args.w_div)
                val_loss += l.item()
                val_mse += F.mse_loss(pred, tgt).item()
        val_loss /= max(len(val_loader), 1)
        val_mse /= max(len(val_loader), 1)
        wall = time.time() - t0
        logger.info("EP %d  train=%.5f val=%.5f val_mse=%.5f comp=%s lr=%.2e (%.0fs)",
                    ep, train_loss, val_loss, val_mse, loss_components_acc,
                    optim.param_groups[0]["lr"], wall)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mse": val_mse, "lr": optim.param_groups[0]["lr"],
                        "wall_s": wall, **{f"train_{k}": v for k, v in loss_components_acc.items()}})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_loss": val_loss, "val_mse": val_mse,
                        "config": vars(args)},
                       args.out_dir / "best.pt")

    (args.out_dir / "history.yaml").write_text(yaml.dump(history))
    logger.info("done. best val_loss=%.5f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
