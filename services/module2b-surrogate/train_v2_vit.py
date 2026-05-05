"""
train_v2_vit.py — Train TerrainViT_V2_S3 on the campaign-v2 native-grid dataset.

Supports:
  --loss-type {mse, charbonnier, s4}
      mse:         simple F.mse_loss
      charbonnier: sqrt((p-t)^2 + eps^2) — robust to outliers (FuXi-CFD)
      s4:          charbonnier + amplitude(spectral) + divergence(physics)
  --resume <path>   reload model from checkpoint (continue training)
  --warmup-epochs   linear warmup before cosine decay (S4 recipe)
  --use-geo         inject native z/AGL channels into the vertical heads
  --use-residual    learn CFD minus a simple ERA5-lifted baseline
  --agl-weight-*    boost near-ground pointwise data terms
  --loss-central-crop-km
                    compute loss only on the central crop while keeping full input context
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

from src.dataset_v2 import DEFAULT_NORM
from src.dataset_v2_vit import WindV2DatasetViT
from src.losses_v2 import mse_loss, total_loss
from src.model_vit_v2 import build_vit_v2

logger = logging.getLogger(__name__)


# ── Norm overrides (Welford → DEFAULT_NORM keys) ─────────────────────────────

def _load_norm_overrides(path):
    if path is None or not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text())
    s = raw["stats"]
    n = {}
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


# ── LR schedule (S4 recipe) ──────────────────────────────────────────────────

def make_warmup_cosine(optim, warmup_epochs: int, total_epochs: int):
    def lr_fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        # cosine 1.0 → 0.05 over [warmup, total)
        progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optim, lr_fn)


def crop_center_xy(
    x: torch.Tensor | None,
    crop_km: float | None,
    dx_m: float = 33.333,
) -> torch.Tensor | None:
    """Crop tensor on the y/x axes while preserving batch, channel, and z axes."""
    if x is None or crop_km is None or crop_km <= 0:
        return x
    ny, nx = x.shape[-3], x.shape[-2]
    cells = max(1, int(round(crop_km * 1000.0 / dx_m)))
    cells_y = min(cells, ny)
    cells_x = min(cells, nx)
    y0 = (ny - cells_y) // 2
    x0 = (nx - cells_x) // 2
    return x[..., y0:y0 + cells_y, x0:x0 + cells_x, :]


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
    ap.add_argument("--use-geo", action="store_true",
                    help="Inject native z/AGL channels in the ViT vertical head.")
    ap.add_argument("--include-slopes", action="store_true",
                    help="Add terrain slope_x/slope_y channels to the 2D terrain encoder.")
    ap.add_argument("--use-residual", action="store_true",
                    help="Train on CFD minus ERA5-lifted baseline instead of absolute fields.")
    ap.add_argument("--agl-weight-alpha", type=float, default=0.0,
                    help="Near-ground loss boost: weight=1+alpha*exp(-AGL/H).")
    ap.add_argument("--agl-weight-height", type=float, default=300.0,
                    help="E-folding height H in metres for AGL loss weighting.")
    ap.add_argument("--loss-central-crop-km", type=float, default=None,
                    help="Compute loss only on a central square crop, e.g. 2 for 2x2 km.")
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

    use_weight = args.agl_weight_alpha > 0.0
    ds_kwargs = dict(
        norm=norm,
        include_slopes=args.include_slopes,
        return_geo=args.use_geo,
        use_residual=args.use_residual,
        return_weight=use_weight,
        agl_weight_alpha=args.agl_weight_alpha,
        agl_weight_height=args.agl_weight_height,
    )
    train_ds = WindV2DatasetViT(args.data_dir, args.splits_yaml, "train", **ds_kwargs)
    val_ds = WindV2DatasetViT(args.data_dir, args.splits_yaml, "val", **ds_kwargs)
    if args.max_train_cases is not None:
        train_ds.cases = train_ds.cases[: args.max_train_cases]
    if args.max_val_cases is not None:
        val_ds.cases = val_ds.cases[: args.max_val_cases]

    sample = train_ds[0]
    terrain, era5 = sample[0], sample[1]
    geo = sample[2] if args.use_geo else None
    target_idx = 3 if args.use_geo else 2
    target = sample[target_idx]
    era5_dim = era5.shape[0]
    nz = target.shape[-1]
    geo_channels = geo.shape[0] if geo is not None else 0
    logger.info("terrain %s | era5 %s | geo_ch=%d | target %s",
                terrain.shape, era5.shape, geo_channels, target.shape)
    logger.info("options: residual=%s slopes=%s agl_weight_alpha=%.2f H=%.1f crop_km=%s",
                args.use_residual, args.include_slopes,
                args.agl_weight_alpha, args.agl_weight_height,
                args.loss_central_crop_km)

    model = build_vit_v2(preset=args.preset, era5_input_dim=era5_dim, nz=nz,
                         terrain_in_channels=terrain.shape[0],
                         geo_channels=geo_channels).to(args.device)
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

    def unpack_batch(batch):
        i = 0
        terrain_b = batch[i]; i += 1
        era5_b = batch[i]; i += 1
        geo_b = None
        if args.use_geo:
            geo_b = batch[i]; i += 1
        tgt_b = batch[i]; i += 1
        weight_b = None
        if use_weight:
            weight_b = batch[i]; i += 1
        return terrain_b, era5_b, geo_b, tgt_b, weight_b

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val, history = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        loss_components_acc: dict[str, float] = {}
        for step, batch in enumerate(train_loader):
            terrain, era5, geo, tgt, weight = unpack_batch(batch)
            terrain = terrain.to(args.device, non_blocking=True)
            era5 = era5.to(args.device, non_blocking=True)
            if geo is not None:
                geo = geo.to(args.device, non_blocking=True)
            tgt = tgt.to(args.device, non_blocking=True)
            if weight is not None:
                weight = weight.to(args.device, non_blocking=True)
            pred = model(terrain, era5, geo)
            pred_loss = crop_center_xy(pred, args.loss_central_crop_km)
            tgt_loss = crop_center_xy(tgt, args.loss_central_crop_km)
            weight_loss = crop_center_xy(weight, args.loss_central_crop_km)
            loss, comp = total_loss(pred_loss, tgt_loss, args.loss_type, weight=weight_loss,
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
                terrain, era5, geo, tgt, weight = unpack_batch(batch)
                terrain = terrain.to(args.device, non_blocking=True)
                era5 = era5.to(args.device, non_blocking=True)
                if geo is not None:
                    geo = geo.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                if weight is not None:
                    weight = weight.to(args.device, non_blocking=True)
                pred = model(terrain, era5, geo)
                pred_loss = crop_center_xy(pred, args.loss_central_crop_km)
                tgt_loss = crop_center_xy(tgt, args.loss_central_crop_km)
                weight_loss = crop_center_xy(weight, args.loss_central_crop_km)
                l, _ = total_loss(pred_loss, tgt_loss, args.loss_type, weight=weight_loss,
                                  w_amp=args.w_amp, w_div=args.w_div)
                val_loss += l.item()
                val_mse += mse_loss(pred_loss, tgt_loss).item()
                val_mse_agl += (
                    mse_loss(pred_loss, tgt_loss, weight_loss).item()
                    if weight_loss is not None else 0.0
                )
        val_loss /= max(len(val_loader), 1)
        val_mse /= max(len(val_loader), 1)
        if use_weight:
            val_mse_agl /= max(len(val_loader), 1)
        wall = time.time() - t0
        logger.info("EP %d  train=%.5f val=%.5f val_mse=%.5f val_mse_agl=%.5f comp=%s lr=%.2e (%.0fs)",
                    ep, train_loss, val_loss, val_mse, val_mse_agl, loss_components_acc,
                    optim.param_groups[0]["lr"], wall)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                        "val_mse": val_mse, "lr": optim.param_groups[0]["lr"],
                        "val_mse_agl": val_mse_agl if use_weight else None,
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
