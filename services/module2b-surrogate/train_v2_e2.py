from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels
from src.dataset_v2_vit_e2 import WindV2DatasetViT_E2
from src.losses_v2 import mse_loss, total_loss
from src.model_vit_v2_e2 import build_vit_v2_e2
from train_v2_vit import _load_norm_overrides, crop_center_xy, make_warmup_cosine

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_toggle_monitor(model: torch.nn.Module, val_ds: WindV2DatasetViT_E2,
                       device: str, n_cases: int, use_geo: bool,
                       use_weight: bool) -> dict[str, float]:
    """Periodic toggle test: compare pred(mask=1) vs pred(mask=0) on a small val
    subset. Returns mean MSE_drop0, MSE_drop1, mean |Δpred|, and Δ = mse_drop1 -
    mse_drop0. Forces obs_dropout=0 on the temp dataset view to always feed a
    valid obs_value; the toggle is then driven by zeroing the mask.

    Lightweight (~20s for n_cases=20) — used as a proxy of OBS-channel
    activation during training.
    """
    was_training = model.training
    model.eval()
    saved_dropout = val_ds.obs_dropout
    val_ds.obs_dropout = 0.0
    mse0_list, mse1_list, diff_list = [], [], []
    try:
        n = min(int(n_cases), len(val_ds))
        for idx in range(n):
            sample = val_ds[idx]
            i = 0
            terrain = sample[i]; i += 1
            era5 = sample[i]; i += 1
            geo = None
            if use_geo:
                geo = sample[i]; i += 1
            target = sample[i]; i += 1
            if use_weight:
                i += 1
            obs_value = sample[i]; i += 1
            obs_mask = sample[i]; i += 1
            obs_ij = sample[i]; i += 1
            terrain_b = terrain.unsqueeze(0).to(device)
            era5_b = era5.unsqueeze(0).to(device)
            geo_b = None if geo is None else geo.unsqueeze(0).to(device)
            tgt_b = target.unsqueeze(0).to(device)
            ov_b = obs_value.unsqueeze(0).to(device)
            om_b = obs_mask.unsqueeze(0).to(device)
            oij_b = obs_ij.unsqueeze(0).to(device)
            pred0 = model(terrain_b, era5_b, geo_b, ov_b, om_b, oij_b)
            pred1 = model(terrain_b, era5_b, geo_b,
                          torch.zeros_like(ov_b), torch.zeros_like(om_b), oij_b)
            mse0_list.append(torch.mean((pred0 - tgt_b) ** 2).item())
            mse1_list.append(torch.mean((pred1 - tgt_b) ** 2).item())
            diff_list.append(torch.mean(torch.abs(pred0 - pred1)).item())
    finally:
        val_ds.obs_dropout = saved_dropout
        if was_training:
            model.train()
    mse0 = float(sum(mse0_list) / max(len(mse0_list), 1))
    mse1 = float(sum(mse1_list) / max(len(mse1_list), 1))
    diff = float(sum(diff_list) / max(len(diff_list), 1))
    return {"mse_drop0": mse0, "mse_drop1": mse1,
            "delta": mse1 - mse0, "mean_abs_pred_diff": diff,
            "n_cases": len(mse0_list)}


def load_resume_weights(model: torch.nn.Module, checkpoint: dict, *,
                        partial: bool = True) -> dict[str, int]:
    state = checkpoint["model"]
    if not partial:
        model.load_state_dict(state)
        return {"loaded": len(state), "skipped": 0, "checkpoint_skipped": 0}
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return {
        "loaded": len(compatible),
        "skipped": len(model_state) - len(compatible),
        "checkpoint_skipped": len(state) - len(compatible),
    }


def maybe_start_mlflow(out_dir: Path, args):
    try:
        import mlflow
        mlflow.set_tracking_uri("file://" + str(out_dir / ".." / "mlruns"))
        mlflow.set_experiment("surrogate_v2_e2_stage1_smoke")
        mlflow.start_run(run_name=out_dir.name)
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})
        return mlflow
    except Exception as exc:
        logger.warning("MLflow unavailable, using history.yaml only: %s", exc)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-yaml", type=Path, required=True)
    ap.add_argument("--norm-yaml", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/models/surrogate_v2_e2_stage1_smoke"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--preset", default="base", choices=["small", "base", "large"])
    ap.add_argument("--loss-type", default="mse", choices=["mse", "charbonnier", "s4"])
    ap.add_argument("--w-amp", type=float, default=0.1)
    ap.add_argument("--w-div", type=float, default=0.05)
    ap.add_argument("--use-geo", action="store_true")
    ap.add_argument("--include-slopes", action="store_true")
    ap.add_argument("--use-residual", action="store_true")
    ap.add_argument("--residual-baseline-mode", default="pressure_index",
                    choices=["pressure_index", "surface"])
    ap.add_argument("--agl-weight-alpha", type=float, default=0.0)
    ap.add_argument("--agl-weight-height", type=float, default=300.0)
    ap.add_argument("--loss-central-crop-km", type=float, default=None)
    ap.add_argument("--target-agl-levels", default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-cases", type=int, default=None)
    ap.add_argument("--max-val-cases", type=int, default=None)
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--obs-dropout", type=float, default=0.5)
    ap.add_argument("--obs-height-m", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--toggle-monitor", action="store_true",
                    help="Run periodic toggle test at --toggle-epochs milestones")
    ap.add_argument("--toggle-epochs", default="5,10,15,20,25,30",
                    help="Comma-separated 1-indexed epochs at which to run the toggle test")
    ap.add_argument("--toggle-n-cases", type=int, default=20)
    ap.add_argument("--toggle-warn-epoch", type=int, default=10,
                    help="Epoch at which we emit a stderr WARNING if Δ < toggle-warn-threshold")
    ap.add_argument("--toggle-warn-threshold", type=float, default=1e-4)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="Save checkpoint every N epochs (0 = best.pt only)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if args.smoke:
        args.max_train_cases = 8
        args.max_val_cases = 4
        args.epochs = 1
        args.batch_size = 2
        args.num_workers = 0

    args.data_dir = Path(os.path.realpath(args.data_dir))
    args.splits_yaml = Path(os.path.realpath(args.splits_yaml))
    if args.norm_yaml is not None:
        args.norm_yaml = Path(os.path.realpath(args.norm_yaml))
    args.resume = Path(os.path.realpath(args.resume))
    if not args.resume.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    norm = {**DEFAULT_NORM, **_load_norm_overrides(args.norm_yaml)}
    target_agl_levels = parse_agl_levels(args.target_agl_levels)
    args.target_agl_levels = (
        None if target_agl_levels is None
        else [float(x) for x in target_agl_levels.tolist()]
    )
    use_weight = args.agl_weight_alpha > 0.0
    ds_kwargs = dict(
        norm=norm, include_slopes=args.include_slopes, return_geo=args.use_geo,
        use_residual=args.use_residual, residual_baseline_mode=args.residual_baseline_mode,
        return_weight=use_weight, agl_weight_alpha=args.agl_weight_alpha,
        agl_weight_height=args.agl_weight_height, target_agl_levels=target_agl_levels,
        obs_dropout=args.obs_dropout, obs_height_m=args.obs_height_m,
    )
    train_ds = WindV2DatasetViT_E2(args.data_dir, args.splits_yaml, "train", **ds_kwargs)
    val_ds = WindV2DatasetViT_E2(args.data_dir, args.splits_yaml, "val", **ds_kwargs)
    if args.max_train_cases is not None:
        train_ds.cases = train_ds.cases[:args.max_train_cases]
    if args.max_val_cases is not None:
        val_ds.cases = val_ds.cases[:args.max_val_cases]

    sample = train_ds[0]
    terrain, era5 = sample[0], sample[1]
    geo = sample[2] if args.use_geo else None
    target = sample[3 if args.use_geo else 2]
    geo_channels = geo.shape[0] if geo is not None else 0
    logger.info("terrain %s | era5 %s | geo_ch=%d | target %s | obs_k=%d",
                terrain.shape, era5.shape, geo_channels, target.shape, train_ds.obs_k)

    model = build_vit_v2_e2(preset=args.preset, era5_input_dim=era5.shape[0],
                            nz=target.shape[-1], terrain_in_channels=terrain.shape[0],
                            geo_channels=geo_channels).to(args.device)
    ck = torch.load(args.resume, map_location=args.device, weights_only=False)
    load_stats = load_resume_weights(model, ck, partial=True)
    logger.info("Resumed from %s epoch=%d val_loss=%.5f loaded=%d skipped=%d checkpoint_skipped=%d",
                args.resume, ck.get("epoch", -1), ck.get("val_loss", float("nan")),
                load_stats["loaded"], load_stats["skipped"],
                load_stats["checkpoint_skipped"])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (make_warmup_cosine(optim, args.warmup_epochs, args.epochs)
             if args.warmup_epochs > 0
             else torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs))
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
        obs_value_b = batch[i]; i += 1
        obs_mask_b = batch[i]; i += 1
        obs_ij_b = batch[i]; i += 1
        return terrain_b, era5_b, geo_b, tgt_b, weight_b, obs_value_b, obs_mask_b, obs_ij_b

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlflow = maybe_start_mlflow(args.out_dir, args)
    best_val, history = float("inf"), []

    toggle_milestones: set[int] = set()
    if args.toggle_monitor:
        try:
            toggle_milestones = {int(x.strip()) for x in args.toggle_epochs.split(",")
                                 if x.strip()}
        except ValueError as exc:
            raise SystemExit(f"--toggle-epochs malformed: {exc}") from exc
        logger.info("toggle monitor enabled at epochs (1-indexed) %s, n_cases=%d, "
                    "warn at epoch %d if Δ<%.2e", sorted(toggle_milestones),
                    args.toggle_n_cases, args.toggle_warn_epoch, args.toggle_warn_threshold)
    try:
        for ep in range(args.epochs):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            comp_acc: dict[str, float] = {}
            for step, batch in enumerate(train_loader):
                terrain, era5, geo, tgt, weight, obs_value, obs_mask, obs_ij = unpack_batch(batch)
                terrain = terrain.to(args.device, non_blocking=True)
                era5 = era5.to(args.device, non_blocking=True)
                geo = None if geo is None else geo.to(args.device, non_blocking=True)
                tgt = tgt.to(args.device, non_blocking=True)
                weight = None if weight is None else weight.to(args.device, non_blocking=True)
                obs_value = obs_value.to(args.device, non_blocking=True)
                obs_mask = obs_mask.to(args.device, non_blocking=True)
                obs_ij = obs_ij.to(args.device, non_blocking=True)
                pred = model(terrain, era5, geo, obs_value, obs_mask, obs_ij)
                pred_loss = crop_center_xy(pred, args.loss_central_crop_km)
                tgt_loss = crop_center_xy(tgt, args.loss_central_crop_km)
                weight_loss = crop_center_xy(weight, args.loss_central_crop_km)
                loss, comp = total_loss(pred_loss, tgt_loss, args.loss_type,
                                        weight=weight_loss, w_amp=args.w_amp, w_div=args.w_div)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
                for k, v in comp.items():
                    comp_acc[k] = comp_acc.get(k, 0.0) + v
                if step % 50 == 0:
                    logger.info("ep %d step %d/%d loss=%.5f comp=%s",
                                ep, step, len(train_loader), loss.item(),
                                {k: round(v, 5) for k, v in comp.items()})
            train_loss /= max(len(train_loader), 1)
            for k in comp_acc:
                comp_acc[k] /= max(len(train_loader), 1)
            sched.step()

            model.eval()
            val_loss = val_mse = val_mse_agl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    terrain, era5, geo, tgt, weight, obs_value, obs_mask, obs_ij = unpack_batch(batch)
                    terrain = terrain.to(args.device, non_blocking=True)
                    era5 = era5.to(args.device, non_blocking=True)
                    geo = None if geo is None else geo.to(args.device, non_blocking=True)
                    tgt = tgt.to(args.device, non_blocking=True)
                    weight = None if weight is None else weight.to(args.device, non_blocking=True)
                    obs_value = obs_value.to(args.device, non_blocking=True)
                    obs_mask = obs_mask.to(args.device, non_blocking=True)
                    obs_ij = obs_ij.to(args.device, non_blocking=True)
                    pred = model(terrain, era5, geo, obs_value, obs_mask, obs_ij)
                    pred_loss = crop_center_xy(pred, args.loss_central_crop_km)
                    tgt_loss = crop_center_xy(tgt, args.loss_central_crop_km)
                    weight_loss = crop_center_xy(weight, args.loss_central_crop_km)
                    l, _ = total_loss(pred_loss, tgt_loss, args.loss_type,
                                      weight=weight_loss, w_amp=args.w_amp, w_div=args.w_div)
                    val_loss += l.item()
                    val_mse += mse_loss(pred_loss, tgt_loss).item()
                    if use_weight:
                        val_mse_agl += mse_loss(pred_loss, tgt_loss, weight_loss).item()
            val_loss /= max(len(val_loader), 1)
            val_mse /= max(len(val_loader), 1)
            val_mse_agl = val_mse_agl / max(len(val_loader), 1) if use_weight else None
            wall = time.time() - t0
            row = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                   "val_mse": val_mse, "val_mse_agl": val_mse_agl,
                   "lr": optim.param_groups[0]["lr"], "wall_s": wall,
                   **{f"train_{k}": v for k, v in comp_acc.items()}}
            history.append(row)
            logger.info("EP %d train=%.5f val=%.5f val_mse=%.5f val_mse_agl=%s lr=%.2e (%.0fs)",
                        ep, train_loss, val_loss, val_mse,
                        "nan" if val_mse_agl is None else f"{val_mse_agl:.5f}",
                        row["lr"], wall)
            if mlflow is not None:
                try:
                    mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))},
                                       step=ep)
                except Exception as exc:
                    logger.warning("MLflow metric logging failed: %s", exc)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "val_loss": val_loss, "val_mse": val_mse,
                            "resume_load_stats": load_stats,
                            "config": vars(args)}, args.out_dir / "best.pt")

            epoch_1based = ep + 1
            if args.checkpoint_every and epoch_1based % args.checkpoint_every == 0:
                ckpt_path = args.out_dir / f"epoch_{epoch_1based:03d}.pt"
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "val_loss": val_loss, "val_mse": val_mse,
                            "resume_load_stats": load_stats,
                            "config": vars(args)}, ckpt_path)
                logger.info("checkpoint snapshot saved to %s", ckpt_path)

            if toggle_milestones and epoch_1based in toggle_milestones:
                tog_t0 = time.time()
                try:
                    tog = run_toggle_monitor(model, val_ds, args.device,
                                             n_cases=args.toggle_n_cases,
                                             use_geo=args.use_geo,
                                             use_weight=use_weight)
                except Exception as exc:
                    logger.warning("toggle monitor failed at epoch %d: %s",
                                   epoch_1based, exc)
                    tog = None
                if tog is not None:
                    row["toggle_mse_drop0"] = tog["mse_drop0"]
                    row["toggle_mse_drop1"] = tog["mse_drop1"]
                    row["toggle_delta"] = tog["delta"]
                    row["toggle_mean_abs_pred_diff"] = tog["mean_abs_pred_diff"]
                    row["toggle_n_cases"] = tog["n_cases"]
                    history[-1] = row  # ensure history.yaml carries the new keys
                    logger.info("TOGGLE ep=%d mse_drop0=%.6f mse_drop1=%.6f "
                                "Δ=%.6e |Δpred|=%.6e n=%d (%.1fs)",
                                epoch_1based, tog["mse_drop0"], tog["mse_drop1"],
                                tog["delta"], tog["mean_abs_pred_diff"],
                                tog["n_cases"], time.time() - tog_t0)
                    if mlflow is not None:
                        try:
                            mlflow.log_metrics({
                                "toggle/mse_drop0": tog["mse_drop0"],
                                "toggle/mse_drop1": tog["mse_drop1"],
                                "toggle/delta": tog["delta"],
                                "toggle/mean_abs_pred_diff": tog["mean_abs_pred_diff"],
                            }, step=ep)
                        except Exception as exc:
                            logger.warning("MLflow toggle log failed: %s", exc)
                    if (epoch_1based == args.toggle_warn_epoch
                            and abs(tog["delta"]) < args.toggle_warn_threshold):
                        sys.stderr.write(
                            f"TOGGLE FLAT — canal OBS pas activé après epoch "
                            f"{epoch_1based} (Δ={tog['delta']:.3e} < threshold "
                            f"{args.toggle_warn_threshold:.1e}), escalate Boss\n"
                        )
                        sys.stderr.flush()
    finally:
        (args.out_dir / "history.yaml").write_text(yaml.dump(history))
        if mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass
    logger.info("done. best val_loss=%.5f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
