"""
train_v2_devine_style.py — Phase H' M_H'0: DEVINE-style training of an ANN
correction in front of a FROZEN surrogate v2 ViT.

Pipeline per step:
    era5_flat, topo_features  →  ANN  →  era5_corrected (skip connection)
    era5_corrected + terrain + geo  →  surrogate_v2 (frozen)  →  pred (B,5,180,180,24)
    extract pred[:, 0:2, 90, 90, k_obs]  →  u_pred, v_pred  →  speed_pred
    loss = devine_speed_loss(speed_pred, speed_obs)
    loss.backward()  → gradient flows THROUGH the frozen surrogate
    optimizer.step()  → only ANN params are updated

Reference: Le Toumelin et al. 2024 NPG Sect. 3.3.

Usage (on Aqua H100, fuxicfd env):
    python train_v2_devine_style.py --config configs/training/devine_style_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from src.ann_correction import ANNCorrection, devine_speed_loss  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import (  # noqa: E402
    ObsCenteredDataset,
    collate_obs_centered,
    watertight_station_split,
    I_CENTER,
    J_CENTER,
)
from src.model_vit_v2 import build_vit_v2  # noqa: E402

logger = logging.getLogger("train_devine")


# ─── Norm overrides loader (same logic as train_v2_vit.py) ───────────────────


def _load_norm_overrides(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text())
    s = raw.get("stats", {})
    n: dict = {}

    def _maybe(key, mean_key=None, std_key=None, scale_min=1e-3):
        if key in s:
            n[(mean_key or f"{key}_offset")] = s[key]["mean"]
            n[(std_key or f"{key}_scale")] = max(s[key]["std"], scale_min)

    if "U_x" in s:
        n["U_x_offset"] = s["U_x"]["mean"]
        n["U_uv_scale"] = max(s["U_x"]["std"], 1e-3)
    if "U_y" in s:
        n["U_y_offset"] = s["U_y"]["mean"]
        n["U_uv_scale"] = max(n.get("U_uv_scale", 0.0), s["U_y"]["std"], 1e-3)
    if "U_z" in s:
        n["U_z_offset"] = s["U_z"]["mean"]
        n["U_w_scale"] = max(s["U_z"]["std"], 1e-3)
    for key in ("T", "q", "terrain", "z", "agl"):
        if key in s:
            mean_k = f"{key}_offset"
            std_k = f"{key}_scale"
            n[mean_k] = s[key]["mean"]
            n[std_k] = max(s[key]["std"], 1e-3 if key != "q" else 1e-6)
    for key in ("era5_u", "era5_v", "era5_T", "era5_q", "t2m", "d2m",
                "u10", "v10", "pressure"):
        if key in s:
            mean_k = f"{key}_offset"
            std_k = f"{key}_scale"
            n[mean_k] = s[key]["mean"]
            scale_min = 1e-6 if key == "era5_q" else 1.0
            n[std_k] = max(s[key]["std"], scale_min)
    return n


# ─── Denorm helpers for central-pixel speed extraction ───────────────────────


def _denorm_uv_at_center(
    pred_norm: torch.Tensor,
    norm: dict,
    k_obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract u_pred, v_pred in m/s at (i=90, j=90, k_obs) from normalised pred.

    pred_norm: (B, 5, 180, 180, 24) — channel order = (u, v, w, T, q)
    k_obs:     (B,) long             — per-sample AGL index

    NOTE: when surrogate is trained with `use_residual=True, mode='surface'`,
    its output is a residual that must be added back to the ERA5-lifted
    baseline tensor to recover absolute u/v. Replicating that here at the
    central pixel would require building the baseline tensor; for M_H'0 we
    use the residual u/v directly as a proxy ("delta from ERA5 surface") and
    add the ERA5 u10/v10 baseline AT THE CENTER as the additive part, so
    speed_pred is in physical m/s.
    """
    B = pred_norm.shape[0]
    batch_idx = torch.arange(B, device=pred_norm.device)
    # Index (B, 0, 90, 90, k_obs) → (B,)
    u_n = pred_norm[batch_idx, 0, I_CENTER, J_CENTER, k_obs]
    v_n = pred_norm[batch_idx, 1, I_CENTER, J_CENTER, k_obs]
    # Denorm
    u_uv_scale = float(norm["U_uv_scale"])
    u_x_off = float(norm["U_x_offset"])
    u_y_off = float(norm["U_y_offset"])
    u = u_n * u_uv_scale + u_x_off
    v = v_n * u_uv_scale + u_y_off
    return u, v


def _era5_baseline_uv_at_center(
    era5_corrected: torch.Tensor,
    norm: dict,
    era5_layout: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct ERA5 u10/v10 at the centre (i=1, j=1 of 3×3) from era5_flat.

    era5_corrected: (B, 408) normalised
    era5_layout:    pre-computed dict with index ranges for each block
    Returns u10, v10 in m/s (physical units).
    """
    u10_idx = era5_layout["u10_slice"]
    v10_idx = era5_layout["v10_slice"]
    # The 3×3 surface block for var X is stored as 9 contiguous floats, raveled
    # in row-major order (3 rows × 3 cols). Centre is index (1,1) → flat 4.
    u10_block = era5_corrected[:, u10_idx]      # (B, 9)
    v10_block = era5_corrected[:, v10_idx]
    u10_n_center = u10_block[:, 4]
    v10_n_center = v10_block[:, 4]
    u10 = u10_n_center * float(norm["u10_scale"]) + float(norm["u10_offset"])
    v10 = v10_n_center * float(norm["v10_scale"]) + float(norm["v10_offset"])
    return u10, v10


def _build_era5_layout(n_pressure: int = 10) -> dict:
    """Index slices into the (era5_dim,) flat vector for each block.

    Order (must match utils/inference_batch.build_features):
        [0 : 4*3*3*N_p]                   → 4 pressure-3D vars (u,v,T,q)
        [4*3*3*N_p : 4*3*3*N_p + 4*3*3]   → 4 surface vars (t2m,d2m,u10,v10)
        [... + N_p]                        → pressure levels
        [... + 2]                          → (lat, z0_eff)
    """
    p_block = 4 * 3 * 3 * n_pressure
    s_block = 4 * 3 * 3
    start = p_block
    layout = {}
    for k, name in enumerate(["t2m", "d2m", "u10", "v10"]):
        layout[f"{name}_slice"] = slice(start + k * 9, start + (k + 1) * 9)
    layout["total_dim"] = p_block + s_block + n_pressure + 2
    return layout


# ─── Build the surrogate v2 (frozen) ─────────────────────────────────────────


def build_frozen_surrogate(
    checkpoint: Path,
    era5_dim: int,
    nz: int,
    terrain_in_channels: int = 4,
    geo_channels: int = 2,
    preset: str = "base",
    device: str = "cuda",
) -> torch.nn.Module:
    model = build_vit_v2(
        preset=preset,
        era5_input_dim=era5_dim,
        nz=nz,
        terrain_in_channels=terrain_in_channels,
        geo_channels=geo_channels,
    ).to(device)
    ck = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Surrogate v2 missing keys: %d (e.g. %s)",
                       len(missing), missing[:3])
    if unexpected:
        logger.warning("Surrogate v2 unexpected keys: %d (e.g. %s)",
                       len(unexpected), unexpected[:3])
    # Freeze all params
    for p in model.parameters():
        p.requires_grad_(False)
    # Keep in eval mode → disables Dropout but keeps BN deterministic. BN params
    # are already frozen (requires_grad=False); running stats stay fixed.
    model.eval()
    n_p = sum(p.numel() for p in model.parameters())
    logger.info("Surrogate v2 frozen: %.2f M params (preset=%s, era5=%d, nz=%d)",
                n_p / 1e6, preset, era5_dim, nz)
    return model


# ─── One pass through the dataloader ─────────────────────────────────────────


def _step(
    ann: torch.nn.Module,
    surrogate: torch.nn.Module,
    batch,
    norm: dict,
    era5_layout: dict,
    device: str,
    *,
    use_ann: bool = True,
    tau_under: float = 0.6,
    tau_over: float = 0.4,
) -> tuple[torch.Tensor, dict]:
    terrain, era5, geo, topo, speed_obs, k_obs, _meta = batch
    terrain = terrain.to(device, non_blocking=True)
    era5 = era5.to(device, non_blocking=True)
    geo = geo.to(device, non_blocking=True)
    topo = topo.to(device, non_blocking=True)
    speed_obs = speed_obs.to(device, non_blocking=True)
    k_obs = k_obs.to(device, non_blocking=True)

    if use_ann:
        era5_corrected = ann(era5, topo)
    else:
        era5_corrected = era5

    # Surrogate forward — gradient flows but parameters are frozen
    pred = surrogate(terrain, era5_corrected, geo)
    u_res, v_res = _denorm_uv_at_center(pred, norm, k_obs)

    # Add ERA5 baseline at centre to recover absolute u/v (use_residual mode='surface')
    u10_b, v10_b = _era5_baseline_uv_at_center(era5_corrected, norm, era5_layout)
    u_pred = u_res + u10_b
    v_pred = v_res + v10_b
    speed_pred = torch.sqrt(u_pred * u_pred + v_pred * v_pred + 1e-8)

    loss = devine_speed_loss(speed_pred, speed_obs,
                             tau_under=tau_under, tau_over=tau_over)
    diag = {
        "loss": float(loss.detach().cpu().item()),
        "mae": float((speed_pred - speed_obs).abs().mean().detach().cpu().item()),
        "bias": float((speed_pred - speed_obs).mean().detach().cpu().item()),
        "speed_obs_mean": float(speed_obs.mean().detach().cpu().item()),
        "speed_pred_mean": float(speed_pred.mean().detach().cpu().item()),
    }
    return loss, diag


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda")
    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(target_agl_levels.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = era5_layout["total_dim"]
    logger.info("Resolved nz=%d, era5_dim=%d", nz, era5_dim)

    # ── Splits ───────────────────────────────────────────────────────────────
    pairings = Path(cfg["pairings_parquet"])
    train_sids, val_sids = watertight_station_split(
        pairings, val_frac=float(cfg.get("val_frac", 0.2)),
        seed=int(cfg.get("seed", 42)),
        exclude_substrings=tuple(cfg.get("exclude_substrings", ["perdigao"])),
    )
    n_smoke_stations = cfg.get("max_stations")
    if n_smoke_stations is not None:
        n_train = max(1, int(n_smoke_stations * (1.0 - cfg.get("val_frac", 0.2))))
        n_val = max(1, int(n_smoke_stations - n_train))
        train_sids = train_sids[:n_train]
        val_sids = val_sids[:n_val]
    logger.info("Stations split: train=%d, val=%d", len(train_sids), len(val_sids))

    # ── Datasets ─────────────────────────────────────────────────────────────
    dem = Path(cfg["dem"])
    worldcover = Path(cfg["worldcover"]) if cfg.get("worldcover") else None
    cache_dir = Path(cfg["cache_dir"])
    common_kwargs = dict(
        era5_store=Path(cfg["era5_store"]),
        dem=dem,
        worldcover=worldcover,
        cache_dir=cache_dir,
        norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)),
        n_workers=int(cfg.get("n_prep_workers", 4)),
        overwrite_cache=bool(cfg.get("overwrite_cache", False)),
    )
    train_ds = ObsCenteredDataset(
        pairings,
        station_filter=train_sids,
        max_pairings=cfg.get("max_train_pairings"),
        **common_kwargs,
    )
    val_ds = ObsCenteredDataset(
        pairings,
        station_filter=val_sids,
        max_pairings=cfg.get("max_val_pairings"),
        **common_kwargs,
    )

    bs = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=collate_obs_centered, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate_obs_centered, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── Frozen surrogate v2 ──────────────────────────────────────────────────
    surrogate = build_frozen_surrogate(
        Path(cfg["surrogate_checkpoint"]),
        era5_dim=era5_dim, nz=nz,
        terrain_in_channels=cfg.get("terrain_in_channels", 4),
        geo_channels=cfg.get("geo_channels", 2),
        preset=cfg.get("surrogate_preset", "base"),
        device=device,
    )

    # ── ANN correction ───────────────────────────────────────────────────────
    ann = ANNCorrection(
        era5_dim=era5_dim,
        topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)),
        zero_init_output=True,
    ).to(device)
    n_ann = sum(p.numel() for p in ann.parameters())
    logger.info("ANN params: %d (%.1f k)", n_ann, n_ann / 1e3)

    base_lr = float(cfg.get("lr", 1e-3))
    optimizer = torch.optim.Adam(ann.parameters(), lr=base_lr)

    # ── Optional LR schedule (M_H'1 config sweep) ────────────────────────────
    n_epochs = int(cfg["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    lr_schedule = str(cfg.get("lr_schedule", "constant")).lower()
    scheduler = None
    if lr_schedule == "warmup_cosine":
        warmup_epochs = int(cfg.get("warmup_epochs", 1))
        lr_min_warmup = float(cfg.get("lr_min_warmup", 1e-4))
        lr_final = float(cfg.get("lr_final", 1e-4))
        warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        cosine_steps = max(1, (n_epochs - warmup_epochs) * steps_per_epoch)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(lr_min_warmup / base_lr, 1e-8),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=lr_final,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )
        logger.info("lr_schedule=warmup_cosine warmup_steps=%d cosine_steps=%d "
                    "lr=%.4g→%.4g→%.4g",
                    warmup_steps, cosine_steps, lr_min_warmup, base_lr, lr_final)
    elif lr_schedule != "constant":
        raise ValueError(f"Unknown lr_schedule={lr_schedule!r}")

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
        logger.info("grad_clip_norm=%.3g", grad_clip_norm)

    tau_under = float(cfg.get("tau_under", 0.6))
    tau_over = float(cfg.get("tau_over", 0.4))
    logger.info("loss tau_under=%.3f tau_over=%.3f", tau_under, tau_over)

    # ── Training loop ────────────────────────────────────────────────────────
    history = []
    best_val_mae = math.inf
    for epoch in range(n_epochs):
        ann.train()
        t0 = time.time()
        agg = {"loss": 0.0, "mae": 0.0, "bias": 0.0, "n": 0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, diag = _step(ann, surrogate, batch, norm, era5_layout, device,
                               use_ann=True,
                               tau_under=tau_under, tau_over=tau_over)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(ann.parameters(), grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            agg["loss"] += diag["loss"]
            agg["mae"] += diag["mae"]
            agg["bias"] += diag["bias"]
            agg["n"] += 1
        train_loss = agg["loss"] / max(1, agg["n"])
        train_mae = agg["mae"] / max(1, agg["n"])
        train_bias = agg["bias"] / max(1, agg["n"])

        # Val
        ann.eval()
        with torch.no_grad():
            vagg = {"loss": 0.0, "mae": 0.0, "bias": 0.0, "n": 0,
                    "mae_raw": 0.0, "bias_raw": 0.0}
            for batch in val_loader:
                _, diag_c = _step(ann, surrogate, batch, norm, era5_layout, device,
                                  use_ann=True,
                                  tau_under=tau_under, tau_over=tau_over)
                _, diag_r = _step(ann, surrogate, batch, norm, era5_layout, device,
                                  use_ann=False,
                                  tau_under=tau_under, tau_over=tau_over)
                vagg["loss"] += diag_c["loss"]
                vagg["mae"] += diag_c["mae"]
                vagg["bias"] += diag_c["bias"]
                vagg["mae_raw"] += diag_r["mae"]
                vagg["bias_raw"] += diag_r["bias"]
                vagg["n"] += 1
        val_loss = vagg["loss"] / max(1, vagg["n"])
        val_mae = vagg["mae"] / max(1, vagg["n"])
        val_bias = vagg["bias"] / max(1, vagg["n"])
        val_mae_raw = vagg["mae_raw"] / max(1, vagg["n"])
        val_bias_raw = vagg["bias_raw"] / max(1, vagg["n"])

        wall = time.time() - t0
        lr_end = float(optimizer.param_groups[0]["lr"])
        entry = {
            "epoch": epoch,
            "wall_s": wall,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_bias": train_bias,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_bias": val_bias,
            "val_mae_raw": val_mae_raw,
            "val_bias_raw": val_bias_raw,
            "delta_mae": val_mae - val_mae_raw,
            "lr_end_epoch": lr_end,
        }
        history.append(entry)
        logger.info(
            "ep=%d wall=%.1fs train_loss=%.4f mae=%.3f | val_loss=%.4f mae=%.3f bias=%+.3f "
            "| RAW mae=%.3f bias=%+.3f | Δmae=%+.3f",
            epoch, wall, train_loss, train_mae, val_loss, val_mae, val_bias,
            val_mae_raw, val_bias_raw, val_mae - val_mae_raw,
        )

        # Checkpoint best (by val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {"model": ann.state_dict(),
                 "epoch": epoch,
                 "val_mae": val_mae,
                 "val_mae_raw": val_mae_raw,
                 "cfg": cfg},
                out_dir / "best.pt",
            )

    (out_dir / "history.yaml").write_text(yaml.safe_dump(history))
    (out_dir / "summary.json").write_text(json.dumps({
        "n_epochs": n_epochs,
        "best_val_mae": best_val_mae,
        "final": history[-1] if history else None,
    }, indent=2))
    logger.info("Done. best_val_mae=%.4f, output_dir=%s", best_val_mae, out_dir)


if __name__ == "__main__":
    main()
