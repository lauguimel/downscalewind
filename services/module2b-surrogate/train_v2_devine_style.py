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
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from src.ann_correction import (  # noqa: E402
    ANNCorrection,
    devine_speed_loss,
    devine_speed_loss_regime,
)
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


# ─── Multi-parquet merge (M_I7) ──────────────────────────────────────────────


def resolve_pairings(cfg: dict, out_dir: Path) -> Path:
    """Return the pairings parquet path, merging `extra_parquets` if present.

    Config schema:
        pairings_parquet: <base path>          # weight 1.0, cfg-level era5_store
        extra_parquets:                        # optional list of entries
          - path: <parquet>
            weight: 2.0                        # uniform sample weight (default 1)
            pop_weights: {tower_icos: 4.0}     # per-`pop`-column override
            era5_store: <zarr>                 # optional per-entry store
            season_era5_stores: {jja2020: <zarr>}  # per-`season`-column store

    Per-pop reweighting is implemented downstream via WeightedRandomSampler on
    a `sample_weight` column (NOT row duplication): fractional weights work,
    epoch length stays = n_rows, and the materialised grid.zarr cache is not
    inflated by duplicate pairings.
    """
    extras = cfg.get("extra_parquets") or []
    base = Path(cfg["pairings_parquet"])
    if not extras:
        return base

    def _load(path: Path, entry: dict | None) -> pd.DataFrame:
        df = pd.read_parquet(path)
        entry = entry or {}
        w = float(entry.get("weight", 1.0))
        df["sample_weight"] = w
        pop_w = entry.get("pop_weights") or {}
        if pop_w:
            if "pop" not in df.columns:
                raise ValueError(f"pop_weights given but no `pop` column in {path}")
            for pop, pw in pop_w.items():
                df.loc[df["pop"] == pop, "sample_weight"] = float(pw)
        store = entry.get("era5_store", cfg.get("era5_store"))
        df["era5_store"] = str(store) if store else ""
        season_stores = entry.get("season_era5_stores") or {}
        if season_stores:
            if "season" not in df.columns:
                raise ValueError(
                    f"season_era5_stores given but no `season` column in {path}")
            for season, sp in season_stores.items():
                df.loc[df["season"] == season, "era5_store"] = str(sp)
        return df

    frames = [_load(base, {"weight": float(cfg.get("base_weight", 1.0))})]
    for entry in extras:
        frames.append(_load(Path(entry["path"]), entry))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Mixed per-source dtypes leave `timestamp` as object after concat, which
    # pyarrow refuses to serialise — coerce to a single datetime64[ns] column.
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    out = out_dir / "merged_pairings.parquet"
    merged.to_parquet(out, index=False)
    logger.info("Merged %d parquets → %s (%d rows, weights %s)",
                1 + len(extras), out, len(merged),
                sorted(merged["sample_weight"].unique().tolist()))
    return out


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
    loss_mode: str = "devine",
    regime_kwargs: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    terrain, era5, geo, topo, speed_obs, k_obs, _meta = batch
    terrain = terrain.to(device, non_blocking=True)
    era5 = era5.to(device, non_blocking=True)
    geo = geo.to(device, non_blocking=True)
    topo = topo.to(device, non_blocking=True)
    speed_obs = speed_obs.to(device, non_blocking=True)
    k_obs = k_obs.to(device, non_blocking=True)

    if use_ann:
        era5_corrected = ann(era5, topo, terrain=terrain)
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

    if loss_mode == "regime":
        loss = devine_speed_loss_regime(
            speed_pred, speed_obs,
            tau_under=tau_under, tau_over=tau_over,
            **(regime_kwargs or {}),
        )
    else:
        loss = devine_speed_loss(speed_pred, speed_obs,
                                 tau_under=tau_under, tau_over=tau_over)
    diag = {
        "loss": float(loss.detach().cpu().item()),
        "mae": float((speed_pred - speed_obs).abs().mean().detach().cpu().item()),
        "bias": float((speed_pred - speed_obs).mean().detach().cpu().item()),
        "speed_obs_mean": float(speed_obs.mean().detach().cpu().item()),
        "speed_pred_mean": float(speed_pred.mean().detach().cpu().item()),
        # raw per-sample vectors for stratified (low/high-wind) diagnostics
        "speed_obs_vec": speed_obs.detach().cpu(),
        "speed_pred_vec": speed_pred.detach().cpu(),
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
    pairings = resolve_pairings(cfg, out_dir)
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
        enable_phys_features=bool(cfg.get("enable_phys_features", False)),
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
    # Per-pop reweighting (M_I7): draw with replacement following the merged
    # parquet's sample_weight column; falls back to plain shuffling when all
    # weights are 1.0 (legacy configs).
    weights = train_ds.sample_weights
    use_sampler = any(abs(w - 1.0) > 1e-9 for w in weights)
    sampler = None
    if use_sampler:
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds), replacement=True,
        )
        logger.info("WeightedRandomSampler on: %d samples, weight range %.2f-%.2f",
                    len(weights), min(weights), max(weights))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers,
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
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", False)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
        use_calm_gate=bool(cfg.get("use_calm_gate", False)),
        gate_v0_init=float(cfg.get("gate_v0_init", 2.5)),
        gate_s_init=float(cfg.get("gate_s_init", 1.0)),
        gate_norm=norm,
    ).to(device)
    n_ann = sum(p.numel() for p in ann.parameters())
    logger.info("ANN params: %d (%.1f k)", n_ann, n_ann / 1e3)

    # ── Optional fine-tune init (M_I7: start from the M_I5 best.pt) ──────────
    init_from = cfg.get("init_from")
    if init_from:
        ck = torch.load(str(init_from), map_location=device, weights_only=False)
        state = ck["model"] if "model" in ck else ck
        # strict=False ONLY to tolerate gate params absent from an older
        # checkpoint; any other mismatch is a hard error.
        missing, unexpected = ann.load_state_dict(state, strict=False)
        gate_keys = {"gate_v0", "gate_s_raw"}
        bad_missing = [k for k in missing if k not in gate_keys]
        bad_unexpected = [k for k in unexpected if k not in gate_keys]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"init_from={init_from}: incompatible state_dict "
                f"(missing={bad_missing}, unexpected={bad_unexpected})"
            )
        logger.info("Initialised ANN from %s (epoch=%s, gate params kept at "
                    "init: %s)", init_from, ck.get("epoch", "?"), list(missing))

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
    loss_mode = str(cfg.get("loss_mode", "devine")).lower()
    regime_kwargs = {}
    if loss_mode == "regime":
        regime_kwargs = dict(
            calm_threshold=float(cfg.get("calm_threshold", 3.0)),
            calm_width=float(cfg.get("calm_width", 1.5)),
            calm_over_penalty=float(cfg.get("calm_over_penalty", 2.0)),
            weight_floor=float(cfg.get("weight_floor", 1.0)),
        )
    elif loss_mode != "devine":
        raise ValueError(f"Unknown loss_mode={loss_mode!r}")
    # low-wind stratification threshold for val diagnostics (obs-based, m/s)
    calm_strat_thr = float(cfg.get("calm_strat_thr",
                                   regime_kwargs.get("calm_threshold", 3.0)))
    logger.info("loss_mode=%s tau_under=%.3f tau_over=%.3f regime=%s strat_thr=%.2f",
                loss_mode, tau_under, tau_over, regime_kwargs or "-", calm_strat_thr)
    step_loss_kwargs = dict(tau_under=tau_under, tau_over=tau_over,
                            loss_mode=loss_mode, regime_kwargs=regime_kwargs)

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
                               use_ann=True, **step_loss_kwargs)
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

        # Val — collect per-sample (obs, corrected pred, raw pred) speeds for
        # wind-stratified diagnostics (the M_I5b key metric is the low-wind bias).
        ann.eval()
        obs_all, corr_all, raw_all, k_all = [], [], [], []
        with torch.no_grad():
            vagg = {"loss": 0.0, "n": 0}
            for batch in val_loader:
                lc, diag_c = _step(ann, surrogate, batch, norm, era5_layout, device,
                                   use_ann=True, **step_loss_kwargs)
                _, diag_r = _step(ann, surrogate, batch, norm, era5_layout, device,
                                  use_ann=False, **step_loss_kwargs)
                vagg["loss"] += diag_c["loss"]
                vagg["n"] += 1
                obs_all.append(diag_c["speed_obs_vec"])
                corr_all.append(diag_c["speed_pred_vec"])
                raw_all.append(diag_r["speed_pred_vec"])
                k_all.append(batch[5].detach().cpu().reshape(-1))
        obs = torch.cat(obs_all)
        corr = torch.cat(corr_all)
        raw = torch.cat(raw_all)
        k_obs_vec = torch.cat(k_all)
        val_loss = vagg["loss"] / max(1, vagg["n"])
        val_mae = float((corr - obs).abs().mean())
        val_bias = float((corr - obs).mean())
        val_mae_raw = float((raw - obs).abs().mean())
        val_bias_raw = float((raw - obs).mean())

        # Wind-stratified (low = obs < calm_strat_thr; high = obs >= thr)
        low = obs < calm_strat_thr
        high = ~low
        n_low = int(low.sum())
        n_high = int(high.sum())

        def _m(t, mask):
            return float(t[mask].mean()) if int(mask.sum()) > 0 else float("nan")

        low_bias_corr = _m(corr - obs, low)
        low_bias_raw = _m(raw - obs, low)
        low_mae_corr = _m((corr - obs).abs(), low)
        low_mae_raw = _m((raw - obs).abs(), low)
        high_bias_corr = _m(corr - obs, high)
        high_bias_raw = _m(raw - obs, high)
        high_mae_corr = _m((corr - obs).abs(), high)
        high_mae_raw = _m((raw - obs).abs(), high)

        # ── Per-height metrics + height-balanced selection score (M_I8) ───────
        # The aggregate val_mae is dominated by the 10 m rows (~96% of the val
        # set: ISD stations only measure at 10 m), so selecting the best epoch on
        # it is blind to the multi-height objective — this is what made M_I7a
        # uninterpretable. `sel_score` instead averages the per-height MAEs so a
        # height counts the same whatever its row count.
        #   mode "equal"      : every height weighs 1/H  (best profile)
        #   mode "protect10m" : 10 m weighs `w10`, the rest share (1 - w10)
        #                       (guards the fire-weather headline)
        agl_levels_v = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
        err_c = (corr - obs).abs()
        per_height: dict[str, dict[str, float]] = {}
        for k in sorted(set(int(x) for x in k_obs_vec.tolist())):
            m = k_obs_vec == k
            h = float(agl_levels_v[k]) if agl_levels_v is not None and k < len(agl_levels_v) else float(k)
            per_height[f"{h:.0f}m"] = {
                "k": k, "n": int(m.sum()),
                "mae_corr": _m(err_c, m),
                "mae_raw": _m((raw - obs).abs(), m),
                "bias_corr": _m(corr - obs, m),
            }
        sel_mode = str(cfg.get("height_weight_mode", "protect10m"))
        w10 = float(cfg.get("height_weight_10m", 0.5))
        maes = {hh: d["mae_corr"] for hh, d in per_height.items()
                if not math.isnan(d["mae_corr"])}
        if not maes:
            sel_score = val_mae
        elif sel_mode == "equal" or "10m" not in maes or len(maes) == 1:
            sel_score = sum(maes.values()) / len(maes)
        else:
            others = [v for hh, v in maes.items() if hh != "10m"]
            sel_score = w10 * maes["10m"] + (1.0 - w10) * (sum(others) / len(others))

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
            # stratified diagnostics (M_I5b)
            "calm_strat_thr": calm_strat_thr,
            "n_low": n_low,
            "n_high": n_high,
            "low_bias_corr": low_bias_corr,
            "low_bias_raw": low_bias_raw,
            "low_mae_corr": low_mae_corr,
            "low_mae_raw": low_mae_raw,
            "high_bias_corr": high_bias_corr,
            "high_bias_raw": high_bias_raw,
            "high_mae_corr": high_mae_corr,
            "high_mae_raw": high_mae_raw,
            # height-resolved diagnostics (M_I8)
            "per_height": per_height,
            "sel_score": sel_score,
            "sel_mode": sel_mode,
        }
        history.append(entry)
        logger.info(
            "   per-height mae_corr: %s",
            " ".join(f"{hh}:{d['mae_corr']:.3f}(n={d['n']})"
                     for hh, d in sorted(per_height.items(),
                                         key=lambda kv: kv[1]["k"])),
        )
        logger.info("   sel_score(%s)=%.4f  (aggregate val_mae=%.4f)",
                    sel_mode, sel_score, val_mae)
        logger.info(
            "ep=%d wall=%.1fs train_loss=%.4f mae=%.3f | val_loss=%.4f mae=%.3f bias=%+.3f "
            "| RAW mae=%.3f bias=%+.3f | Δmae=%+.3f",
            epoch, wall, train_loss, train_mae, val_loss, val_mae, val_bias,
            val_mae_raw, val_bias_raw, val_mae - val_mae_raw,
        )
        logger.info(
            "   LOW(<%.1f, n=%d): bias corr=%+.3f raw=%+.3f | mae corr=%.3f raw=%.3f "
            "|| HIGH(n=%d): bias corr=%+.3f raw=%+.3f | mae corr=%.3f raw=%.3f",
            calm_strat_thr, n_low, low_bias_corr, low_bias_raw, low_mae_corr, low_mae_raw,
            n_high, high_bias_corr, high_bias_raw, high_mae_corr, high_mae_raw,
        )

        # Checkpoint. `best.pt` now tracks the height-balanced sel_score, and
        # every epoch is also kept so a run can be re-diagnosed (or a different
        # selection rule applied) without retraining — M_I7a could not be.
        ckpt = {"model": ann.state_dict(),
                "epoch": epoch,
                "val_mae": val_mae,
                "val_mae_raw": val_mae_raw,
                "sel_score": sel_score,
                "per_height": per_height,
                "cfg": cfg}
        torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")
        if sel_score < best_val_mae:
            best_val_mae = sel_score
            torch.save(ckpt, out_dir / "best.pt")

    (out_dir / "history.yaml").write_text(yaml.safe_dump(history))
    best_entry = min(history, key=lambda e: e.get("sel_score", math.inf)) if history else None
    (out_dir / "summary.json").write_text(json.dumps({
        "n_epochs": n_epochs,
        # `best_sel_score` is what best.pt was selected on (height-balanced);
        # `best_val_mae_aggregate` is the plain aggregate at that same epoch,
        # kept so the number stays comparable with pre-M_I8 runs.
        "best_sel_score": best_val_mae,
        "selection_mode": str(cfg.get("height_weight_mode", "protect10m")),
        "best_epoch": best_entry.get("epoch") if best_entry else None,
        "best_val_mae_aggregate": best_entry.get("val_mae") if best_entry else None,
        "best_per_height": best_entry.get("per_height") if best_entry else None,
        "final": history[-1] if history else None,
    }, indent=2))
    logger.info("Done. best sel_score=%.4f (epoch=%s), output_dir=%s",
                best_val_mae, best_entry.get("epoch") if best_entry else "?", out_dir)


if __name__ == "__main__":
    main()
