from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

_PROJECT = Path(__file__).resolve().parents[2]
_SURROGATE = _PROJECT / "services" / "module2b-surrogate"
if str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))

from src.ann_correction import ANNCorrection  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels  # noqa: E402
from src.dataset_v2_obs_centered import (  # noqa: E402
    I_CENTER,
    J_CENTER,
    NI,
    NJ,
    ObsCenteredDataset,
    collate_obs_centered,
)
from train_v2_devine_style import (  # noqa: E402
    _build_era5_layout,
    _era5_baseline_uv_at_center,
    _load_norm_overrides,
    build_frozen_surrogate,
)

logger = logging.getLogger("audit_devine_perdigao")
RINGS = (0, 1, 2, 3, 5)
SMOOTH_SPIKE_RATIO_THRESHOLD = 3.0


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    return obj


def _ts_from_ns(value) -> pd.Timestamp:
    return pd.Timestamp(str(np.array(int(value)).astype("datetime64[ns]")))


def _decode_site_id(value) -> str:
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _infer_axes(shape: tuple[int, ...], n_time: int, n_site: int, n_height: int) -> tuple[int, int, int]:
    matches: list[tuple[int, int, int]] = []
    for t_ax in range(3):
        for s_ax in range(3):
            for h_ax in range(3):
                if len({t_ax, s_ax, h_ax}) != 3:
                    continue
                if shape[t_ax] == n_time and shape[s_ax] == n_site and shape[h_ax] == n_height:
                    matches.append((t_ax, s_ax, h_ax))
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer (time,site,height) axes for shape={shape}, "
            f"lengths={(n_time, n_site, n_height)}, matches={matches}"
        )
    return matches[0]


def build_perdigao_pairings(
    obs_zarr: Path,
    height_target: float = 10.0,
    time_subsample: int | None = None,
    max_stations: int | None = None,
) -> pd.DataFrame:
    g = zarr.open_group(str(obs_zarr), mode="r")
    site_ids = [_decode_site_id(x) for x in g["coords/site_id"][:]]
    lats = np.asarray(g["coords/lat"][:], dtype=np.float64)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float64)
    elevs = np.asarray(g["coords/altitude_m"][:], dtype=np.float64)
    heights = np.asarray(g["coords/height_m"][:], dtype=np.float64)
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    h_idx = int(np.argmin(np.abs(heights - float(height_target))))

    u_arr = np.asarray(g["sites/u"][:], dtype=np.float32)
    v_arr = np.asarray(g["sites/v"][:], dtype=np.float32)
    axes = _infer_axes(u_arr.shape, len(times), len(site_ids), len(heights))
    u_tsh = np.moveaxis(u_arr, axes, (0, 1, 2))
    v_tsh = np.moveaxis(v_arr, axes, (0, 1, 2))
    speed = np.sqrt(u_tsh[:, :, h_idx] ** 2 + v_tsh[:, :, h_idx] ** 2)

    ts = pd.to_datetime([_ts_from_ns(x) for x in times])
    period = np.asarray(
        (ts >= pd.Timestamp("2017-05-01")) & (ts <= pd.Timestamp("2017-06-30 23:59:59.999999999")),
        dtype=bool,
    )
    rows: list[dict] = []
    n_sites = len(site_ids) if max_stations is None else min(len(site_ids), int(max_stations))
    for s_idx in range(n_sites):
        valid = np.flatnonzero(period & np.isfinite(speed[:, s_idx]) & (speed[:, s_idx] > 0.0))
        if time_subsample is not None and valid.size > int(time_subsample):
            # Evenly spread across the IOP (not the first N) so the subsample
            # spans distinct 6h ERA5 blocks / wind regimes. Stride-based.
            stride = max(1, valid.size // int(time_subsample))
            valid = valid[::stride][: int(time_subsample)]
        for t_idx in valid:
            rows.append(
                {
                    "station_id": f"perdigao_{site_ids[s_idx]}",
                    "timestamp": pd.Timestamp(ts[t_idx]),
                    "lat": float(lats[s_idx]),
                    "lon": float(lons[s_idx]),
                    "elev": float(elevs[s_idx]),
                    "height_obs": float(heights[h_idx]),
                    "speed_obs": float(speed[t_idx, s_idx]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid Perdigao pairings found in {obs_zarr}")
    logger.info(
        "Perdigao pairings: rows=%d stations=%d height=%.3f axes(time,site,height)=%s",
        len(df),
        df["station_id"].nunique(),
        float(heights[h_idx]),
        axes,
    )
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def ring_offsets() -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = [(0, 0)]
    for k in RINGS[1:]:
        offsets.extend([(k, 0), (-k, 0), (0, k), (0, -k)])
    return offsets


def denorm_uv_neighbourhood(
    pred_norm: torch.Tensor,
    norm: dict,
    k_obs: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    batch_idx = torch.arange(pred_norm.shape[0], device=pred_norm.device)
    u_uv_scale = float(norm["U_uv_scale"])
    u_x_off = float(norm["U_x_offset"])
    u_y_off = float(norm["U_y_offset"])
    out: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for di, dj in offsets:
        ii = min(max(I_CENTER + int(di), 0), NI - 1)
        jj = min(max(J_CENTER + int(dj), 0), NJ - 1)
        u_n = pred_norm[batch_idx, 0, ii, jj, k_obs]
        v_n = pred_norm[batch_idx, 1, ii, jj, k_obs]
        out[(di, dj)] = (u_n * u_uv_scale + u_x_off, v_n * u_uv_scale + u_y_off)
    return out


def _load_ann(cfg: dict, checkpoint: Path, era5_dim: int, device: str) -> ANNCorrection:
    ann = ANNCorrection(
        era5_dim=era5_dim,
        topo_dim=int(cfg.get("topo_dim", 8)),
        hidden_units=tuple(cfg.get("hidden_units", [50, 10])),
        dropout=float(cfg.get("dropout", 0.25)),
        zero_init_output=True,
        use_terrain_encoder=bool(cfg.get("use_terrain_encoder", False)),
        terrain_latent_dim=int(cfg.get("terrain_latent_dim", 48)),
        terrain_in_channels=int(cfg.get("terrain_in_channels", 4)),
    ).to(device)
    ck = torch.load(str(checkpoint), map_location=device, weights_only=False)
    ann.load_state_dict(ck["model"])
    ann.eval()
    return ann


def _speed_by_offset(
    pred: torch.Tensor,
    era5_flat: torch.Tensor,
    norm: dict,
    era5_layout: dict,
    k_obs: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    uv = denorm_uv_neighbourhood(pred, norm, k_obs, offsets)
    u10, v10 = _era5_baseline_uv_at_center(era5_flat, norm, era5_layout)
    speeds: dict[tuple[int, int], torch.Tensor] = {}
    for off, (u_res, v_res) in uv.items():
        u = u_res + u10
        v = v_res + v10
        speeds[off] = torch.sqrt(u * u + v * v + 1e-8)
    return speeds


def _forward_rows(
    ann: torch.nn.Module,
    surrogate: torch.nn.Module,
    loader: DataLoader,
    norm: dict,
    era5_layout: dict,
    device: str,
    *,
    limit_batches: int | None,
    dry_run: bool,
) -> tuple[list[dict], dict[tuple[int, int], list[float]]]:
    offsets = ring_offsets()
    rows: list[dict] = []
    heat: dict[tuple[int, int], list[float]] = {off: [] for off in offsets}
    ann.eval()
    surrogate.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            terrain, era5, geo, topo, speed_obs, k_obs, meta = batch
            terrain = terrain.to(device, non_blocking=True)
            era5 = era5.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            topo = topo.to(device, non_blocking=True)
            speed_obs = speed_obs.to(device, non_blocking=True)
            k_obs = k_obs.to(device, non_blocking=True)

            era5_corr = ann(era5, topo, terrain=terrain)
            pred_corr = surrogate(terrain, era5_corr, geo)
            pred_raw = surrogate(terrain, era5, geo)
            speed_corr = _speed_by_offset(pred_corr, era5_corr, norm, era5_layout, k_obs, offsets)
            speed_raw = _speed_by_offset(pred_raw, era5, norm, era5_layout, k_obs, offsets)
            delta = {off: (speed_corr[off] - speed_raw[off]).detach().cpu().numpy() for off in offsets}
            corr_np = {off: speed_corr[off].detach().cpu().numpy() for off in offsets}
            raw_np = {off: speed_raw[off].detach().cpu().numpy() for off in offsets}
            obs_np = speed_obs.detach().cpu().numpy()

            if dry_run:
                print(f"terrain.shape={tuple(terrain.shape)}", flush=True)
                print(f"pred_corr.shape={tuple(pred_corr.shape)} pred_raw.shape={tuple(pred_raw.shape)}", flush=True)
                print(f"neighbourhood_offsets={offsets}", flush=True)

            for i, m in enumerate(meta):
                for off in offsets:
                    if np.isfinite(delta[off][i]):
                        heat[off].append(float(delta[off][i]))
                ring_mean = {}
                for k in RINGS[1:]:
                    vals = np.array([delta[off][i] for off in [(k, 0), (-k, 0), (0, k), (0, -k)]], dtype=np.float64)
                    ring_mean[k] = float(np.nanmean(vals))
                smooth_vals = [delta[(0, 0)][i]]
                for k in (1, 2, 3):
                    smooth_vals.extend(delta[off][i] for off in [(k, 0), (-k, 0), (0, k), (0, -k)])
                denom = float(np.nanmean(np.abs([ring_mean[1], ring_mean[2], ring_mean[3]]))) + 1e-6
                row = {
                    "station_id": str(m["station_id"]),
                    "timestamp_iso": str(m["timestamp_iso"]),
                    "speed_obs": float(obs_np[i]),
                    "speed_corr_centre": float(corr_np[(0, 0)][i]),
                    "speed_raw_centre": float(raw_np[(0, 0)][i]),
                    "delta_centre": float(delta[(0, 0)][i]),
                    "delta_ring_1": ring_mean[1],
                    "delta_ring_2": ring_mean[2],
                    "delta_ring_3": ring_mean[3],
                    "delta_ring_5": ring_mean[5],
                    "smoothness": float(np.nanstd(np.asarray(smooth_vals, dtype=np.float64))),
                    "spike_ratio": float(abs(delta[(0, 0)][i]) / denom),
                }
                rows.append(row)
                if dry_run:
                    print(
                        f"PAIRING {row['station_id']} {row['timestamp_iso']} "
                        f"speed_obs={row['speed_obs']:.4f}",
                        flush=True,
                    )
                    for off in offsets:
                        print(
                            f"  offset={off} speed_corr={corr_np[off][i]:.6f} "
                            f"speed_raw={raw_np[off][i]:.6f} delta={delta[off][i]:+.6f}",
                            flush=True,
                        )
                    return rows, heat
    return rows, heat


def _build_dataset(cfg: dict, norm: dict, pairings: Path, args) -> ObsCenteredDataset:
    return ObsCenteredDataset(
        pairings,
        era5_store=Path(cfg["era5_store"]),
        dem=Path(cfg["dem"]),
        worldcover=Path(cfg["worldcover"]) if cfg.get("worldcover") else None,
        cache_dir=Path(args.cache_dir),
        norm=norm,
        target_agl_levels=cfg.get("target_agl_levels", "agl_0_100_24"),
        max_era5_delta_h=float(cfg.get("max_era5_delta_h", 3.5)),
        seed=int(cfg.get("seed", 42)),
        n_workers=int(args.n_prep_workers if args.n_prep_workers is not None else cfg.get("n_prep_workers", 4)),
        overwrite_cache=False,
        require_cached=False,
        enable_phys_features=bool(cfg.get("enable_phys_features", False)),
    )


def _err(pred: pd.Series, obs: pd.Series, absolute: bool) -> float:
    p = pred.to_numpy(dtype=np.float64)
    o = obs.to_numpy(dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(o)
    if not mask.any():
        return float("nan")
    d = p[mask] - o[mask]
    return float(np.abs(d).mean() if absolute else d.mean())


def _summary(df: pd.DataFrame, wall_s: float) -> dict:
    ratios: dict[str, float] = {}
    abs_c = np.abs(df["delta_centre"].to_numpy(dtype=np.float64)) + 1e-6
    for k in RINGS[1:]:
        ratios[f"ring_{k}"] = float(np.nanmean(np.abs(df[f"delta_ring_{k}"]) / abs_c))
    spike = df["spike_ratio"].to_numpy(dtype=np.float64) > SMOOTH_SPIKE_RATIO_THRESHOLD
    return {
        "wall_s": wall_s,
        "centre_mae": {
            "corrected": _err(df["speed_corr_centre"], df["speed_obs"], True),
            "raw": _err(df["speed_raw_centre"], df["speed_obs"], True),
            "bias_corrected": _err(df["speed_corr_centre"], df["speed_obs"], False),
            "bias_raw": _err(df["speed_raw_centre"], df["speed_obs"], False),
            "n_pairings": int(len(df)),
            "n_stations": int(df["station_id"].nunique()),
        },
        "propagation": {
            "smoothness_mean": float(np.nanmean(df["smoothness"])),
            "smoothness_median": float(np.nanmedian(df["smoothness"])),
            "spike_ratio_median": float(np.nanmedian(df["spike_ratio"])),
            "spike_ratio_p90": float(np.nanpercentile(df["spike_ratio"], 90)),
            "mean_abs_delta_ring_over_abs_delta_centre": ratios,
            "spike_ratio_threshold": SMOOTH_SPIKE_RATIO_THRESHOLD,
            "classification_note": "spike if spike_ratio > threshold; smooth otherwise",
            "pct_spike": float(np.nanmean(spike) * 100.0),
            "pct_smooth": float(np.nanmean(~spike) * 100.0),
        },
        "notes": {
            "era5_time_delta_h": 6,
            "perdigao_is_immutable_test_set": True,
        },
    }


def _write_heatmap(heat: dict[tuple[int, int], list[float]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((11, 11), np.nan, dtype=np.float32)
    for (di, dj), vals in heat.items():
        if vals:
            grid[di + 5, dj + 5] = float(np.nanmean(vals))
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", extent=[-5.5, 5.5, -5.5, 5.5], cmap="coolwarm")
    ax.set_title("Mean Perdigao DEVINE correction delta")
    ax.set_xlabel("dj pixels")
    ax.set_ylabel("di pixels")
    fig.colorbar(im, ax=ax, label="speed_corr - speed_raw (m/s)")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _missing_forward_inputs(cfg: dict, ann_checkpoint: Path) -> list[str]:
    checks = [
        ann_checkpoint,
        Path(cfg["surrogate_checkpoint"]),
        Path(cfg["era5_store"]),
        Path(cfg["dem"]),
        Path(cfg["norm_yaml"]),
    ]
    missing: list[str] = []
    for p in checks:
        if not p.exists():
            missing.append(str(p))
    return missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ann-checkpoint", type=Path, default=None)
    ap.add_argument("--obs-zarr", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("scratch/perdigao_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/validation/phase_H_prime_perdigao"))
    ap.add_argument("--pairings-out", type=Path, default=Path("tmp/perdigao_iop_pairings.parquet"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--time-subsample", type=int, default=None)
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--n-prep-workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    t0 = time.time()
    cfg = yaml.safe_load(args.config.read_text())
    ann_checkpoint = args.ann_checkpoint or (Path(cfg["output_dir"]) / "best.pt")
    if args.dry_run:
        args.device = "cpu"
        args.max_stations = 1
        args.time_subsample = 1
        args.limit_batches = 1 if args.limit_batches is None else args.limit_batches

    pairings = build_perdigao_pairings(args.obs_zarr, time_subsample=args.time_subsample, max_stations=args.max_stations)
    args.pairings_out.parent.mkdir(parents=True, exist_ok=True)
    pairings.to_parquet(args.pairings_out, index=False)
    print(f"PAIRINGS rows={len(pairings)} stations={pairings['station_id'].nunique()} out={args.pairings_out}", flush=True)
    print(pairings.head(min(5, len(pairings))).to_string(index=False), flush=True)

    levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    nz = int(levels.size)
    era5_layout = _build_era5_layout(n_pressure=int(cfg.get("n_pressure_levels", 10)))
    era5_dim = int(era5_layout["total_dim"])
    offsets = ring_offsets()
    print(f"RESOLVED era5_dim={era5_dim} nz={nz} offsets={offsets}", flush=True)

    missing = _missing_forward_inputs(cfg, Path(ann_checkpoint))
    if missing:
        msg = "DRY-RUN PARTIAL: forward skipped, missing " + ", ".join(missing)
        if args.dry_run:
            print(msg, flush=True)
            return
        raise FileNotFoundError(msg)

    norm = {**DEFAULT_NORM, **_load_norm_overrides(Path(cfg["norm_yaml"]))}
    device = args.device or cfg.get("device", "cuda")
    ds = _build_dataset(cfg, norm, args.pairings_out, args)
    if len(ds) == 0:
        raise RuntimeError("No usable Perdigao pairings survived dataset construction")
    bs = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 4))
    nw = int(args.num_workers if args.num_workers is not None else cfg.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_obs_centered,
                        pin_memory=str(device).startswith("cuda"), persistent_workers=nw > 0)
    surrogate = build_frozen_surrogate(Path(cfg["surrogate_checkpoint"]), era5_dim=era5_dim, nz=nz,
                                       terrain_in_channels=4, geo_channels=int(cfg.get("geo_channels", 2)),
                                       preset=cfg.get("surrogate_preset", "base"), device=device)
    ann = _load_ann(cfg, Path(ann_checkpoint), era5_dim, device)
    rows, heat = _forward_rows(ann, surrogate, loader, norm, era5_layout, device,
                               limit_batches=args.limit_batches, dry_run=args.dry_run)
    if args.dry_run:
        print("DRY-RUN FULL: forward ran locally", flush=True)
        return

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No forward rows produced")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "perdigao_propagation.csv", index=False)
    summary = _summary(df, time.time() - t0)
    (args.out_dir / "perdigao_summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    _write_heatmap(heat, args.out_dir / "delta_correction_heatmap.png")
    logger.info("Wrote Perdigao propagation audit to %s", args.out_dir)


if __name__ == "__main__":
    main()
