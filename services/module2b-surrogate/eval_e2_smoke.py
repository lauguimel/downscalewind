from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from src.dataset_v2 import DEFAULT_NORM, parse_agl_levels
from src.dataset_v2_vit_e2 import WindV2DatasetViT_E2
from src.model_vit_v2_e2 import build_vit_v2_e2
from train_v2_vit import _load_norm_overrides


def unpack_sample(sample, use_geo: bool, use_weight: bool):
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
    case_name = sample[i]
    return terrain, era5, geo, target, obs_value, obs_mask, obs_ij, case_name


def tensor1(x, device):
    return x.unsqueeze(0).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/scratch/maitreje/dsw/training_v2"))
    ap.add_argument("--splits-yaml", type=Path,
                    default=Path("/scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_splits.yaml"))
    ap.add_argument("--norm-yaml", type=Path,
                    default=Path("/scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_norm.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-val-cases", type=int, default=None)
    ap.add_argument("--obs-height-m", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--toggle-dropout", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ck.get("config", {})
    use_geo = bool(cfg.get("use_geo", True))
    use_weight = float(cfg.get("agl_weight_alpha", 0.0)) > 0.0
    norm = {**DEFAULT_NORM, **_load_norm_overrides(args.norm_yaml)}
    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels", "agl_0_100_24"))
    ds = WindV2DatasetViT_E2(
        os.path.realpath(args.data_dir), os.path.realpath(args.splits_yaml), "val",
        norm=norm, include_slopes=bool(cfg.get("include_slopes", True)),
        return_geo=use_geo, use_residual=bool(cfg.get("use_residual", True)),
        residual_baseline_mode=cfg.get("residual_baseline_mode", "surface"),
        return_weight=use_weight, agl_weight_alpha=float(cfg.get("agl_weight_alpha", 2.0)),
        agl_weight_height=float(cfg.get("agl_weight_height", 50.0)),
        target_agl_levels=target_agl_levels, obs_dropout=0.0,
        obs_height_m=args.obs_height_m,
    )
    max_val = args.max_val_cases or cfg.get("max_val_cases") or 4
    ds.cases = ds.cases[:int(max_val)]

    sample = ds[0]
    terrain, era5, geo, target, *_ = unpack_sample(sample, use_geo, use_weight)
    model = build_vit_v2_e2(
        preset=cfg.get("preset", "base"), era5_input_dim=era5.shape[0],
        nz=target.shape[-1], terrain_in_channels=terrain.shape[0],
        geo_channels=0 if geo is None else geo.shape[0],
    ).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    mse0, mse1, diffs, rows = [], [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            sample = ds[idx]
            terrain, era5, geo, target, obs_value, obs_mask, obs_ij, case_name = unpack_sample(
                sample, use_geo, use_weight
            )
            terrain_b = tensor1(terrain, args.device)
            era5_b = tensor1(era5, args.device)
            geo_b = None if geo is None else tensor1(geo, args.device)
            target_b = tensor1(target, args.device)
            obs_value_b = tensor1(obs_value, args.device)
            obs_mask_b = tensor1(obs_mask, args.device)
            obs_ij_b = tensor1(obs_ij, args.device)
            pred0 = model(terrain_b, era5_b, geo_b, obs_value_b, obs_mask_b, obs_ij_b)
            pred1 = model(terrain_b, era5_b, geo_b, torch.zeros_like(obs_value_b),
                          torch.zeros_like(obs_mask_b), obs_ij_b)
            mse0.append(torch.mean((pred0 - target_b) ** 2).item())
            mse1.append(torch.mean((pred1 - target_b) ** 2).item())
            diffs.append(torch.mean(torch.abs(pred0 - pred1)).item())

            i, j = [int(x) for x in obs_ij.tolist()]
            k = ds.obs_k
            p0 = torch.sqrt(pred0[0, 0, i, j, k] ** 2 + pred0[0, 1, i, j, k] ** 2).item()
            p1 = torch.sqrt(pred1[0, 0, i, j, k] ** 2 + pred1[0, 1, i, j, k] ** 2).item()
            tgt = torch.sqrt(target_b[0, 0, i, j, k] ** 2 + target_b[0, 1, i, j, k] ** 2).item()
            row = {"case": case_name, "i": i, "j": j, "k": k,
                   "pred_drop0_speed": p0, "pred_drop1_speed": p1,
                   "target_speed": tgt}
            rows.append(row)
            print(f"{case_name} ij=({i},{j}) k={k} speed drop0={p0:.6f} "
                  f"drop1={p1:.6f} target={tgt:.6f}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_cases": len(ds),
        "obs_k": ds.obs_k,
        "mse_drop0": float(np.mean(mse0)),
        "mse_drop1": float(np.mean(mse1)),
        "mean_abs_pred_drop0_minus_drop1": float(np.mean(diffs)),
        "samples": rows,
    }
    print("MSE_drop0={mse_drop0:.8f} MSE_drop1={mse_drop1:.8f} "
          "mean_abs_diff={mean_abs_pred_drop0_minus_drop1:.8f}".format(**summary))
    out = args.checkpoint.parent / "toggle_test.yaml"
    out.write_text(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
