"""
Evaluate campaign-v2 residual surrogates in physical units.

This script reconstructs absolute fields for residual models:

    pred_abs_norm = era5_lifted_norm + model(input)

then denormalises to physical units and compares against the CFD teacher.
It also reports the raw ERA5-lifted baseline, so skill is measured against
the same baseline used by the residual target.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr

from src.dataset_v2 import DEFAULT_NORM, WindV2Dataset
from src.dataset_v2_vit import WindV2DatasetViT
from src.model_fno3d import FNO3D
from src.model_vit_v2 import build_vit_v2

logger = logging.getLogger(__name__)

VAR_NAMES = ("u", "v", "w", "T", "q")
AGL_BANDS = (
    ("agl_0_50m", 0.0, 50.0),
    ("agl_50_150m", 50.0, 150.0),
    ("agl_150_300m", 150.0, 300.0),
    ("agl_300_1000m", 300.0, 1000.0),
)


@dataclass
class MetricAccumulator:
    n: int = 0
    se: float = 0.0
    ae: float = 0.0
    bias_sum: float = 0.0
    base_se: float = 0.0
    base_ae: float = 0.0
    base_bias_sum: float = 0.0
    sum_p: float = 0.0
    sum_t: float = 0.0
    sum_p2: float = 0.0
    sum_t2: float = 0.0
    sum_pt: float = 0.0

    def update(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        baseline: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        if mask is not None:
            pred = pred[mask]
            true = true[mask]
            baseline = baseline[mask]
        else:
            pred = pred.ravel()
            true = true.ravel()
            baseline = baseline.ravel()
        if pred.size == 0:
            return

        err = pred - true
        base_err = baseline - true
        self.n += int(pred.size)
        self.se += float(np.sum(err * err, dtype=np.float64))
        self.ae += float(np.sum(np.abs(err), dtype=np.float64))
        self.bias_sum += float(np.sum(err, dtype=np.float64))
        self.base_se += float(np.sum(base_err * base_err, dtype=np.float64))
        self.base_ae += float(np.sum(np.abs(base_err), dtype=np.float64))
        self.base_bias_sum += float(np.sum(base_err, dtype=np.float64))
        self.sum_p += float(np.sum(pred, dtype=np.float64))
        self.sum_t += float(np.sum(true, dtype=np.float64))
        self.sum_p2 += float(np.sum(pred * pred, dtype=np.float64))
        self.sum_t2 += float(np.sum(true * true, dtype=np.float64))
        self.sum_pt += float(np.sum(pred * true, dtype=np.float64))

    def as_dict(self) -> dict[str, float | int]:
        if self.n == 0:
            return {"n": 0}
        rmse = (self.se / self.n) ** 0.5
        base_rmse = (self.base_se / self.n) ** 0.5
        cov = self.sum_pt - self.sum_p * self.sum_t / self.n
        var_p = self.sum_p2 - self.sum_p * self.sum_p / self.n
        var_t = self.sum_t2 - self.sum_t * self.sum_t / self.n
        corr = cov / max((var_p * var_t) ** 0.5, 1e-12)
        return {
            "n": self.n,
            "rmse": rmse,
            "mae": self.ae / self.n,
            "bias": self.bias_sum / self.n,
            "corr": corr,
            "baseline_rmse": base_rmse,
            "baseline_mae": self.base_ae / self.n,
            "baseline_bias": self.base_bias_sum / self.n,
            "skill_vs_baseline": 1.0 - rmse / max(base_rmse, 1e-12),
        }


@dataclass
class EvalAccumulators:
    global_metrics: dict[str, MetricAccumulator] = field(
        default_factory=lambda: {v: MetricAccumulator() for v in VAR_NAMES}
    )
    agl_metrics: dict[str, dict[str, MetricAccumulator]] = field(
        default_factory=lambda: {
            band[0]: {v: MetricAccumulator() for v in VAR_NAMES}
            for band in AGL_BANDS
        }
    )
    k_metrics: dict[str, list[MetricAccumulator]] = field(
        default_factory=lambda: {
            v: [MetricAccumulator() for _ in range(40)]
            for v in VAR_NAMES
        }
    )


def load_norm_overrides(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    s = raw["stats"]
    n: dict[str, float] = {}
    if "U_x" in s:
        n["U_x_offset"] = s["U_x"]["mean"]
        n["U_uv_scale"] = max(s["U_x"]["std"], 1e-3)
    if "U_y" in s:
        n["U_y_offset"] = s["U_y"]["mean"]
        n["U_uv_scale"] = max(n.get("U_uv_scale", 0.0), s["U_y"]["std"], 1e-3)
    if "U_z" in s:
        n["U_z_offset"] = s["U_z"]["mean"]
        n["U_w_scale"] = max(s["U_z"]["std"], 1e-3)
    if "T" in s:
        n["T_offset"], n["T_scale"] = s["T"]["mean"], max(s["T"]["std"], 1e-3)
    if "q" in s:
        n["q_offset"], n["q_scale"] = s["q"]["mean"], max(s["q"]["std"], 1e-6)
    if "terrain" in s:
        n["terrain_scale"] = max(s["terrain"]["std"], 1.0)
    if "z" in s:
        n["z_scale"] = max(s["z"]["std"], 1.0)
    if "agl" in s:
        n["agl_scale"] = max(s["agl"]["std"], 1.0)
    if "era5_u" in s:
        n["era5_u_offset"] = s["era5_u"]["mean"]
        n["era5_u_scale"] = max(s["era5_u"]["std"], 1.0)
    if "era5_v" in s:
        n["era5_v_offset"] = s["era5_v"]["mean"]
        n["era5_v_scale"] = max(s["era5_v"]["std"], 1.0)
    if "era5_T" in s:
        n["era5_T_offset"], n["era5_T_scale"] = (
            s["era5_T"]["mean"], max(s["era5_T"]["std"], 1.0)
        )
    if "era5_q" in s:
        n["era5_q_offset"] = s["era5_q"]["mean"]
        n["era5_q_scale"] = max(s["era5_q"]["std"], 1e-6)
    if "t2m" in s:
        n["t2m_offset"], n["t2m_scale"] = s["t2m"]["mean"], max(s["t2m"]["std"], 1.0)
    if "d2m" in s:
        n["d2m_offset"], n["d2m_scale"] = s["d2m"]["mean"], max(s["d2m"]["std"], 1.0)
    if "u10" in s:
        n["u10_offset"] = s["u10"]["mean"]
        n["u10_scale"] = max(s["u10"]["std"], 1.0)
    if "v10" in s:
        n["v10_offset"] = s["v10"]["mean"]
        n["v10_scale"] = max(s["v10"]["std"], 1.0)
    if "pressure" in s:
        n["pressure_offset"], n["pressure_scale"] = (
            s["pressure"]["mean"], max(s["pressure"]["std"], 1.0)
        )
    return n


def denormalize_fields(x: np.ndarray, norm: dict[str, float]) -> dict[str, np.ndarray]:
    return {
        "u": x[0] * norm["U_uv_scale"] + norm["U_x_offset"],
        "v": x[1] * norm["U_uv_scale"] + norm["U_y_offset"],
        "w": x[2] * norm["U_w_scale"] + norm["U_z_offset"],
        "T": x[3] * norm["T_scale"] + norm["T_offset"],
        "q": x[4] * norm["q_scale"] + norm["q_offset"],
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ck:
        raise KeyError(f"{path} does not contain a 'model' state_dict")
    return ck


def build_dataset(args, norm: dict[str, float], ck_cfg: dict[str, Any]):
    include_slopes = bool(ck_cfg.get("include_slopes", False))
    use_geo = bool(ck_cfg.get("use_geo", False))
    if args.model_type == "vit":
        return WindV2DatasetViT(
            args.data_dir,
            args.splits_yaml,
            args.split,
            norm=norm,
            include_slopes=include_slopes,
            return_geo=use_geo,
            use_residual=False,
        )
    return WindV2Dataset(
        args.data_dir,
        args.splits_yaml,
        args.split,
        norm=norm,
        include_slopes=include_slopes,
        use_residual=False,
    )


def build_model(args, ck: dict[str, Any], sample, device: torch.device):
    cfg = ck.get("config", {})
    if args.model_type == "vit":
        terrain = sample[0]
        era5 = sample[1]
        use_geo = bool(cfg.get("use_geo", False))
        geo_channels = sample[2].shape[0] if use_geo else 0
        target_idx = 3 if use_geo else 2
        nz = sample[target_idx].shape[-1]
        model = build_vit_v2(
            preset=cfg.get("preset", args.vit_preset),
            era5_input_dim=era5.shape[0],
            nz=nz,
            terrain_in_channels=terrain.shape[0],
            geo_channels=geo_channels,
        )
    else:
        inp = sample[0]
        model = FNO3D(
            in_channels=inp.shape[0],
            out_channels=5,
            width=int(cfg.get("width", args.width)),
            modes=tuple(cfg.get("modes", args.modes)),
            n_layers=int(cfg.get("n_layers", args.n_layers)),
        )
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model


def open_case_store(dataset, idx: int):
    return zarr.open_group(str(dataset.cases[idx] / "grid.zarr"), mode="r")


def get_case_id(sample) -> str:
    return str(sample[-1])


def get_target(sample, model_type: str, use_geo: bool) -> torch.Tensor:
    if model_type == "vit":
        return sample[3] if use_geo else sample[2]
    return sample[1]


def predict_one(model, sample, model_type: str, use_geo: bool, device: torch.device,
                use_amp: bool) -> torch.Tensor:
    if model_type == "vit":
        terrain = sample[0].unsqueeze(0).to(device, non_blocking=True)
        era5 = sample[1].unsqueeze(0).to(device, non_blocking=True)
        geo = sample[2].unsqueeze(0).to(device, non_blocking=True) if use_geo else None
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(terrain, era5, geo)
    else:
        inp = sample[0].unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(inp)
    return pred.squeeze(0).detach().float().cpu()


def update_metrics(acc: EvalAccumulators, pred, true, baseline, agl):
    for ci, vn in enumerate(VAR_NAMES):
        p = pred[vn]
        t = true[vn]
        b = baseline[vn]
        acc.global_metrics[vn].update(p, t, b)
        for band_name, lo, hi in AGL_BANDS:
            mask = (agl >= lo) & (agl < hi)
            acc.agl_metrics[band_name][vn].update(p, t, b, mask)
        for k in range(p.shape[-1]):
            acc.k_metrics[vn][k].update(p[..., k], t[..., k], b[..., k])


def case_metrics(case_id: str, pred, true, baseline) -> dict[str, Any]:
    out: dict[str, Any] = {"case_id": case_id}
    for vn in VAR_NAMES:
        err = pred[vn] - true[vn]
        base_err = baseline[vn] - true[vn]
        rmse = float(np.sqrt(np.mean(err * err)))
        base_rmse = float(np.sqrt(np.mean(base_err * base_err)))
        out[f"rmse_{vn}"] = rmse
        out[f"mae_{vn}"] = float(np.mean(np.abs(err)))
        out[f"bias_{vn}"] = float(np.mean(err))
        out[f"baseline_rmse_{vn}"] = base_rmse
        out[f"skill_{vn}"] = 1.0 - rmse / max(base_rmse, 1e-12)
    return out


def summarise(acc: EvalAccumulators, *, args, ck: dict[str, Any], n_cases: int,
              elapsed_s: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_type": args.model_type,
        "weights": str(args.weights),
        "split": args.split,
        "n_cases": n_cases,
        "elapsed_s": elapsed_s,
        "checkpoint_epoch": ck.get("epoch"),
        "checkpoint_val_loss": ck.get("val_loss"),
        "checkpoint_val_mse": ck.get("val_mse"),
        "checkpoint_config": ck.get("config", {}),
        "global": {vn: acc.global_metrics[vn].as_dict() for vn in VAR_NAMES},
        "agl_bands": {
            band: {vn: acc.agl_metrics[band][vn].as_dict() for vn in VAR_NAMES}
            for band, _, _ in AGL_BANDS
        },
        "k_levels": {
            vn: [m.as_dict() for m in acc.k_metrics[vn]]
            for vn in VAR_NAMES
        },
    }
    return summary


def write_outputs(output_dir: Path, summary: dict[str, Any],
                  per_case: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not per_case:
        return
    keys = list(per_case[0].keys())
    with (output_dir / "per_case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(per_case)


def evaluate(args) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = load_checkpoint(args.weights, device)
    ck_cfg = ck.get("config", {})
    norm = {**DEFAULT_NORM, **load_norm_overrides(args.norm_yaml)}
    dataset = build_dataset(args, norm, ck_cfg)
    if args.max_cases is not None:
        dataset.cases = dataset.cases[: args.max_cases]
    if len(dataset) == 0:
        raise RuntimeError(f"No cases found for split={args.split}")

    sample0 = dataset[0]
    model = build_model(args, ck, sample0, device)
    use_geo = bool(ck_cfg.get("use_geo", False))
    use_residual = bool(ck_cfg.get("use_residual", False))
    logger.info("Evaluating %s on %s: %d cases, residual=%s, geo=%s",
                args.model_type, device, len(dataset), use_residual, use_geo)

    acc = EvalAccumulators()
    per_case: list[dict[str, Any]] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            store = open_case_store(dataset, i)
            # FNO uses cuFFT internally; half precision is unsafe for the padded
            # 184x184x40 grid, so AMP is limited to ViT evaluation.
            use_amp = args.amp and args.model_type == "vit"
            pred_norm = predict_one(model, sample, args.model_type, use_geo,
                                    device, use_amp).numpy()
            true_norm = get_target(sample, args.model_type, use_geo).numpy()
            baseline_norm = dataset._build_era5_baseline_tensor(store)
            if use_residual:
                pred_abs_norm = pred_norm + baseline_norm
            else:
                pred_abs_norm = pred_norm

            pred = denormalize_fields(pred_abs_norm, norm)
            true = denormalize_fields(true_norm, norm)
            baseline = denormalize_fields(baseline_norm, norm)

            terrain = np.asarray(store["input/terrain"][:], dtype=np.float32)
            z = np.asarray(store["coords/z"][:], dtype=np.float32)
            agl = z - terrain[:, :, None]

            update_metrics(acc, pred, true, baseline, agl)
            if args.per_case:
                per_case.append(case_metrics(get_case_id(sample), pred, true, baseline))
            if (i + 1) % args.log_every == 0 or (i + 1) == len(dataset):
                logger.info("  %d/%d cases", i + 1, len(dataset))

    elapsed = time.time() - t0
    summary = summarise(acc, args=args, ck=ck, n_cases=len(dataset),
                        elapsed_s=elapsed)
    write_outputs(args.output, summary, per_case)
    logger.info("Saved metrics to %s", args.output)
    for vn in VAR_NAMES:
        g = summary["global"][vn]
        logger.info(
            "%s: RMSE=%.4g baseline=%.4g skill=%.3f MAE=%.4g bias=%.4g",
            vn, g["rmse"], g["baseline_rmse"], g["skill_vs_baseline"],
            g["mae"], g["bias"],
        )
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["fno", "vit"], required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits-yaml", type=Path, required=True)
    ap.add_argument("--norm-yaml", type=Path, default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--per-case", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--vit-preset", default="base", choices=["small", "base", "large"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, nargs=3, default=(16, 16, 8))
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
