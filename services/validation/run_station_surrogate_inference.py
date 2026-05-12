"""
Run station-level DownscaleWind UVT inference from prepared v2 grid.zarr inputs.

This script deliberately starts from input-ready `grid.zarr` stores. Building
those stores for arbitrary stations is a separate data-generation step because
the v2 surrogate expects the same native-grid contract as training:
terrain/z0, terrain-following z/AGL, ERA5 3x3 pressure/surface inputs, and
inflow metadata.

Expected manifest columns:
  - case_id
  - station_id
  - date or timestamp_utc
  - lat, lon
  - optional grid_zarr; otherwise use --grid-dir/{case_id}/grid.zarr
  - optional pres in Pa or hPa for q->RH conversion

Output columns include `t_downscaled_c`, `rh_downscaled_pct`, and
`wind_downscaled_ms`, which can be passed to build_fwi_baseline_validation.py
via --downscaled-uvt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURROGATE_DIR = PROJECT_ROOT / "services/module2b-surrogate"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURROGATE_DIR))

from evaluate_v2_physical import denormalize_fields, load_norm_overrides  # noqa: E402
from shared.fwi import specific_humidity_to_rh  # noqa: E402
from src.dataset_v2 import DEFAULT_NORM, NI, NJ, build_era5_baseline_tensor, parse_agl_levels  # noqa: E402
from src.model_vit_v2 import build_vit_v2  # noqa: E402


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        raise ValueError("Parquet manifests require pandas; pass a CSV manifest for GPU inference")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_present(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "nat", "none"}


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def as_date(row: dict[str, str]) -> str:
    if is_present(row.get("date")):
        return parse_datetime(str(row["date"])).date().isoformat()
    return parse_datetime(str(row["timestamp_utc"])).date().isoformat()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"{path} does not contain a model state_dict")
    return checkpoint


def pressure_hpa(row: dict[str, str]) -> float:
    if not is_present(row.get("pres")):
        return 1013.25
    value = float(str(row["pres"]))
    return value / 100.0 if value > 2000.0 else value


def normalise_terrain_and_geo(store, norm: dict[str, float], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terrain_raw = np.asarray(store["input/terrain"][:], dtype=np.float32)
    terrain = terrain_raw / norm["terrain_scale"]
    z0_eff = float(store["input"].attrs.get("z0_eff", 0.0)) / norm["z0_scale"]
    z0_map = np.full((NI, NJ), z0_eff, dtype=np.float32)

    parts = [terrain.astype(np.float32)]
    if bool(cfg.get("include_slopes", False)):
        slope_y, slope_x = np.gradient(terrain_raw, 33.333, 33.333)
        parts.extend([slope_x.astype(np.float32), slope_y.astype(np.float32)])
    parts.append(z0_map)
    terrain_2d = np.stack(parts, axis=0)

    target_agl_levels = parse_agl_levels(cfg.get("target_agl_levels"))
    if target_agl_levels is None:
        z = np.asarray(store["coords/z"][:], dtype=np.float32)
        agl = z - terrain_raw[:, :, None]
        levels = agl[NI // 2, NJ // 2, :].astype(np.float32)
    else:
        levels = target_agl_levels.astype(np.float32)
        agl = np.broadcast_to(levels[None, None, :], (NI, NJ, levels.size)).copy()
        z = terrain_raw[:, :, None] + agl

    geo = np.stack([z / norm["z_scale"], agl / norm["agl_scale"]], axis=0).astype(np.float32)
    return terrain_2d, geo, levels


def normalise_era5(store, norm: dict[str, float]) -> np.ndarray:
    plev = np.asarray(store["input/era5_pressure_levels"][:], dtype=np.float32)
    flat_parts = []
    for var, scale, offset in [
        ("u", norm["era5_u_scale"], norm["era5_u_offset"]),
        ("v", norm["era5_v_scale"], norm["era5_v_offset"]),
        ("T", norm["era5_T_scale"], norm["era5_T_offset"]),
        ("q", norm["era5_q_scale"], norm["era5_q_offset"]),
    ]:
        arr = np.asarray(store[f"input/era5_3d/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    for var, scale, offset in [
        ("t2m", norm["t2m_scale"], norm["t2m_offset"]),
        ("d2m", norm["d2m_scale"], norm["d2m_offset"]),
        ("u10", norm["u10_scale"], norm["u10_offset"]),
        ("v10", norm["v10_scale"], norm["v10_offset"]),
    ]:
        arr = np.asarray(store[f"input/era5_surface/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    flat_parts.append(((plev - norm["pressure_offset"]) / norm["pressure_scale"]).astype(np.float32))
    lat = float(store["input"].attrs.get("lat", 0.0)) / norm["lat_scale"]
    z0_eff = float(store["input"].attrs.get("z0_eff", 0.0)) / norm["z0_scale"]
    flat_parts.append(np.array([lat, z0_eff], dtype=np.float32))
    return np.concatenate(flat_parts).astype(np.float32)


def interp_profile(levels: np.ndarray, values: np.ndarray, target_agl: float) -> float:
    order = np.argsort(levels)
    return float(np.interp(float(target_agl), levels[order], values[order]))


def extract_station_values(fields: dict[str, np.ndarray], levels: np.ndarray, p_hpa: float) -> dict[str, float]:
    iy = NI // 2
    ix = NJ // 2
    u10 = interp_profile(levels, fields["u"][iy, ix, :], 10.0)
    v10 = interp_profile(levels, fields["v"][iy, ix, :], 10.0)
    w10 = interp_profile(levels, fields["w"][iy, ix, :], 10.0)
    t2 = interp_profile(levels, fields["T"][iy, ix, :], 2.0)
    q2 = interp_profile(levels, fields["q"][iy, ix, :], 2.0)
    rh2 = float(specific_humidity_to_rh(np.array([q2]), np.array([t2]), np.array([p_hpa]))[0])
    return {
        "u10_downscaled_ms": u10,
        "v10_downscaled_ms": v10,
        "w10_downscaled_ms": w10,
        "wind_downscaled_ms": float(np.hypot(u10, v10)),
        "t_downscaled_c": t2 - 273.15,
        "q_downscaled_kgkg": q2,
        "rh_downscaled_pct": rh2,
    }


def build_model(checkpoint: dict, sample_shapes: tuple[int, int, int], device: torch.device):
    cfg = checkpoint.get("config", {})
    terrain_channels, era5_dim, nz = sample_shapes
    model = build_vit_v2(
        preset=cfg.get("preset", "base"),
        era5_input_dim=era5_dim,
        nz=nz,
        terrain_in_channels=terrain_channels,
        geo_channels=2 if bool(cfg.get("use_geo", False)) else 0,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def infer_one(model, store, norm: dict[str, float], cfg: dict, device: torch.device, amp: bool) -> tuple[dict, np.ndarray]:
    terrain_2d, geo, levels = normalise_terrain_and_geo(store, norm, cfg)
    era5 = normalise_era5(store, norm)

    terrain_t = torch.from_numpy(terrain_2d).unsqueeze(0).to(device)
    era5_t = torch.from_numpy(era5).unsqueeze(0).to(device)
    geo_t = torch.from_numpy(geo).unsqueeze(0).to(device) if bool(cfg.get("use_geo", False)) else None

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        pred = model(terrain_t, era5_t, geo_t).squeeze(0).detach().float().cpu().numpy()

    if bool(cfg.get("use_residual", False)):
        pred = pred + build_era5_baseline_tensor(
            store,
            norm,
            pred.shape[-1],
            mode=str(cfg.get("residual_baseline_mode", "pressure_index")),
        )
    return denormalize_fields(pred, norm), levels


def resolve_grid_zarr(row: dict[str, str], grid_dir: Path | None) -> Path:
    if is_present(row.get("grid_zarr")):
        return Path(str(row["grid_zarr"]))
    if grid_dir is None:
        raise ValueError("Manifest has no grid_zarr column; pass --grid-dir")
    return grid_dir / str(row["case_id"]) / "grid.zarr"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    manifest_path: Path,
    weights_path: Path,
    norm_yaml: Path,
    output_path: Path,
    *,
    grid_dir: Path | None,
    device_name: str,
    amp: bool,
    strict: bool,
) -> None:
    manifest = load_rows(manifest_path)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(weights_path, device)
    cfg = checkpoint.get("config", {})
    norm = {**DEFAULT_NORM, **load_norm_overrides(norm_yaml)}

    rows = []
    missing = []
    model = None
    for row in manifest:
        grid_zarr = resolve_grid_zarr(row, grid_dir)
        if not grid_zarr.exists():
            missing.append({"case_id": row.get("case_id"), "station_id": row.get("station_id"), "grid_zarr": str(grid_zarr)})
            if strict:
                raise FileNotFoundError(grid_zarr)
            continue
        store = zarr.open_group(str(grid_zarr), mode="r")
        terrain_2d, geo, levels = normalise_terrain_and_geo(store, norm, cfg)
        era5 = normalise_era5(store, norm)
        if model is None:
            model = build_model(checkpoint, (terrain_2d.shape[0], era5.shape[0], geo.shape[-1]), device)
        t0 = time.time()
        fields, levels = infer_one(model, store, norm, cfg, device, amp)
        dt = time.time() - t0
        values = extract_station_values(fields, levels, pressure_hpa(row))
        values.update(
            {
                "case_id": row.get("case_id"),
                "station_id": row.get("station_id"),
                "date": as_date(row),
                "timestamp_utc": row.get("timestamp_utc"),
                "lat": float(row["lat"]) if is_present(row.get("lat")) else float(store["input"].attrs.get("lat", np.nan)),
                "lon": float(row["lon"]) if is_present(row.get("lon")) else float(store["input"].attrs.get("lon", np.nan)),
                "grid_zarr": str(grid_zarr),
                "inference_s": dt,
                "device": str(device),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_mse": checkpoint.get("val_mse"),
            }
        )
        rows.append(values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows)
    meta = {
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "norm_yaml": str(norm_yaml),
        "n_manifest": int(len(manifest)),
        "n_inferred": int(len(rows)),
        "n_missing_grid_zarr": int(len(missing)),
        "checkpoint_config": {k: str(v) for k, v in cfg.items()},
    }
    output_path.with_suffix(".run.json").write_text(json.dumps(meta, indent=2))
    if missing:
        write_csv(output_path.with_name(output_path.stem + "_missing.csv"), missing)
    print(f"manifest_rows={len(manifest)}")
    print(f"inferred={len(rows)}")
    print(f"missing_grid_zarr={len(missing)}")
    print(f"output={output_path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--norm-yaml",
        type=Path,
        default=PROJECT_ROOT / "data/campaign/complex_terrain_v1/manifests/dataset_v2_norm.yaml",
    )
    parser.add_argument("--grid-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        args.manifest,
        args.weights,
        args.norm_yaml,
        args.output,
        grid_dir=args.grid_dir,
        device_name=args.device,
        amp=args.amp,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
