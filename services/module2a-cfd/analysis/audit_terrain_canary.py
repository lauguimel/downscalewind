#!/usr/bin/env python3
"""Audit analytic flat/ridge terrain canaries exported as grid.zarr."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr

from audit_v2_teacher_wind import crop_mask, finite_float, inflow_speed_at, interp_columns


def parse_heights(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def flow_unit(inflow: dict) -> tuple[float, float]:
    fx = finite_float(inflow.get("flowDir_x"))
    fy = finite_float(inflow.get("flowDir_y"))
    norm = math.hypot(fx, fy)
    if norm > 1e-6:
        return fx / norm, fy / norm
    wind_dir = finite_float(inflow.get("wind_dir"))
    if not math.isfinite(wind_dir):
        return 1.0, 0.0
    wd = math.radians(wind_dir)
    return -math.sin(wd), -math.cos(wd)


def load_inflow(case_dir: Path) -> dict:
    path = case_dir / "inflow.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "terrain_canary_manifest.json"
    if not path.exists():
        return {"variants": []}
    return json.loads(path.read_text())


def variant_by_case(manifest: dict) -> dict[str, dict]:
    return {str(v.get("case_name")): v for v in manifest.get("variants", [])}


def finite_min(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.min(vals)) if vals.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else float("nan")


def finite_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def audit_grid(grid_zarr: Path, variant: dict, heights: list[float], crop_km: float) -> list[dict[str, object]]:
    g = zarr.open_group(str(grid_zarr), mode="r")
    U = np.asarray(g["target/U"][:], dtype=np.float32)
    speed = np.hypot(U[..., 0], U[..., 1])
    terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
    z = np.asarray(g["coords/z"][:], dtype=np.float32)
    agl = z - terrain[:, :, None]
    x = np.asarray(g["coords/x"][:], dtype=np.float64)
    y = np.asarray(g["coords/y"][:], dtype=np.float64)

    case_dir = grid_zarr.parent
    inflow = load_inflow(case_dir)
    mask_crop = crop_mask(g, crop_km)
    ci, cj = terrain.shape[0] // 2, terrain.shape[1] // 2
    kind = str(variant.get("terrain_kind", "unknown"))

    rows: list[dict[str, object]] = []
    for h in heights:
        speed_h = interp_columns(agl, speed, h)
        inflow_h = inflow_speed_at(inflow, h)
        crop_vals = speed_h[mask_crop]
        row = {
            "case": grid_zarr.parent.name,
            "terrain_kind": kind,
            "height_agl_m": h,
            "inflow_speed_ms": inflow_h,
            "crop_speed_mean_ms": finite_mean(crop_vals),
            "center_speed_ms": float(speed_h[ci, cj]),
            "crop_to_inflow": finite_mean(crop_vals) / max(inflow_h, 1e-6)
            if math.isfinite(inflow_h)
            else float("nan"),
            "center_to_inflow": float(speed_h[ci, cj]) / max(inflow_h, 1e-6)
            if math.isfinite(inflow_h)
            else float("nan"),
            "terrain_relief_crop_m": float(np.nanmax(terrain[mask_crop]) - np.nanmin(terrain[mask_crop])),
        }
        rows.append(row)

    if kind == "ridge_cos2":
        fx, fy = flow_unit(inflow)
        X, Y = np.meshgrid(x, y, indexing="ij")
        s = X * fx + Y * fy
        ridge_half_width = float(variant.get("ridge_half_width_m", 1000.0))
        h0 = float(variant.get("ridge_height_m", 200.0))
        z_base = float(variant.get("terrain_base_z_m", np.nanmin(terrain)))
        speed_10 = interp_columns(agl, speed, 10.0)
        inflow_10 = inflow_speed_at(inflow, 10.0)

        crest_mask = mask_crop & (terrain >= z_base + 0.95 * h0)
        lee_mask = mask_crop & (s >= 0.25 * ridge_half_width) & (s <= 1.50 * ridge_half_width)
        rows.append(
            {
                "case": grid_zarr.parent.name,
                "terrain_kind": kind,
                "height_agl_m": 10.0,
                "metric": "ridge_crest_max",
                "inflow_speed_ms": inflow_10,
                "crest_speed_max_ms": finite_max(speed_10[crest_mask]),
                "crest_max_to_inflow": finite_max(speed_10[crest_mask]) / max(inflow_10, 1e-6)
                if math.isfinite(inflow_10)
                else float("nan"),
                "n_crest_cells": int(np.count_nonzero(crest_mask)),
            }
        )
        rows.append(
            {
                "case": grid_zarr.parent.name,
                "terrain_kind": kind,
                "height_agl_m": 10.0,
                "metric": "ridge_lee_min",
                "inflow_speed_ms": inflow_10,
                "lee_speed_min_ms": finite_min(speed_10[lee_mask]),
                "lee_min_to_inflow": finite_min(speed_10[lee_mask]) / max(inflow_10, 1e-6)
                if math.isfinite(inflow_10)
                else float("nan"),
                "n_lee_cells": int(np.count_nonzero(lee_mask)),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    flat_rows = [
        r for r in rows
        if r.get("terrain_kind") == "flat" and "crop_to_inflow" in r and "center_to_inflow" in r
    ]
    ridge_crest = [r for r in rows if r.get("metric") == "ridge_crest_max"]
    ridge_lee = [r for r in rows if r.get("metric") == "ridge_lee_min"]
    summary = {
        "flat": {
            "crop_to_inflow_by_height": {
                str(r["height_agl_m"]): r["crop_to_inflow"] for r in flat_rows
            },
            "center_to_inflow_by_height": {
                str(r["height_agl_m"]): r["center_to_inflow"] for r in flat_rows
            },
        },
        "ridge": {
            "crest_max_to_inflow_10m": ridge_crest[0].get("crest_max_to_inflow")
            if ridge_crest
            else None,
            "lee_min_to_inflow_10m": ridge_lee[0].get("lee_min_to_inflow") if ridge_lee else None,
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canary-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, default=None)
    ap.add_argument("--heights", default="2,10,20,50,100")
    ap.add_argument("--crop-km", type=float, default=2.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    canary_dir = args.canary_dir.resolve()
    manifest = load_manifest(canary_dir)
    variants = variant_by_case(manifest)
    rows: list[dict[str, object]] = []
    for grid_zarr in sorted((canary_dir / "cases").glob("case_ts*/grid.zarr")):
        variant = variants.get(grid_zarr.parent.name, {})
        rows.extend(audit_grid(grid_zarr, variant, parse_heights(args.heights), args.crop_km))

    write_csv(args.output, rows)
    write_summary(args.summary_output or args.output.with_suffix(".summary.json"), rows)
    print(f"cases={len(sorted((canary_dir / 'cases').glob('case_ts*/grid.zarr')))}")
    print(f"rows={len(rows)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
