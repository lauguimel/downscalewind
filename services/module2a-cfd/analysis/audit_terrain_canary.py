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


def finite_percentile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.nanpercentile(vals, q)) if vals.size else float("nan")


MULTI_HILL_HILLS = [
    {"id": "N", "pos_x": 0.0, "pos_y": 1500.0, "H": 250.0, "L": 800.0},
    {"id": "SE", "pos_x": 1299.0, "pos_y": -750.0, "H": 200.0, "L": 600.0},
    {"id": "SW", "pos_x": -1299.0, "pos_y": -750.0, "H": 300.0, "L": 1000.0},
]


def multi_hill_crop_mask(x: np.ndarray, y: np.ndarray, crop_km: float | None) -> np.ndarray:
    if crop_km is None or crop_km <= 0:
        return np.ones((len(x), len(y)), dtype=bool)
    half = crop_km * 1000.0
    return (np.abs(x)[:, None] <= half) & (np.abs(y)[None, :] <= half)


def multi_hill_inflow_speed_at(inflow: dict, height: float) -> float:
    val = inflow_speed_at(inflow, height)
    if math.isfinite(val):
        return val
    z = np.asarray(inflow.get("z_levels", []), dtype=float)
    spd = np.asarray(inflow.get("wind_speed_levels", []), dtype=float)
    if z.size == spd.size and z.size:
        order = np.argsort(z)
        return float(np.interp(float(height), z[order], spd[order]))
    return float("nan")


def add_row(rows: list[dict[str, object]], variant: str, height: float, mask: str, stat: str, value: object) -> None:
    rows.append({"variant": variant, "height_m": height, "mask": mask, "stat": stat, "value": value})


def add_stats(rows: list[dict[str, object]], variant: str, height: float, mask: str, values: np.ndarray, prefix: str) -> None:
    add_row(rows, variant, height, mask, f"n_{prefix}", int(np.count_nonzero(np.isfinite(values))))
    add_row(rows, variant, height, mask, f"mean_{prefix}", finite_mean(values))
    add_row(rows, variant, height, mask, f"median_{prefix}", finite_percentile(values, 50))
    add_row(rows, variant, height, mask, f"p10_{prefix}", finite_percentile(values, 10))
    add_row(rows, variant, height, mask, f"p50_{prefix}", finite_percentile(values, 50))
    add_row(rows, variant, height, mask, f"p90_{prefix}", finite_percentile(values, 90))
    add_row(rows, variant, height, mask, f"max_{prefix}", finite_max(values))


def multi_hill_masks(x: np.ndarray, y: np.ndarray, terrain: np.ndarray, mask_crop: np.ndarray, inflow: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fx, fy = flow_unit(inflow)
    z_base = float(np.nanmin(terrain))
    crest: dict[str, np.ndarray] = {}
    lee: dict[str, np.ndarray] = {}
    for hill in MULTI_HILL_HILLS:
        hid = str(hill["id"])
        s = (X - hill["pos_x"]) * fx + (Y - hill["pos_y"]) * fy
        crest[hid] = mask_crop & (terrain >= z_base + 0.85 * hill["H"])
        lee[hid] = mask_crop & (s >= 0.25 * hill["L"]) & (s <= 2.0 * hill["L"]) & (terrain <= z_base + 0.3 * hill["H"])
    return crest, lee


def write_multi_hill_figure(
    grid_zarr: Path,
    variant: str,
    x: np.ndarray,
    y: np.ndarray,
    terrain: np.ndarray,
    ratio_10: np.ndarray,
    mask_crop: np.ndarray,
    inflow: dict,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fx, fy = flow_unit(inflow)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=120)
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    im0 = axes[0].imshow(terrain.T, origin="lower", extent=extent, cmap="terrain")
    axes[0].arrow(0, 0, fx * 500.0, fy * 500.0, width=45.0, color="black", length_includes_head=True)
    axes[0].set_title("terrain")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ratio_10.T, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=2.0)
    axes[1].set_title("speed/inflow 10 m")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    vals = np.asarray(ratio_10, dtype=np.float64)[mask_crop]
    vals = vals[np.isfinite(vals)]
    axes[2].hist(vals, bins=40, range=(0.0, 2.5), density=True, color="#3b7ea1")
    axes[2].set_xlim(0.0, 2.5)
    axes[2].set_title("crop PDF")
    for ax in axes[:2]:
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
    axes[2].set_xlabel("speed/inflow")
    axes[2].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(output_dir / f"{variant}.png")
    plt.close(fig)


def audit_grid_multi_hill(
    grid_zarr: Path,
    variant: str,
    heights: list[float],
    crop_km: float,
    figure_dir: Path | None = None,
) -> list[dict[str, object]]:
    g = zarr.open_group(str(grid_zarr), mode="r")
    U = np.asarray(g["target/U"][:], dtype=np.float32)
    speed = np.hypot(U[..., 0], U[..., 1])
    terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
    z = np.asarray(g["coords/z"][:], dtype=np.float32)
    agl = z - terrain[:, :, None]
    x = np.asarray(g["coords/x"][:], dtype=np.float64)
    y = np.asarray(g["coords/y"][:], dtype=np.float64)
    inflow = load_inflow(grid_zarr.parent)
    mask_crop = multi_hill_crop_mask(x, y, crop_km)
    crest_masks, lee_masks = multi_hill_masks(x, y, terrain, mask_crop, inflow)
    z_base = float(np.nanmin(terrain))
    max_h = max(float(h["H"]) for h in MULTI_HILL_HILLS)
    flat_mask = mask_crop & (terrain <= z_base + 0.1 * max_h)

    rows: list[dict[str, object]] = []
    ratio_10 = None
    for h in heights:
        speed_h = interp_columns(agl, speed, h)
        inflow_h = multi_hill_inflow_speed_at(inflow, h)
        ratio = speed_h / max(inflow_h, 1e-6) if math.isfinite(inflow_h) else np.full_like(speed_h, np.nan)
        if abs(h - 10.0) < 1e-6:
            ratio_10 = ratio
        if abs(h - 2.0) < 1e-6:
            add_row(rows, variant, h, "comment", "known_buggy_inflow_speed_at_2m", "known_buggy_inflow_speed_at_2m")
        add_row(rows, variant, h, "crop", "inflow_speed", inflow_h)
        add_stats(rows, variant, h, "crop", speed_h[mask_crop], "speed")
        add_stats(rows, variant, h, "crop", ratio[mask_crop], "speed_to_inflow")

        crest_max = []
        crest_p90 = []
        lee_min = []
        lee_p10 = []
        for hill in MULTI_HILL_HILLS:
            hid = str(hill["id"])
            cvals = speed_h[crest_masks[hid]]
            lvals = speed_h[lee_masks[hid]]
            add_row(rows, variant, h, f"crest_{hid}", "n_cells", int(np.count_nonzero(crest_masks[hid])))
            add_row(rows, variant, h, f"crest_{hid}", "max_speed", finite_max(cvals))
            add_row(rows, variant, h, f"crest_{hid}", "p90_speed", finite_percentile(cvals, 90))
            add_row(rows, variant, h, f"lee_{hid}", "n_cells", int(np.count_nonzero(lee_masks[hid])))
            add_row(rows, variant, h, f"lee_{hid}", "min_speed", finite_min(lvals))
            add_row(rows, variant, h, f"lee_{hid}", "p10_speed", finite_percentile(lvals, 10))
            crest_max.append(finite_max(cvals))
            crest_p90.append(finite_percentile(cvals, 90))
            lee_min.append(finite_min(lvals))
            lee_p10.append(finite_percentile(lvals, 10))
        add_row(rows, variant, h, "crest", "max_speed", finite_max(np.asarray(crest_max)))
        add_row(rows, variant, h, "crest", "p90_speed", finite_max(np.asarray(crest_p90)))
        add_row(rows, variant, h, "lee", "min_speed", finite_min(np.asarray(lee_min)))
        add_row(rows, variant, h, "lee", "p10_speed", finite_min(np.asarray(lee_p10)))
        add_row(rows, variant, h, "flat", "mean_speed", finite_mean(speed_h[flat_mask]))
        add_row(rows, variant, h, "flat", "p10_speed", finite_percentile(speed_h[flat_mask], 10))

        vals = ratio[mask_crop]
        vals = vals[np.isfinite(vals)]
        counts, edges = np.histogram(vals, bins=40, range=(0.0, 2.5))
        width = edges[1] - edges[0]
        denom = float(counts.sum()) * width
        density = counts / denom if denom > 0 else np.full_like(counts, np.nan, dtype=np.float64)
        for idx, count in enumerate(counts):
            label = f"bin_{edges[idx]:.4f}_{edges[idx + 1]:.4f}"
            add_row(rows, variant, h, "pdf", f"{label}_count", int(count))
            add_row(rows, variant, h, "pdf", f"{label}_density", float(density[idx]))

    if ratio_10 is None:
        inflow_10 = multi_hill_inflow_speed_at(inflow, 10.0)
        ratio_10 = interp_columns(agl, speed, 10.0) / max(inflow_10, 1e-6)
    write_multi_hill_figure(
        grid_zarr,
        variant,
        x,
        y,
        terrain,
        ratio_10,
        mask_crop,
        inflow,
        figure_dir or (grid_zarr.parent.parent / "figures"),
    )
    return rows


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
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--canary-dir", type=Path)
    source.add_argument("--grid-zarr", type=Path)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--terrain-kind", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, default=None)
    ap.add_argument("--heights", default="2,10,20,50,100")
    ap.add_argument("--crop-km", type=float, default=2.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.grid_zarr is not None:
        if not args.variant:
            ap.error("--grid-zarr requires --variant")
        if args.terrain_kind != "multi_hill":
            ap.error("--grid-zarr requires --terrain-kind multi_hill")
        rows = audit_grid_multi_hill(
            args.grid_zarr.resolve(),
            args.variant,
            parse_heights(args.heights),
            args.crop_km,
            args.output.parent / "figures",
        )
        write_csv(args.output, rows)
        print("cases=1")
        print(f"rows={len(rows)}")
        print(f"output={args.output}")
        return 0

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
