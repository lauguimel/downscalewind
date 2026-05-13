"""
Audit v2 OpenFOAM teacher wind against ERA5/input boundary references.

This checks whether the CFD teacher itself damps near-surface wind before any
surrogate training is involved. It reads exported v2 `grid.zarr` cases and
compares target/U at fixed AGL heights with:

  - ERA5 surface u10/v10 at the centre of the stored 3x3 ERA5 block;
  - the case `inflow.json` reconstructed profile when available.

The main output is a per-case/per-height CSV plus an aggregate summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


def parse_heights(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def interp_columns(native_agl: np.ndarray, values: np.ndarray, height: float) -> np.ndarray:
    """Interpolate (ni,nj,nk) values onto one AGL height per column."""
    agl = np.asarray(native_agl, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)
    k1 = np.sum(agl < float(height), axis=-1)
    k1 = np.clip(k1, 1, agl.shape[-1] - 1).astype(np.int64)
    k0 = k1 - 1
    z0 = np.take_along_axis(agl, k0[:, :, None], axis=-1)[:, :, 0]
    z1 = np.take_along_axis(agl, k1[:, :, None], axis=-1)[:, :, 0]
    v0 = np.take_along_axis(vals, k0[:, :, None], axis=-1)[:, :, 0]
    v1 = np.take_along_axis(vals, k1[:, :, None], axis=-1)[:, :, 0]
    frac = np.clip((float(height) - z0) / np.maximum(z1 - z0, 1e-6), 0.0, 1.0)
    return v0 + (v1 - v0) * frac


def crop_mask(g, crop_km: float | None) -> np.ndarray:
    x = np.asarray(g["coords/x"][:], dtype=np.float32)
    y = np.asarray(g["coords/y"][:], dtype=np.float32)
    if crop_km is None or crop_km <= 0:
        return np.ones((len(x), len(y)), dtype=bool)
    half = crop_km * 1000.0 / 2.0
    return (np.abs(x)[:, None] <= half) & (np.abs(y)[None, :] <= half)


def load_inflow_from_grid(g) -> dict:
    case_dir = g.attrs.get("case_dir")
    if case_dir:
        p = Path(str(case_dir)) / "inflow.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
    return {}


def inflow_speed_at(inflow: dict, height: float) -> float:
    z = inflow.get("z_levels")
    if z is None:
        return float("nan")
    z_arr = np.asarray(z, dtype=float)
    if "ux_profile" in inflow and "uy_profile" in inflow:
        u = np.asarray(inflow["ux_profile"], dtype=float)
        v = np.asarray(inflow["uy_profile"], dtype=float)
        spd = np.hypot(u, v)
    elif "u_profile" in inflow:
        spd = np.asarray(inflow["u_profile"], dtype=float)
    else:
        return float("nan")
    if z_arr.size != spd.size or z_arr.size == 0:
        return float("nan")
    order = np.argsort(z_arr)
    return float(np.interp(float(height), z_arr[order], spd[order]))


def centre_surface_speed(g) -> tuple[float, float, float]:
    surf = g["input/era5_surface"]
    u10 = float(np.asarray(surf["u10"][:], dtype=np.float32)[1, 1])
    v10 = float(np.asarray(surf["v10"][:], dtype=np.float32)[1, 1])
    return u10, v10, float(np.hypot(u10, v10))


def mean_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sub = np.asarray(values, dtype=np.float64)[mask]
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "n": int(sub.size),
        "mean": float(np.nanmean(sub)),
        "p10": float(np.nanpercentile(sub, 10)),
        "p50": float(np.nanpercentile(sub, 50)),
        "p90": float(np.nanpercentile(sub, 90)),
    }


def flow_unit(inflow: dict) -> tuple[float, float]:
    """Return horizontal flow direction vector in CFD x/y coordinates."""
    fx = finite_float(inflow.get("flowDir_x"))
    fy = finite_float(inflow.get("flowDir_y"))
    norm = math.hypot(fx, fy)
    if norm > 1e-6:
        return fx / norm, fy / norm
    wind_dir = finite_float(inflow.get("wind_dir"))
    if not math.isfinite(wind_dir):
        return float("nan"), float("nan")
    wd_rad = math.radians(wind_dir)
    return -math.sin(wd_rad), -math.cos(wd_rad)


def edge_masks(shape: tuple[int, int], inflow: dict, width: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return upstream/downstream edge masks from meteorological wind_dir."""
    ni, nj = shape
    up = np.zeros((ni, nj), dtype=bool)
    down = np.zeros((ni, nj), dtype=bool)
    ix, iy = flow_unit(inflow)
    if not math.isfinite(ix) or not math.isfinite(iy):
        return up, down
    w = max(1, min(width, ni // 2, nj // 2))
    if abs(ix) >= abs(iy):
        if ix > 0:
            up[:w, :] = True
            down[-w:, :] = True
        else:
            up[-w:, :] = True
            down[:w, :] = True
    else:
        if iy > 0:
            up[:, :w] = True
            down[:, -w:] = True
        else:
            up[:, -w:] = True
            down[:, :w] = True
    return up, down


def terrain_masks(g, terrain: np.ndarray, inflow: dict, base_mask: np.ndarray) -> dict[str, np.ndarray | float]:
    """Build simple relief masks inside the requested crop."""
    x = np.asarray(g["coords/x"][:], dtype=np.float64)
    y = np.asarray(g["coords/y"][:], dtype=np.float64)
    z = np.asarray(terrain, dtype=np.float64)
    dzdx, dzdy = np.gradient(z, x, y, edge_order=1)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    base = np.asarray(base_mask, dtype=bool) & np.isfinite(z) & np.isfinite(slope_deg)
    if not base.any():
        empty = np.zeros_like(base, dtype=bool)
        return {
            "slope_deg": slope_deg,
            "windward": empty,
            "lee": empty,
            "crest": empty,
            "valley": empty,
            "relief_crop_m": float("nan"),
            "slope_crop_mean_deg": float("nan"),
            "slope_crop_p90_deg": float("nan"),
        }

    fx, fy = flow_unit(inflow)
    along_flow_slope = dzdx * fx + dzdy * fy if math.isfinite(fx) and math.isfinite(fy) else np.full_like(z, np.nan)
    slope_ref = max(3.0, float(np.nanpercentile(slope_deg[base], 60)))
    high_ref = float(np.nanpercentile(z[base], 75))
    low_ref = float(np.nanpercentile(z[base], 25))
    windward_ref = float(np.nanpercentile(along_flow_slope[base], 70)) if np.isfinite(along_flow_slope[base]).any() else float("nan")
    lee_ref = float(np.nanpercentile(along_flow_slope[base], 30)) if np.isfinite(along_flow_slope[base]).any() else float("nan")

    steep = slope_deg >= slope_ref
    windward = base & steep & np.isfinite(along_flow_slope) & (along_flow_slope >= windward_ref)
    lee = base & steep & np.isfinite(along_flow_slope) & (along_flow_slope <= lee_ref)
    crest = base & (z >= high_ref)
    valley = base & (z <= low_ref)

    return {
        "slope_deg": slope_deg,
        "windward": windward,
        "lee": lee,
        "crest": crest,
        "valley": valley,
        "relief_crop_m": float(np.nanmax(z[base]) - np.nanmin(z[base])),
        "slope_crop_mean_deg": float(np.nanmean(slope_deg[base])),
        "slope_crop_p90_deg": float(np.nanpercentile(slope_deg[base], 90)),
    }


def audit_case(grid_zarr: Path, heights: list[float], crop_km: float | None) -> list[dict[str, object]]:
    g = zarr.open_group(str(grid_zarr), mode="r")
    if "target/U" not in g:
        return []

    U = np.asarray(g["target/U"][:], dtype=np.float32)
    speed = np.hypot(U[..., 0], U[..., 1])
    terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
    z = np.asarray(g["coords/z"][:], dtype=np.float32)
    agl = z - terrain[:, :, None]
    mask_all = np.ones(terrain.shape, dtype=bool)
    mask_crop = crop_mask(g, crop_km)
    ci, cj = terrain.shape[0] // 2, terrain.shape[1] // 2

    u10, v10, era5_u10 = centre_surface_speed(g)
    inflow = load_inflow_from_grid(g)
    mask_up, mask_down = edge_masks(terrain.shape, inflow)
    relief = terrain_masks(g, terrain, inflow, mask_crop)
    z0_eff = finite_float(g["input"].attrs.get("z0_eff"))
    u_star = finite_float(inflow.get("u_star"))

    rows: list[dict[str, object]] = []
    for h in heights:
        speed_h = interp_columns(agl, speed, h)
        u_h = interp_columns(agl, U[..., 0], h)
        v_h = interp_columns(agl, U[..., 1], h)
        all_stats = mean_stats(speed_h, mask_all)
        crop_stats = mean_stats(speed_h, mask_crop)
        upstream_stats = mean_stats(speed_h, mask_up) if mask_up.any() else {}
        downstream_stats = mean_stats(speed_h, mask_down) if mask_down.any() else {}
        windward_stats = mean_stats(speed_h, relief["windward"]) if np.asarray(relief["windward"]).any() else {}
        lee_stats = mean_stats(speed_h, relief["lee"]) if np.asarray(relief["lee"]).any() else {}
        crest_stats = mean_stats(speed_h, relief["crest"]) if np.asarray(relief["crest"]).any() else {}
        valley_stats = mean_stats(speed_h, relief["valley"]) if np.asarray(relief["valley"]).any() else {}
        center_speed = float(speed_h[ci, cj])
        inflow_h = inflow_speed_at(inflow, h)
        crop_values = np.asarray(speed_h, dtype=np.float64)[mask_crop]
        crop_values = crop_values[np.isfinite(crop_values)]
        frac_crop_above_inflow = (
            float(np.mean(crop_values >= inflow_h)) if crop_values.size and math.isfinite(inflow_h) else float("nan")
        )
        rows.append(
            {
                "grid_zarr": str(grid_zarr),
                "case": grid_zarr.parent.name,
                "site_id": g.attrs.get("site_id", ""),
                "height_agl_m": h,
                "terrain_relief_crop_m": relief["relief_crop_m"],
                "terrain_slope_crop_mean_deg": relief["slope_crop_mean_deg"],
                "terrain_slope_crop_p90_deg": relief["slope_crop_p90_deg"],
                "era5_u10_ms": era5_u10,
                "era5_u10_u_ms": u10,
                "era5_u10_v_ms": v10,
                "inflow_speed_ms": inflow_h,
                "z0_eff_m": z0_eff,
                "u_star_ms": u_star,
                "cfd_speed_all_mean_ms": all_stats["mean"],
                "cfd_speed_all_p10_ms": all_stats["p10"],
                "cfd_speed_all_p50_ms": all_stats["p50"],
                "cfd_speed_all_p90_ms": all_stats["p90"],
                "cfd_speed_crop_mean_ms": crop_stats["mean"],
                "cfd_speed_crop_p10_ms": crop_stats["p10"],
                "cfd_speed_crop_p50_ms": crop_stats["p50"],
                "cfd_speed_crop_p90_ms": crop_stats["p90"],
                "cfd_speed_center_ms": center_speed,
                "cfd_speed_upstream_edge_mean_ms": upstream_stats.get("mean", float("nan")),
                "cfd_speed_downstream_edge_mean_ms": downstream_stats.get("mean", float("nan")),
                "cfd_speed_windward_mean_ms": windward_stats.get("mean", float("nan")),
                "cfd_speed_windward_p90_ms": windward_stats.get("p90", float("nan")),
                "cfd_speed_lee_mean_ms": lee_stats.get("mean", float("nan")),
                "cfd_speed_lee_p90_ms": lee_stats.get("p90", float("nan")),
                "cfd_speed_crest_mean_ms": crest_stats.get("mean", float("nan")),
                "cfd_speed_crest_p90_ms": crest_stats.get("p90", float("nan")),
                "cfd_speed_valley_mean_ms": valley_stats.get("mean", float("nan")),
                "cfd_speed_valley_p90_ms": valley_stats.get("p90", float("nan")),
                "cfd_u_center_ms": float(u_h[ci, cj]),
                "cfd_v_center_ms": float(v_h[ci, cj]),
                "fraction_crop_above_inflow": frac_crop_above_inflow,
                "ratio_crop_to_era5_u10": crop_stats["mean"] / max(era5_u10, 1e-6),
                "ratio_center_to_era5_u10": center_speed / max(era5_u10, 1e-6),
                "ratio_upstream_edge_to_inflow": upstream_stats.get("mean", float("nan")) / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_downstream_edge_to_inflow": downstream_stats.get("mean", float("nan")) / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_crop_to_inflow": crop_stats["mean"] / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_center_to_inflow": center_speed / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_windward_mean_to_inflow": windward_stats.get("mean", float("nan")) / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_crest_p90_to_inflow": crest_stats.get("p90", float("nan")) / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
                "ratio_valley_mean_to_inflow": valley_stats.get("mean", float("nan")) / max(inflow_h, 1e-6)
                if math.isfinite(inflow_h)
                else float("nan"),
            }
        )
    return rows


def discover_cases(data_dir: Path, limit: int, seed: int) -> list[Path]:
    cases = sorted(data_dir.glob("*_case_ts*/grid.zarr"))
    if not cases:
        cases = sorted(data_dir.glob("case_ts*/grid.zarr"))
    if limit and limit < len(cases):
        rng = random.Random(seed)
        cases = sorted(rng.sample(cases, limit))
    return cases


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    heights = sorted({float(r["height_agl_m"]) for r in rows})
    for h in heights:
        sub = [r for r in rows if float(r["height_agl_m"]) == h]
        row: dict[str, object] = {"height_agl_m": h, "n": len(sub)}
        for key in [
            "era5_u10_ms",
            "inflow_speed_ms",
            "cfd_speed_crop_mean_ms",
            "cfd_speed_center_ms",
            "ratio_crop_to_era5_u10",
            "ratio_center_to_era5_u10",
            "ratio_crop_to_inflow",
            "ratio_center_to_inflow",
            "ratio_upstream_edge_to_inflow",
            "ratio_downstream_edge_to_inflow",
            "ratio_windward_mean_to_inflow",
            "ratio_crest_p90_to_inflow",
            "ratio_valley_mean_to_inflow",
            "fraction_crop_above_inflow",
            "terrain_relief_crop_m",
            "terrain_slope_crop_mean_deg",
            "terrain_slope_crop_p90_deg",
            "z0_eff_m",
            "u_star_ms",
        ]:
            vals = np.asarray([finite_float(r.get(key)) for r in sub], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_median"] = float(np.median(vals))
                row[f"{key}_p10"] = float(np.percentile(vals, 10))
                row[f"{key}_p90"] = float(np.percentile(vals, 90))
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_median"] = float("nan")
                row[f"{key}_p10"] = float("nan")
                row[f"{key}_p90"] = float("nan")
        out.append(row)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heights", default="2,10,50,100")
    ap.add_argument("--crop-km", type=float, default=2.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    cases = discover_cases(args.data_dir, args.limit, args.seed)
    rows: list[dict[str, object]] = []
    for idx, grid_zarr in enumerate(cases):
        if idx % 25 == 0:
            print(f"[{idx}/{len(cases)}] {grid_zarr}")
        try:
            rows.extend(audit_case(grid_zarr, parse_heights(args.heights), args.crop_km))
        except Exception as exc:
            rows.append({"grid_zarr": str(grid_zarr), "case": grid_zarr.parent.name, "error": str(exc)})

    write_csv(args.output, rows)
    summary = aggregate([r for r in rows if "error" not in r])
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.csv")
    write_csv(summary_path, summary)
    print(f"cases={len(cases)}")
    print(f"rows={len(rows)}")
    print(f"output={args.output}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
