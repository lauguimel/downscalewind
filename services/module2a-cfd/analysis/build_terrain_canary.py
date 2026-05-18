#!/usr/bin/env python3
"""Build terrain canaries from one prepared OpenFOAM case.

Two modes:
- ``analytic`` (default): replace the source terrain with a synthetic flat plate
  or a 2D cosine-squared ridge.  Used to validate bulk + speedup conservation.
- ``z0_treatment``: keep the real terrain, vary only the wall roughness
  treatment (WorldCover mapped, optionally capped, or uniform).  Used to
  disentangle the wall-z0 contribution from the top/pressure-gradient fixes
  inside the best-stack recovery configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = Path(__file__).resolve().parents[1]
for path in (str(MODULE_DIR), str(SCRIPT_DIR)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(MODULE_DIR))

from run_multisite_campaign import DEFAULT_MESH, write_tbm_dict  # noqa: E402

from build_top_bc_canary import patch_top_bcs, replace_patch_block  # noqa: E402
from build_wall_z0_canary import patch_block, patch_wall_fields, read_end_time  # noqa: E402
from build_wind_conservation_canary import (  # noqa: E402
    estimate_pg_from_era5_geopotential,
    estimate_pg_from_profile_wind,
    finite_float,
    load_inflow,
    parse_csv_floats,
    render_fvoptions,
)


def is_time_dir(path: Path) -> bool:
    if not path.is_dir() or path.name == "0":
        return False
    try:
        float(path.name)
    except ValueError:
        return False
    return True


def copy_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {"polyMesh", "boundaryData", "triSurface"} or item.suffix == ".zarr":
            continue
        target = dst / item.name
        if item.is_dir():
            copy_dir(item, target)
        else:
            shutil.copy2(item, target)


def copy_base_case(base_case: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in base_case.iterdir():
        if item.name.startswith("processor") or item.name.startswith("log.") or is_time_dir(item):
            continue
        if item.suffix == ".zarr":
            continue
        target = dst / item.name
        if item.is_dir():
            copy_dir(item, target)
        else:
            shutil.copy2(item, target)
    for stale in ("Cx", "Cy", "Cz"):
        p = dst / "0" / stale
        if p.exists():
            p.unlink()


def parse_tbm_scalar(text: str, key: str, default: float) -> float:
    match = re.search(rf"\b{re.escape(key)}\s+([0-9.eE+-]+)\s*;", text)
    return float(match.group(1)) if match else default


def parse_tbm_vector(text: str, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    match = re.search(rf"\b{re.escape(key)}\s+\(([^)]+)\)\s*;", text)
    if not match:
        return default
    vals = [float(x) for x in match.group(1).split()]
    return tuple(vals[:3]) if len(vals) >= 3 else default


def load_mesh_cfg(base_case: Path) -> dict:
    cfg = dict(DEFAULT_MESH)
    path = base_case / "system" / "terrainBlockMesherDict"
    if not path.exists():
        return cfg
    text = path.read_text(errors="replace")
    dims = parse_tbm_vector(text, "dimensions", (cfg["inner_size_m"], cfg["inner_size_m"], cfg["height_m"]))
    blocks = parse_tbm_vector(text, "blocks", (cfg["inner_blocks"], cfg["inner_blocks"], 1.0))
    cells = parse_tbm_vector(text, "cells", (cfg["cells_per_block_xy"], cfg["cells_per_block_xy"], cfg["cells_z"]))
    cfg["inner_size_m"] = float(dims[0])
    cfg["height_m"] = float(dims[2])
    cfg["inner_blocks"] = int(blocks[0])
    cfg["cells_per_block_xy"] = int(cells[0])
    cfg["cells_z"] = int(cells[2])
    cfg["cylinder_radius_m"] = parse_tbm_scalar(text, "radius", cfg["cylinder_radius_m"])
    cfg["radial_cells"] = int(parse_tbm_scalar(text, "radialBlockCells", cfg["radial_cells"]))
    cfg["radial_grading"] = parse_tbm_scalar(text, "radialGrading", cfg["radial_grading"])
    cfg["cylinder_sections"] = int(parse_tbm_scalar(text, "numberOfSections", cfg["cylinder_sections"]))
    cfg["blend_distance_m"] = parse_tbm_scalar(text, "dMax", cfg["blend_distance_m"])
    cfg["p_above_z"] = parse_tbm_scalar(text, "p_above", cfg["p_above_z"])
    cfg["max_dist_proj"] = parse_tbm_scalar(text, "maxDistProj", cfg["max_dist_proj"])
    cfg["grading_z"] = parse_tbm_scalar(text, "gradingFactors", cfg["grading_z"])
    return cfg


def read_stl_z_range(path: Path) -> tuple[float, float]:
    if not path.exists():
        return 0.0, 0.0
    z_vals: list[float] = []
    vertex_re = re.compile(r"\bvertex\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)")
    try:
        for line in path.read_text(errors="ignore").splitlines():
            match = vertex_re.search(line)
            if match:
                z_vals.append(float(match.group(1)))
    except UnicodeDecodeError:
        z_vals = []
    if z_vals:
        return float(min(z_vals)), float(max(z_vals))
    try:
        from stl import mesh as stl_mesh

        terrain = stl_mesh.Mesh.from_file(str(path))
        z = terrain.vectors[:, :, 2]
        return float(np.min(z)), float(np.max(z))
    except Exception:
        return 0.0, 0.0


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


def terrain_z(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    base_z: float,
    inflow: dict,
    ridge_height: float,
    ridge_half_width: float,
) -> np.ndarray:
    if kind == "flat":
        return np.full_like(x, base_z, dtype=np.float64)
    if kind != "ridge_cos2":
        raise ValueError(f"Unsupported terrain kind: {kind}")
    fx, fy = flow_unit(inflow)
    s = x * fx + y * fy
    out = np.full_like(x, base_z, dtype=np.float64)
    mask = np.abs(s) <= ridge_half_width
    out[mask] += ridge_height * np.cos(0.5 * math.pi * s[mask] / ridge_half_width) ** 2
    return out


def write_ascii_stl(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("solid terrain\n")
        for j in range(y.shape[0] - 1):
            for i in range(x.shape[1] - 1):
                p00 = (x[j, i], y[j, i], z[j, i])
                p10 = (x[j, i + 1], y[j, i + 1], z[j, i + 1])
                p01 = (x[j + 1, i], y[j + 1, i], z[j + 1, i])
                p11 = (x[j + 1, i + 1], y[j + 1, i + 1], z[j + 1, i + 1])
                for tri in ((p00, p10, p01), (p10, p11, p01)):
                    f.write("  facet normal 0 0 1\n    outer loop\n")
                    for vx, vy, vz in tri:
                        f.write(f"      vertex {vx:.3f} {vy:.3f} {vz:.3f}\n")
                    f.write("    endloop\n  endfacet\n")
        f.write("endsolid terrain\n")


def write_analytic_terrain(
    case_dir: Path,
    kind: str,
    base_z: float,
    inflow: dict,
    mesh_cfg: dict,
    stl_resolution_m: float,
    ridge_height_m: float,
    ridge_half_width_m: float,
) -> dict:
    radius = float(mesh_cfg["cylinder_radius_m"])
    extent = 2.0 * radius + 2000.0
    n = int(math.ceil(extent / stl_resolution_m)) + 1
    coords = np.linspace(-extent / 2.0, extent / 2.0, n, dtype=np.float64)
    X, Y = np.meshgrid(coords, coords)
    Z = terrain_z(kind, X, Y, base_z, inflow, ridge_height_m, ridge_half_width_m)
    stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
    write_ascii_stl(stl_path, X, Y, Z)
    system_dir = case_dir / "system"
    solver_system = {
        name: (system_dir / name).read_text()
        for name in ("controlDict", "fvSchemes", "fvSolution")
        if (system_dir / name).exists()
    }
    write_tbm_dict(mesh_cfg, "terrain.stl", case_dir, float(np.nanmin(Z) - 50.0))
    for name, text in solver_system.items():
        (system_dir / name).write_text(text)
    return {
        "terrain_base_z_m": float(base_z),
        "terrain_z_min_m": float(np.nanmin(Z)),
        "terrain_z_max_m": float(np.nanmax(Z)),
        "stl_resolution_m": float(stl_resolution_m),
        "stl_extent_m": float(extent),
        "stl_points_per_axis": int(n),
    }


MULTI_HILL_HILLS = [
    {"id": "N", "pos_x": 0.0, "pos_y": 1500.0, "H": 250.0, "L": 800.0, "shape": "cos2"},
    {"id": "SE", "pos_x": 1299.0, "pos_y": -750.0, "H": 200.0, "L": 600.0, "shape": "cos2"},
    {"id": "SW", "pos_x": -1299.0, "pos_y": -750.0, "H": 300.0, "L": 1000.0, "shape": "cos2"},
]

MULTI_HILL_VARIANTS = {
    "V0": ("inletOutlet", "zeroGrad", "OFF", 0.05, "wc", 270.0),
    "V1": ("slip", "fixedValue0", "flip", 0.005, "wc_cap_0.05", 270.0),
    "V2": ("inletOutlet", "zeroGrad", "flip", 0.005, "wc_cap_0.05", 270.0),
    "V3": ("slip", "fixedValue0", "OFF", 0.005, "wc_cap_0.05", 270.0),
    "V4": ("slip", "fixedValue0", "native", 0.005, "wc_cap_0.05", 270.0),
    "V5": ("slip", "fixedValue0", "flip", 0.05, "wc_cap_0.05", 270.0),
    "V6": ("slip", "fixedValue0", "flip", 0.005, "uniform_0.05", 270.0),
    "V7": ("slip", "zeroGrad", "flip", 0.005, "wc_cap_0.05", 270.0),
    "V8": ("inletOutlet", "zeroGrad", "flip", 0.005, "wc_cap_0.05", 270.0),
    "V0n": ("inletOutlet", "zeroGrad", "OFF", 0.05, "wc", 0.0),
    "V1n": ("slip", "fixedValue0", "flip", 0.005, "wc_cap_0.05", 0.0),
    "V9": ("inletOutlet", "zeroGrad", "flip", 0.05, "wc", 270.0),
}


def wind_components_from_dir(wind_dir_deg: float) -> tuple[float, float]:
    wd = math.radians(float(wind_dir_deg))
    return -math.sin(wd), -math.cos(wd)


def apply_wind_dir(inflow: dict, wind_dir_deg: float) -> dict:
    fx, fy = wind_components_from_dir(wind_dir_deg)
    return {**inflow, "wind_dir": float(wind_dir_deg), "flowDir_x": fx, "flowDir_y": fy}


def terrain_z_multi_hill(x: np.ndarray, y: np.ndarray, base_z: float) -> np.ndarray:
    relief = np.zeros_like(x, dtype=np.float64)
    for hill in MULTI_HILL_HILLS:
        r = np.hypot(x - hill["pos_x"], y - hill["pos_y"])
        contrib = np.where(r <= hill["L"], hill["H"] * np.cos(0.5 * math.pi * r / hill["L"]) ** 2, 0.0)
        relief = np.maximum(relief, contrib)
    return np.full_like(x, base_z, dtype=np.float64) + relief


def write_multi_hill_terrain(case_dir: Path, base_z: float, mesh_cfg: dict, stl_resolution_m: float) -> dict:
    radius = float(mesh_cfg["cylinder_radius_m"])
    extent = 2.0 * radius + 2000.0
    n = int(math.ceil(extent / stl_resolution_m)) + 1
    coords = np.linspace(-extent / 2.0, extent / 2.0, n, dtype=np.float64)
    X, Y = np.meshgrid(coords, coords)
    Z = terrain_z_multi_hill(X, Y, base_z)
    stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
    write_ascii_stl(stl_path, X, Y, Z)
    system_dir = case_dir / "system"
    solver_system = {
        name: (system_dir / name).read_text()
        for name in ("controlDict", "fvSchemes", "fvSolution")
        if (system_dir / name).exists()
    }
    write_tbm_dict(mesh_cfg, "terrain.stl", case_dir, float(np.nanmin(Z) - 50.0))
    for name, text in solver_system.items():
        (system_dir / name).write_text(text)
    return {
        "terrain_base_z_m": float(base_z),
        "terrain_z_min_m": float(np.nanmin(Z)),
        "terrain_z_max_m": float(np.nanmax(Z)),
        "stl_resolution_m": float(stl_resolution_m),
        "stl_extent_m": float(extent),
        "stl_points_per_axis": int(n),
        "hills": MULTI_HILL_HILLS,
    }


def patch_top_bcs_variant(case_dir: Path, top_u: str, top_p: str) -> None:
    u_body = {
        "slip": "        type            slip;",
        "inletOutlet": "        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform (0 0 0);",
    }[top_u]
    p_body = {
        "fixedValue0": "        type            fixedValue;\n        value           uniform 0;",
        "zeroGrad": "        type            zeroGradient;",
    }[top_p]
    replacements = {"U": u_body, "p": p_body, "p_rgh": p_body, "k": "        type            zeroGradient;", "epsilon": "        type            zeroGradient;"}
    for field, body in replacements.items():
        path = case_dir / "0" / field
        if path.exists():
            path.write_text(replace_patch_block(path.read_text(errors="replace"), "top", body))


def multi_hill_knobs(args: argparse.Namespace) -> dict:
    top_u, top_p, pg_mode, z0_wall, z0_treatment, wind_dir = MULTI_HILL_VARIANTS[args.variant]
    return {
        "top_u": args.top_u_bc or top_u,
        "top_p": args.top_p_bc or top_p,
        "pg_mode": args.pg_mode or pg_mode,
        "z0_wall": float(z0_wall),
        "z0_treatment": args.z0_treatment or z0_treatment,
        "wind_dir": float(args.wind_dir_deg if args.wind_dir_deg is not None else wind_dir),
    }


def placeholder_inflow() -> dict:
    z = [10, 50, 100, 500, 1000, 1500]
    spd = [6, 9, 11, 14, 16, 17]
    return apply_wind_dir(
        {
            "site_lat": 37.0,
            "site_lon": -5.5,
            "z_levels": z,
            "wind_speed_levels": spd,
            "u_profile": spd,
            "ux_profile": spd,
            "uy_profile": [0.0] * len(spd),
            "era5_source": "placeholder",
            "timestamp": "2020-01-01T00:00:00",
        },
        270.0,
    )


def normalize_placeholder_wall_fields(case_dir: Path) -> None:
    for field in ("nut", "epsilon"):
        path = case_dir / "0" / field
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        for patch_name in ("terrain", "bottom"):
            def update(body: str, field: str = field) -> str:
                out = body
                if field == "nut":
                    out = re.sub(r"type\s+nutkWallFunction\s*;", "type            atmNutkWallFunction;", out)
                if field == "epsilon":
                    out = re.sub(r"type\s+epsilonWallFunction\s*;", "type            atmEpsilonWallFunction;", out)
                if not re.search(r"\bz0\b", out):
                    out = re.sub(r"(type\s+\w+\s*;)", r"\1\n        z0              uniform 0.05;", out, count=1)
                return out

            text, _, _ = patch_block(text, patch_name, update)
        path.write_text(text)


def make_dry_run_base(output_dir: Path, overwrite: bool) -> Path:
    base = output_dir / "_placeholder_base"
    src = MODULE_DIR.parents[1] / "data" / "cases" / "phase1_cylinder" / "case_prod_hpc"
    copy_base_case(src, base, overwrite=overwrite)
    (base / "inflow.json").write_text(json.dumps(placeholder_inflow(), indent=2))
    normalize_placeholder_wall_fields(base)
    return base


def apply_multi_hill_z0(case_dir: Path, treatment_spec: str, z0_wall: float, nut_wall_function: str) -> dict:
    treatment = parse_z0_treatment(treatment_spec)
    if treatment["kind"] == "uniform":
        patch_wall_fields(case_dir, treatment["value_m"], nut_wall_function)
        return {"treatment": treatment_spec, "mode": "uniform", "z0_m": float(treatment["value_m"])}
    patch_wall_fields(case_dir, z0_wall, nut_wall_function)
    patch_terrain_z0_block_to_mapped(case_dir)
    return {
        "treatment": treatment_spec,
        "mode": treatment["kind"],
        "cap_m": float(treatment["cap_m"]) if treatment["cap_m"] is not None else None,
    }


def estimate_multi_hill_pg(inflow: dict, args: argparse.Namespace, pg_mode: str) -> dict:
    if pg_mode == "OFF":
        return {"enabled": False, "mode": "OFF"}
    pg_args = argparse.Namespace(**vars(args))
    pg_args.pg_sign = pg_mode
    return {**estimate_pg(inflow, pg_args), "enabled": True, "mode": pg_mode}


def make_multi_hill_variant(args: argparse.Namespace) -> tuple[dict, dict]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_case = make_dry_run_base(output_dir, args.overwrite) if args.dry_run_inflow else args.base_case.resolve()
    knobs = multi_hill_knobs(args)
    inflow = apply_wind_dir(load_inflow(base_case), knobs["wind_dir"])
    mesh_cfg = dict(DEFAULT_MESH)
    mesh_cfg.update({"inner_size_m": 6000, "inner_blocks": 30, "cells_per_block_xy": 6, "height_m": 2500, "cells_z": 40})
    base_z, _ = read_stl_z_range(base_case / "constant" / "triSurface" / "terrain.stl")

    case_name = f"case_ts000_multi_hill_{args.variant}"
    case_dir = output_dir / "cases" / case_name
    copy_base_case(base_case, case_dir, overwrite=args.overwrite)
    (case_dir / "inflow.json").write_text(json.dumps(inflow, indent=2))
    terrain_meta = write_multi_hill_terrain(case_dir, base_z, mesh_cfg, args.stl_resolution)
    patch_top_bcs_variant(case_dir, knobs["top_u"], knobs["top_p"])
    z0_info = apply_multi_hill_z0(case_dir, knobs["z0_treatment"], knobs["z0_wall"], args.nut_wall_function)
    pg = estimate_multi_hill_pg(inflow, args, knobs["pg_mode"])
    physics = {"coriolis": True}
    if pg.get("enabled"):
        physics["pressure_gradient"] = {"enabled": True, "dp_dx": pg["dp_dx"], "dp_dy": pg["dp_dy"]}
    render_fvoptions(case_dir, inflow, physics)

    top_label = "slip_fixed_p" if (knobs["top_u"], knobs["top_p"]) == ("slip", "fixedValue0") else f"{knobs['top_u']}_{knobs['top_p']}"
    variant = {
        "name": args.variant,
        "terrain_kind": f"multi_hill_{args.variant}",
        "case_name": case_name,
        "case_dir": str(case_dir),
        "description": f"multi_hill {args.variant}; top_U={knobs['top_u']}; top_p={knobs['top_p']}; pg={knobs['pg_mode']}; z0={knobs['z0_treatment']}; wind_dir={knobs['wind_dir']:g}",
        "top_bc": top_label,
        "top_u_bc": knobs["top_u"],
        "top_p_bc": knobs["top_p"],
        "pg_mode": knobs["pg_mode"],
        "wind_dir_deg": knobs["wind_dir"],
        "z0_wall_m": knobs["z0_wall"],
        "z0": z0_info,
        "pressure_gradient": pg,
        "nut_wall_function": args.nut_wall_function,
        **terrain_meta,
    }
    manifest_path = output_dir / "terrain_canary_manifest.json"
    manifest = load_existing_manifest(manifest_path)
    manifest.update({"base_case": str(base_case), "output_dir": str(output_dir), "time": args.time or read_end_time(base_case), "mesh_cfg": mesh_cfg})
    manifest = write_manifest(output_dir, manifest, variant)
    write_export_script(output_dir, manifest, inflow, manifest["time"])
    write_readme(output_dir, manifest)
    return manifest, variant


def estimate_pg(inflow: dict, args: argparse.Namespace) -> dict:
    height_band = None
    if args.geo_height_band:
        vals = parse_csv_floats(args.geo_height_band)
        if len(vals) != 2:
            raise ValueError("--geo-height-band expects lo,hi")
        height_band = (vals[0], vals[1])
    if args.force_profile_pg:
        pg = estimate_pg_from_profile_wind(inflow, args.profile_pg_height)
    else:
        try:
            pg = estimate_pg_from_era5_geopotential(
                inflow,
                requested_hpa=parse_csv_floats(args.geo_pressure_levels),
                height_band_m=height_band,
            )
        except Exception as exc:
            pg = estimate_pg_from_profile_wind(inflow, args.profile_pg_height)
            pg["fallback_reason"] = str(exc)
    if args.pg_sign == "flip":
        pg = {**pg, "dp_dx": -pg["dp_dx"], "dp_dy": -pg["dp_dy"], "sign_applied": "flip"}
    else:
        pg = {**pg, "sign_applied": "native"}
    return pg


def load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {"variants": []}
    return json.loads(path.read_text())


def write_manifest(output_dir: Path, manifest: dict, variant: dict) -> dict:
    variants = [v for v in manifest.get("variants", []) if v.get("terrain_kind") != variant["terrain_kind"]]
    variants.append(variant)
    manifest["variants"] = sorted(variants, key=lambda v: v["terrain_kind"])
    (output_dir / "terrain_canary_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def write_export_script(output_dir: Path, manifest: dict, inflow: dict, time_name: str) -> None:
    era5 = inflow.get("era5_source")
    timestamp = inflow.get("timestamp")
    lat = finite_float(inflow.get("site_lat"))
    lon = finite_float(inflow.get("site_lon"))
    script = output_dir / "export_and_audit_terrain_canary.sh"
    if not era5 or not timestamp or not math.isfinite(lat) or not math.isfinite(lon):
        script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
echo "Cannot export automatically: inflow.json lacks era5_source, timestamp, site_lat or site_lon." >&2
exit 2
"""
        )
        script.chmod(script.stat().st_mode | 0o755)
        return

    export_py = MODULE_DIR / "export_to_grid_zarr_v2.py"
    audit_py = MODULE_DIR / "analysis" / "audit_v2_teacher_wind.py"
    terrain_audit_py = MODULE_DIR / "analysis" / "audit_terrain_canary.py"
    commands: list[str] = []
    for v in manifest.get("variants", []):
        case_dir = Path(v["case_dir"]).resolve()
        commands.append(
            f'if [ -f "{case_dir / time_name / "U"}" ]; then\n'
            f'  python3 "{export_py.resolve()}" '
            f'--case-dir "{case_dir}" '
            f'--site-id "{v["name"]}" '
            f'--site-lat "{lat:.8f}" '
            f'--site-lon "{lon:.8f}" '
            f'--era5-zarr "{era5}" '
            f'--timestamp "{timestamp}" '
            f'--time "{time_name}" '
            f'--output "{case_dir / "grid.zarr"}" '
            f'--overwrite\n'
            f'else\n'
            f'  echo "skip export: {case_dir} has no {time_name}/U" >&2\n'
            f'fi'
        )

    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

{chr(10).join(commands)}

python3 "{audit_py.resolve()}" \\
  --data-dir "{(output_dir / "cases").resolve()}" \\
  --output "{(output_dir / "terrain_canary_wind_audit.csv").resolve()}" \\
  --summary-output "{(output_dir / "terrain_canary_wind_audit_summary.csv").resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2

python3 "{terrain_audit_py.resolve()}" \\
  --canary-dir "{output_dir.resolve()}" \\
  --output "{(output_dir / "terrain_canary_metrics.csv").resolve()}" \\
  --summary-output "{(output_dir / "terrain_canary_metrics_summary.json").resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def write_readme(output_dir: Path, manifest: dict) -> None:
    lines = [
        "# Terrain Canary",
        "",
        "Analytic terrain canaries with identical inflow and best-stack recovery settings.",
        "",
        "Variants:",
    ]
    for v in manifest.get("variants", []):
        lines.append(f"- `{v['case_name']}`: {v['description']}")
    lines.extend(
        [
            "",
            "Execution order on Aqua:",
            "",
            "```bash",
            "terrainBlockMesher, writeCellCentres, init_from_era5, simpleFoam, export, audit",
            "```",
            "",
            "Decision gates:",
            "",
            "- flat: `crop/inflow >= 0.95` across 2,10,20,50,100 m AGL;",
            "- ridge: `crest_max_to_inflow_10m >= 1.30` for absolute-teacher validity;",
            "- if flat < 0.85 and ridge crest < 1.15, freeze regeneration and redesign lateral BCs.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def parse_z0_treatment(spec: str) -> dict:
    """Parse a z0 treatment specifier.

    Accepted forms:
        ``wc``                -> WorldCover natif (uncapped)
        ``wc_cap_<value>``    -> WorldCover mapped, cap value applied
        ``uniform_<value>``   -> uniform z0 value, no mapping
    """
    spec = spec.strip()
    if spec == "wc":
        return {"kind": "wc", "cap_m": None}
    if spec.startswith("wc_cap_"):
        cap = float(spec[len("wc_cap_"):])
        return {"kind": "wc_capped", "cap_m": cap}
    if spec.startswith("uniform_"):
        val = float(spec[len("uniform_"):])
        return {"kind": "uniform", "value_m": val}
    raise ValueError(f"Unknown z0 treatment specifier: {spec!r}")


MAPPED_Z0_BLOCK = (
    "z0\n        {\n"
    "            type        mappedFile;\n"
    "            mapMethod   nearest;\n"
    "            fieldTable  z0;\n"
    "        }"
)


def patch_terrain_z0_block_to_mapped(case_dir: Path) -> None:
    """Switch the terrain/bottom z0 entries in 0/nut and 0/epsilon to mappedFile."""
    for field in ("nut", "epsilon"):
        path = case_dir / "0" / field
        text = path.read_text(errors="replace")
        for patch_name in ("terrain", "bottom"):
            def update(body: str) -> str:
                out, n = re.subn(
                    r"z0\s+uniform\s+[\d.eE+-]+\s*;",
                    MAPPED_Z0_BLOCK,
                    body,
                )
                return out if n else body

            text, _, _ = patch_block(text, patch_name, update)
        path.write_text(text)


def apply_z0_cap_to_boundary_data(z0_file: Path, cap_m: float) -> tuple[float, float, float]:
    """Cap the values in an OpenFOAM scalar list file (in place).

    Returns (raw_max, capped_min, capped_max) for reporting.
    """
    text = z0_file.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse scalar list: {z0_file}")
    n = int(match.group(1))
    start = match.end()
    end = text.rfind(")")
    arr = np.fromstring(text[start:end], sep="\n", count=n)
    if len(arr) != n:
        raise ValueError(f"{z0_file}: expected {n} scalars, got {len(arr)}")
    raw_max = float(arr.max())
    capped = np.minimum(arr, cap_m)
    lines = [str(n), "("]
    lines.extend(f"{v:.6e}" for v in capped)
    lines.append(")")
    z0_file.write_text("\n".join(lines) + "\n")
    return raw_max, float(capped.min()), float(capped.max())


def write_z0_treatment_export_script(
    output_dir: Path,
    manifest: dict,
    inflow: dict,
    time_name: str,
) -> None:
    """Export grid.zarr + run wind audit for z0_treatment variants."""
    era5 = inflow.get("era5_source")
    timestamp = inflow.get("timestamp")
    lat = finite_float(inflow.get("site_lat"))
    lon = finite_float(inflow.get("site_lon"))
    script = output_dir / "export_and_audit_z0_treatment.sh"
    if not era5 or not timestamp or not math.isfinite(lat) or not math.isfinite(lon):
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'echo "Cannot export automatically: inflow.json lacks era5_source, timestamp, site_lat or site_lon." >&2\n'
            "exit 2\n"
        )
        script.chmod(script.stat().st_mode | 0o755)
        return

    export_py = MODULE_DIR / "export_to_grid_zarr_v2.py"
    audit_py = MODULE_DIR / "analysis" / "audit_v2_teacher_wind.py"
    wall_audit_py = MODULE_DIR / "analysis" / "audit_wall_z0.py"
    commands: list[str] = []
    for v in manifest.get("variants", []):
        case_dir = Path(v["case_dir"]).resolve()
        commands.append(
            f'if [ -f "{case_dir / time_name / "U"}" ]; then\n'
            f'  python3 "{export_py.resolve()}" '
            f'--case-dir "{case_dir}" '
            f'--site-id "{v["name"]}" '
            f'--site-lat "{lat:.8f}" '
            f'--site-lon "{lon:.8f}" '
            f'--era5-zarr "{era5}" '
            f'--timestamp "{timestamp}" '
            f'--time "{time_name}" '
            f'--output "{case_dir / "grid.zarr"}" '
            f'--overwrite\n'
            f'  SOLVED_CASES+=("{case_dir}")\n'
            f'else\n'
            f'  echo "skip export: {case_dir} has no {time_name}/U" >&2\n'
            f'fi'
        )

    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

SOLVED_CASES=()

{chr(10).join(commands)}

python3 "{audit_py.resolve()}" \\
  --data-dir "{(output_dir / "cases").resolve()}" \\
  --output "{(output_dir / "z0_treatment_wind_audit.csv").resolve()}" \\
  --summary-output "{(output_dir / "z0_treatment_wind_audit_summary.csv").resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2

if [ "${{#SOLVED_CASES[@]}}" -gt 0 ]; then
  python3 "{wall_audit_py.resolve()}" \\
    "${{SOLVED_CASES[@]}}" \\
    --time "{time_name}" \\
    --output "{(output_dir / "z0_treatment_wall_audit.csv").resolve()}"
fi
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def make_z0_treatment_variant(args: argparse.Namespace, treatment_spec: str) -> tuple[dict, dict]:
    """Build one z0_treatment variant from a real-terrain base case.

    The mesh is NOT regenerated here; the PBS step is responsible for running
    terrainBlockMesher, then (for mapped variants) generate_z0_field.py and the
    optional cap.  The builder records all the parameters in the manifest.
    """
    base_case = args.base_case.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inflow = load_inflow(base_case)
    mesh_cfg = load_mesh_cfg(base_case)
    pg = estimate_pg(inflow, args)
    treatment = parse_z0_treatment(treatment_spec)

    case_name = f"case_ts000_{treatment_spec.replace('.', 'p')}"
    case_dir = output_dir / "cases" / case_name
    copy_base_case(base_case, case_dir, overwrite=args.overwrite)
    shutil.copy2(base_case / "inflow.json", case_dir / "inflow.json")

    # The v1 base case has a pre-built mesh; the TBM dict and STL inputs were
    # cleaned after meshing.  Copy (not symlink) the polyMesh and the lateral
    # boundaryData (cylindrical sections) into the variant so the solver runs
    # directly without re-meshing or re-initialising ERA5 inlet profiles.
    # Symlinks would break inside Apptainer because only the case_dir is
    # bind-mounted, so symlink targets under /mnt/weka/scratch/... fall outside
    # the container's filesystem.
    src_polymesh = base_case / "constant" / "polyMesh"
    if src_polymesh.exists():
        dst_polymesh = case_dir / "constant" / "polyMesh"
        if dst_polymesh.exists() or dst_polymesh.is_symlink():
            if dst_polymesh.is_symlink():
                dst_polymesh.unlink()
            else:
                shutil.rmtree(dst_polymesh)
        shutil.copytree(src_polymesh, dst_polymesh)

    src_bdata = base_case / "constant" / "boundaryData"
    if src_bdata.exists():
        dst_bdata = case_dir / "constant" / "boundaryData"
        dst_bdata.mkdir(parents=True, exist_ok=True)
        for sub in src_bdata.iterdir():
            if sub.name == "terrain":
                # Skip terrain — WC variants will write a fresh one; uniform
                # variants do not need it.
                continue
            dst_sub = dst_bdata / sub.name
            if dst_sub.exists() or dst_sub.is_symlink():
                if dst_sub.is_symlink():
                    dst_sub.unlink()
                else:
                    shutil.rmtree(dst_sub)
            shutil.copytree(sub, dst_sub)

    patch_top_bcs(case_dir)
    render_fvoptions(
        case_dir,
        inflow,
        {
            "coriolis": True,
            "pressure_gradient": {
                "enabled": True,
                "dp_dx": pg["dp_dx"],
                "dp_dy": pg["dp_dy"],
            },
        },
    )

    if treatment["kind"] == "uniform":
        patch_wall_fields(case_dir, treatment["value_m"], args.nut_wall_function)
        z0_info: dict = {
            "mode": "uniform",
            "z0_m": float(treatment["value_m"]),
        }
        z0_desc = f"uniform z0={treatment['value_m']:g} m"
    else:
        # Start from a clean uniform block so the regex switch is deterministic,
        # then replace with the mapped block.  The boundary-data file itself is
        # generated downstream in the PBS once the mesh exists.
        patch_wall_fields(case_dir, 0.05, args.nut_wall_function)
        patch_terrain_z0_block_to_mapped(case_dir)
        z0_info = {
            "mode": treatment["kind"],
            "wc_tif": str(args.worldcover.resolve()) if args.worldcover else None,
            "site_lat": float(args.site_lat) if args.site_lat is not None else finite_float(inflow.get("site_lat")),
            "site_lon": float(args.site_lon) if args.site_lon is not None else finite_float(inflow.get("site_lon")),
            "cap_m": float(treatment["cap_m"]) if treatment["cap_m"] is not None else None,
        }
        z0_desc = (
            "WorldCover mapped"
            if treatment["kind"] == "wc"
            else f"WorldCover mapped, cap={treatment['cap_m']:g} m"
        )

    variant = {
        "name": treatment_spec,
        "treatment": treatment_spec,
        "terrain_kind": "real",
        "case_name": case_name,
        "case_dir": str(case_dir),
        "description": (
            f"real terrain; {z0_desc}; slip top; p/p_rgh top fixedValue 0; pg_geo sign={pg['sign_applied']}"
        ),
        "nut_wall_function": args.nut_wall_function,
        "top_bc": "slip_fixed_p",
        "pressure_gradient": pg,
        "z0": z0_info,
    }

    manifest_path = output_dir / "z0_treatment_canary_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"variants": []}
    variants = [v for v in manifest.get("variants", []) if v.get("treatment") != treatment_spec]
    variants.append(variant)
    manifest["variants"] = sorted(variants, key=lambda v: v["treatment"])
    manifest.update(
        {
            "base_case": str(base_case),
            "output_dir": str(output_dir),
            "time": args.time or read_end_time(base_case),
            "mesh_cfg": mesh_cfg,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_z0_treatment_export_script(output_dir, manifest, inflow, manifest["time"])
    return manifest, variant


def make_variant(args: argparse.Namespace) -> tuple[dict, dict]:
    base_case = args.base_case.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inflow = load_inflow(base_case)
    mesh_cfg = load_mesh_cfg(base_case)
    pg = estimate_pg(inflow, args)
    base_z, _ = read_stl_z_range(base_case / "constant" / "triSurface" / "terrain.stl")

    case_name = f"case_ts000_{args.terrain_kind}"
    case_dir = output_dir / "cases" / case_name
    copy_base_case(base_case, case_dir, overwrite=args.overwrite)
    shutil.copy2(base_case / "inflow.json", case_dir / "inflow.json")

    terrain_meta = write_analytic_terrain(
        case_dir=case_dir,
        kind=args.terrain_kind,
        base_z=base_z,
        inflow=inflow,
        mesh_cfg=mesh_cfg,
        stl_resolution_m=args.stl_resolution,
        ridge_height_m=args.ridge_height,
        ridge_half_width_m=args.ridge_half_width,
    )
    patch_wall_fields(case_dir, args.z0_wall, args.nut_wall_function)
    patch_top_bcs(case_dir)
    render_fvoptions(
        case_dir,
        inflow,
        {
            "coriolis": True,
            "pressure_gradient": {
                "enabled": True,
                "dp_dx": pg["dp_dx"],
                "dp_dy": pg["dp_dy"],
            },
        },
    )

    variant = {
        "name": args.terrain_kind,
        "terrain_kind": args.terrain_kind,
        "case_name": case_name,
        "case_dir": str(case_dir),
        "description": (
            f"{args.terrain_kind}; z0_wall={args.z0_wall:g} m; slip top; "
            f"p/p_rgh top fixedValue 0; pg_geo sign={pg['sign_applied']}"
        ),
        "z0_wall_m": float(args.z0_wall),
        "nut_wall_function": args.nut_wall_function,
        "top_bc": "slip_fixed_p",
        "pressure_gradient": pg,
        "ridge_height_m": float(args.ridge_height) if args.terrain_kind == "ridge_cos2" else 0.0,
        "ridge_half_width_m": float(args.ridge_half_width) if args.terrain_kind == "ridge_cos2" else 0.0,
        **terrain_meta,
    }
    manifest_path = output_dir / "terrain_canary_manifest.json"
    manifest = load_existing_manifest(manifest_path)
    manifest.update(
        {
            "base_case": str(base_case),
            "output_dir": str(output_dir),
            "time": args.time or read_end_time(base_case),
            "mesh_cfg": mesh_cfg,
        }
    )
    manifest = write_manifest(output_dir, manifest, variant)
    write_export_script(output_dir, manifest, inflow, manifest["time"])
    write_readme(output_dir, manifest)
    return manifest, variant


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("analytic", "z0_treatment"), default="analytic")
    ap.add_argument("--base-case", type=Path, default=None)
    ap.add_argument("--terrain-kind", choices=("flat", "ridge_cos2", "multi_hill"), default=None,
                    help="required for --mode analytic")
    ap.add_argument("--variant", choices=tuple(MULTI_HILL_VARIANTS), default=None,
                    help="multi_hill variant id")
    ap.add_argument("--top-u-bc", choices=("slip", "inletOutlet"), default=None)
    ap.add_argument("--top-p-bc", choices=("fixedValue0", "zeroGrad"), default=None)
    ap.add_argument("--pg-mode", choices=("OFF", "flip", "native"), default=None)
    ap.add_argument("--z0-treatment", choices=("wc", "wc_cap_0.05", "uniform_0.05"), default=None)
    ap.add_argument("--wind-dir-deg", type=float, default=None)
    ap.add_argument("--dry-run-inflow", action="store_true")
    ap.add_argument("--variants", default=None,
                    help="comma-separated z0 treatments for --mode z0_treatment "
                         "(e.g. wc,wc_cap_0.05,wc_cap_0.01,uniform_0.05)")
    ap.add_argument("--worldcover", type=Path, default=None,
                    help="WorldCover GeoTIFF path; required for wc/wc_cap variants")
    ap.add_argument("--site-lat", type=float, default=None)
    ap.add_argument("--site-lon", type=float, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--z0-wall", type=float, default=0.005)
    ap.add_argument("--nut-wall-function", default="atmNutkWallFunction")
    ap.add_argument("--stl-resolution", type=float, default=30.0)
    ap.add_argument("--ridge-height", type=float, default=200.0)
    ap.add_argument("--ridge-half-width", type=float, default=1000.0)
    ap.add_argument("--pg-sign", choices=("native", "flip"), default="flip")
    ap.add_argument("--geo-pressure-levels", default="850,800,700")
    ap.add_argument("--geo-height-band", default=None)
    ap.add_argument("--force-profile-pg", action="store_true")
    ap.add_argument("--profile-pg-height", type=float, default=1500.0)
    ap.add_argument("--time", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.mode == "z0_treatment":
        if args.base_case is None:
            ap.error("--mode z0_treatment requires --base-case")
        if not args.variants:
            ap.error("--mode z0_treatment requires --variants")
        specs = [s for s in args.variants.split(",") if s.strip()]
        needs_wc = any(s == "wc" or s.startswith("wc_cap_") for s in specs)
        if needs_wc and args.worldcover is None:
            ap.error("--worldcover is required when any wc/wc_cap_* variant is requested")
        manifest = None
        for spec in specs:
            manifest, variant = make_z0_treatment_variant(args, spec)
            print(f"variant={spec} case={variant['case_dir']}")
        print(f"canary={args.output_dir.resolve()}")
        print(f"variants={len(specs)}")
        print(f"time={manifest['time'] if manifest else 'n/a'}")
        print(f"export_audit={args.output_dir.resolve() / 'export_and_audit_z0_treatment.sh'}")
        return 0

    if not args.terrain_kind:
        ap.error("--mode analytic requires --terrain-kind")
    if args.terrain_kind == "multi_hill":
        if args.variant is None:
            ap.error("--terrain-kind multi_hill requires --variant")
        if args.base_case is None and not args.dry_run_inflow:
            ap.error("--terrain-kind multi_hill requires --base-case unless --dry-run-inflow is set")
        manifest, variant = make_multi_hill_variant(args)
        print(f"canary={args.output_dir.resolve()}")
        print(f"case={variant['case_dir']}")
        print(f"terrain_kind={variant['terrain_kind']}")
        print(f"time={manifest['time']}")
        print(f"export_audit={args.output_dir.resolve() / 'export_and_audit_terrain_canary.sh'}")
        return 0
    if args.base_case is None:
        ap.error("--mode analytic requires --base-case")
    manifest, variant = make_variant(args)
    print(f"canary={args.output_dir.resolve()}")
    print(f"case={variant['case_dir']}")
    print(f"terrain_kind={variant['terrain_kind']}")
    print(f"time={manifest['time']}")
    print(f"export_audit={args.output_dir.resolve() / 'export_and_audit_terrain_canary.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
