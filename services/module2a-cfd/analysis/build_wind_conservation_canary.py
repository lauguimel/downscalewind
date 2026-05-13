#!/usr/bin/env python3
"""
Build OpenFOAM canary variants for near-surface wind conservation tests.

The canary starts from one already prepared OpenFOAM case and creates variants
that differ only in the momentum-forcing terms:

  - control: unchanged case
  - pg_geo: geostrophic pressure-gradient source from ERA5 geopotential
  - pg_geo_flip: sign-flipped pressure-gradient source, for sign validation
  - mean_force: diagnostic meanVelocityForce source

The script does not run OpenFOAM by default. It writes shell scripts in the
output directory to run, export grid.zarr, and audit the variants.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from jinja2 import Environment, FileSystemLoader


MODULE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = MODULE_DIR / "templates" / "openfoam"
OMEGA = 7.2921e-5
G = 9.80665


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def coriolis_parameter(latitude_deg: float) -> float:
    return 2.0 * OMEGA * math.sin(math.radians(latitude_deg))


def load_inflow(case_dir: Path) -> dict:
    path = case_dir / "inflow.json"
    if not path.exists():
        raise FileNotFoundError(f"inflow.json not found in {case_dir}")
    return json.loads(path.read_text())


def profile_wind_at(inflow: dict, height_m: float) -> tuple[float, float]:
    z = np.asarray(inflow.get("z_levels", []), dtype=float)
    ux = np.asarray(inflow.get("ux_profile", []), dtype=float)
    uy = np.asarray(inflow.get("uy_profile", []), dtype=float)
    if z.size == 0 or ux.size != z.size or uy.size != z.size:
        raise ValueError("inflow.json needs z_levels, ux_profile, uy_profile")
    order = np.argsort(z)
    u = float(np.interp(height_m, z[order], ux[order]))
    v = float(np.interp(height_m, z[order], uy[order]))
    return u, v


def pressure_levels(inflow: dict, n_levels: int) -> np.ndarray:
    raw = inflow.get("pressure_hPa")
    if raw is None:
        return np.full(n_levels, np.nan, dtype=float)
    levels = np.asarray(raw, dtype=float)
    if levels.size != n_levels:
        return np.full(n_levels, np.nan, dtype=float)
    return levels


def local_xy_from_latlon(lats: np.ndarray, lons: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    x = (lon2 - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat2 - lat0) * 110_540.0
    return x, y


def fit_geopotential_gradient(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit Phi=a*x+b*y+c and return dPhi/dx, dPhi/dy, rmse."""
    mask = np.isfinite(phi) & np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        raise ValueError("Need at least 4 finite ERA5 grid points for plane fit")
    A = np.column_stack([x[mask].ravel(), y[mask].ravel(), np.ones(int(mask.sum()))])
    coeff, *_ = np.linalg.lstsq(A, phi[mask].ravel(), rcond=None)
    pred = A @ coeff
    rmse = float(np.sqrt(np.mean((pred - phi[mask].ravel()) ** 2)))
    return float(coeff[0]), float(coeff[1]), rmse


def select_geo_levels(
    inflow: dict,
    z_geo: np.ndarray,
    requested_hpa: list[float],
    height_band_m: tuple[float, float] | None,
) -> list[int]:
    n_levels = z_geo.shape[-1]
    levels = pressure_levels(inflow, n_levels)
    selected: list[int] = []

    if requested_hpa and np.isfinite(levels).any():
        for p in requested_hpa:
            idx = int(np.nanargmin(np.abs(levels - p)))
            if idx not in selected:
                selected.append(idx)
        return selected

    if height_band_m is not None:
        z_mean = np.nanmean(z_geo, axis=(0, 1))
        lo, hi = height_band_m
        selected = [int(i) for i in np.where((z_mean >= lo) & (z_mean <= hi))[0]]
        if selected:
            return selected

    if np.isfinite(levels).any():
        selected = [int(i) for i, p in enumerate(levels) if 700.0 <= p <= 900.0]
    if not selected:
        z_mean = np.nanmean(z_geo, axis=(0, 1))
        selected = [int(i) for i in np.argsort(np.abs(z_mean - 1500.0))[:3]]
    return selected


def estimate_pg_from_era5_geopotential(
    inflow: dict,
    requested_hpa: list[float],
    height_band_m: tuple[float, float] | None,
) -> dict:
    grid = inflow.get("era5_grid")
    if not grid:
        raise ValueError("inflow.json has no era5_grid block")

    lats = np.asarray(grid["lats"], dtype=float)
    lons = np.asarray(grid["lons"], dtype=float)
    z_geo = np.asarray(grid["z_geo"], dtype=float)
    if z_geo.ndim != 3:
        raise ValueError("era5_grid.z_geo must have shape (lat, lon, level)")

    lat0 = finite_float(inflow.get("site_lat"), float(np.nanmean(lats)))
    lon0 = finite_float(inflow.get("site_lon"), float(np.nanmean(lons)))
    f = coriolis_parameter(lat0)
    if abs(f) < 1e-8:
        raise ValueError(f"Coriolis parameter too small at latitude {lat0}")

    x, y = local_xy_from_latlon(lats, lons, lat0, lon0)
    levels = pressure_levels(inflow, z_geo.shape[-1])
    selected = select_geo_levels(inflow, z_geo, requested_hpa, height_band_m)

    rows = []
    ax_values = []
    ay_values = []
    for idx in selected:
        phi = G * z_geo[:, :, idx]
        dphidx, dphidy, rmse = fit_geopotential_gradient(phi, x, y)
        # Pressure-gradient acceleration in x/y. This is what fvOptions adds to U.
        ax = -dphidx
        ay = -dphidy
        ug = -dphidy / f
        vg = dphidx / f
        ax_values.append(ax)
        ay_values.append(ay)
        rows.append(
            {
                "level_index": idx,
                "pressure_hPa": float(levels[idx]) if np.isfinite(levels[idx]) else float("nan"),
                "z_geo_mean_m": float(np.nanmean(z_geo[:, :, idx])),
                "dPhi_dx_ms2": dphidx,
                "dPhi_dy_ms2": dphidy,
                "source_x_ms2": ax,
                "source_y_ms2": ay,
                "u_geostrophic_ms": ug,
                "v_geostrophic_ms": vg,
                "plane_rmse_m2s2": rmse,
            }
        )

    return {
        "method": "era5_geopotential_plane",
        "latitude_deg": lat0,
        "longitude_deg": lon0,
        "f_coriolis_s-1": f,
        "dp_dx": float(np.nanmedian(ax_values)),
        "dp_dy": float(np.nanmedian(ay_values)),
        "levels": rows,
    }


def estimate_pg_from_profile_wind(inflow: dict, height_m: float) -> dict:
    lat = finite_float(inflow.get("site_lat"))
    if not math.isfinite(lat):
        raise ValueError("site_lat missing; required for profile-wind pressure gradient fallback")
    f = coriolis_parameter(lat)
    ux, uy = profile_wind_at(inflow, height_m)
    return {
        "method": "profile_wind_geostrophic_approx",
        "latitude_deg": lat,
        "f_coriolis_s-1": f,
        "reference_height_m": float(height_m),
        "u_ref_ms": ux,
        "v_ref_ms": uy,
        "dp_dx": float(-f * uy),
        "dp_dy": float(f * ux),
    }


def read_end_time(case_dir: Path, default: str = "500") -> str:
    path = case_dir / "system" / "controlDict"
    if not path.exists():
        return default
    match = re.search(r"\bendTime\s+([^;]+);", path.read_text())
    return match.group(1).strip() if match else default


def is_time_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name == "0":
        return False
    try:
        float(path.name)
    except ValueError:
        return False
    return True


def copy_dir(src: Path, dst: Path, *, symlink_static: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if symlink_static and item.name in {"polyMesh", "boundaryData", "triSurface"}:
                target.symlink_to(item.resolve(), target_is_directory=True)
            else:
                shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target)


def copy_base_case(base_case: Path, dst: Path, *, symlink_static: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for item in base_case.iterdir():
        name = item.name
        if name.startswith("processor") or name.startswith("log.") or is_time_dir(item):
            continue
        target = dst / name
        if item.is_dir():
            copy_dir(item, target, symlink_static=symlink_static)
        else:
            shutil.copy2(item, target)


def render_fvoptions(case_dir: Path, inflow: dict, physics: dict) -> None:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR / "constant")), keep_trailing_newline=True)
    tmpl = env.get_template("fvOptions.j2")
    ctx = {
        "physics": physics,
        "solver": {},
        "canopy": {"enabled": False},
        "site": {
            "latitude": finite_float(inflow.get("site_lat"), 45.0),
            "longitude": finite_float(inflow.get("site_lon"), 0.0),
        },
        "inflow": inflow,
    }
    out = tmpl.render(**ctx)
    path = case_dir / "constant" / "fvOptions"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)


def make_variants(
    base_case: Path,
    output_dir: Path,
    inflow: dict,
    pg: dict,
    mean_force_height_m: float,
    overwrite: bool,
    symlink_static: bool,
    include_sign_flip: bool,
) -> list[dict]:
    ux_mf, uy_mf = profile_wind_at(inflow, mean_force_height_m)
    variants = [
        {
            "name": "control",
            "kind": "control",
            "physics": None,
            "description": "unchanged base case",
        },
        {
            "name": "pg_geo",
            "kind": "pressure_gradient",
            "physics": {
                "coriolis": True,
                "pressure_gradient": {
                    "enabled": True,
                    "dp_dx": pg["dp_dx"],
                    "dp_dy": pg["dp_dy"],
                },
            },
            "description": "ERA5 geostrophic pressure-gradient source",
        },
        {
            "name": "mean_force",
            "kind": "mean_velocity_force",
            "physics": {
                "coriolis": True,
                "mean_velocity_force": {
                    "enabled": True,
                    "ubar_x": ux_mf,
                    "ubar_y": uy_mf,
                    "ubar_z": 0.0,
                    "relaxation": 0.2,
                },
            },
            "description": f"diagnostic meanVelocityForce at {mean_force_height_m:g} m inflow vector",
        },
    ]
    if include_sign_flip:
        variants.insert(
            2,
            {
                "name": "pg_geo_flip",
                "kind": "pressure_gradient_sign_flip",
                "physics": {
                    "coriolis": True,
                    "pressure_gradient": {
                        "enabled": True,
                        "dp_dx": -pg["dp_dx"],
                        "dp_dy": -pg["dp_dy"],
                    },
                },
                "description": "sign-flipped pressure-gradient source for OpenFOAM sign validation",
            },
        )

    out = []
    cases_root = output_dir / "cases"
    for variant in variants:
        case_name = f"case_ts000_{variant['name']}"
        dst = cases_root / case_name
        copy_base_case(base_case, dst, symlink_static=symlink_static, overwrite=overwrite)
        if variant["physics"] is not None:
            render_fvoptions(dst, inflow, variant["physics"])
        shutil.copy2(base_case / "inflow.json", dst / "inflow.json")
        item = {
            **variant,
            "case_name": case_name,
            "case_dir": str(dst),
        }
        out.append(item)
    return out


def write_run_script(output_dir: Path, variants: list[dict], n_cores: int) -> None:
    script = output_dir / "run_canary_local_of.sh"
    reconstruct = MODULE_DIR / "reconstruct_fields.py"
    case_lines = "\n".join(f'  "{Path(v["case_dir"]).resolve()}"' for v in variants)
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

# Requires an OpenFOAM environment where decomposePar, mpirun and simpleFoam are available.
# On Aqua, run this inside the same OpenFOAM container/module used for production cases.

N_CORES="${{N_CORES:-{n_cores}}}"
PYTHON="${{PYTHON:-python3}}"
RECONSTRUCT="{reconstruct.resolve()}"

CASES=(
{case_lines}
)

for CASE in "${{CASES[@]}}"; do
  echo "== $CASE =="
  cd "$CASE"
  if [ -f constant/momentumTransport ] && [ ! -f constant/turbulenceProperties ]; then
    cp constant/momentumTransport constant/turbulenceProperties
  fi
  rm -rf processor*
  decomposePar -force 2>&1 | tee log.decomposePar
  if [ -d constant/boundaryData ]; then
    for d in processor*/; do
      ln -sfn ../../constant/boundaryData "$d/constant/boundaryData"
    done
  fi
  mpirun --allow-run-as-root --oversubscribe -np "$N_CORES" simpleFoam -parallel 2>&1 | tee log.simpleFoam
  "$PYTHON" "$RECONSTRUCT" --case-dir . --time latest --write-foam --fields U T q k epsilon nut p p_rgh
done
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def write_export_script(output_dir: Path, variants: list[dict], inflow: dict, time_name: str) -> None:
    era5 = inflow.get("era5_source")
    timestamp = inflow.get("timestamp")
    lat = finite_float(inflow.get("site_lat"))
    lon = finite_float(inflow.get("site_lon"))
    script = output_dir / "export_and_audit_canary.sh"
    audit_csv = output_dir / "canary_wind_audit.csv"
    summary_csv = output_dir / "canary_wind_audit_summary.csv"

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
    commands = []
    for v in variants:
        case_dir = Path(v["case_dir"]).resolve()
        commands.append(
            f'python3 "{export_py.resolve()}" '
            f'--case-dir "{case_dir}" '
            f'--site-id "{v["name"]}" '
            f'--site-lat "{lat:.8f}" '
            f'--site-lon "{lon:.8f}" '
            f'--era5-zarr "{era5}" '
            f'--timestamp "{timestamp}" '
            f'--time "{time_name}" '
            f'--output "{case_dir / "grid.zarr"}" '
            f'--overwrite'
        )
    command_block = "\n".join(commands)

    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

{command_block}

python3 "{audit_py.resolve()}" \\
  --data-dir "{(output_dir / "cases").resolve()}" \\
  --output "{audit_csv.resolve()}" \\
  --summary-output "{summary_csv.resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def write_readme(output_dir: Path, pg: dict, variants: list[dict]) -> None:
    lines = [
        "# Wind Conservation Canary",
        "",
        "Variants built from one base OpenFOAM case:",
    ]
    for v in variants:
        lines.append(f"- `{v['case_name']}`: {v['description']}")
    lines.extend(
        [
            "",
            "Pressure-gradient estimate:",
            "",
            "```json",
            json.dumps(pg, indent=2),
            "```",
            "",
            "Run sequence:",
            "",
            "```bash",
            f"bash {output_dir / 'run_canary_local_of.sh'}",
            f"bash {output_dir / 'export_and_audit_canary.sh'}",
            "```",
            "",
            "Decision rule:",
            "",
            "- `control` should reproduce the current damping.",
            "- one of `pg_geo` or `pg_geo_flip` must restore flat/bulk wind; the other validates the sign.",
            "- `mean_force` is a diagnostic clamp, not the publication candidate.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-case", type=Path, required=True, help="Prepared OpenFOAM case with inflow.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-static", action="store_true",
                    help="Copy polyMesh/boundaryData/triSurface instead of symlinking them")
    ap.add_argument("--geo-pressure-levels", default="850,800,700",
                    help="Comma-separated ERA5 pressure levels to use for geopotential plane fit")
    ap.add_argument("--geo-height-band", default=None,
                    help="Optional absolute geopotential-height band, e.g. 500,2500. Used if pressure levels unavailable.")
    ap.add_argument("--force-profile-pg", action="store_true",
                    help="Ignore era5_grid and estimate pressure-gradient from free-profile wind")
    ap.add_argument("--profile-pg-height", type=float, default=1500.0)
    ap.add_argument("--mean-force-height", type=float, default=80.0)
    ap.add_argument("--no-sign-flip", action="store_true")
    ap.add_argument("--n-cores", type=int, default=24)
    ap.add_argument("--time", default=None, help="OpenFOAM time directory for export; default: system/controlDict endTime")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_case = args.base_case.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inflow = load_inflow(base_case)

    height_band = None
    if args.geo_height_band:
        vals = parse_csv_floats(args.geo_height_band)
        if len(vals) != 2:
            raise ValueError("--geo-height-band expects two values: lo,hi")
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

    variants = make_variants(
        base_case=base_case,
        output_dir=output_dir,
        inflow=inflow,
        pg=pg,
        mean_force_height_m=args.mean_force_height,
        overwrite=args.overwrite,
        symlink_static=not args.copy_static,
        include_sign_flip=not args.no_sign_flip,
    )

    time_name = args.time or read_end_time(base_case)
    manifest = {
        "base_case": str(base_case),
        "output_dir": str(output_dir),
        "time": time_name,
        "pressure_gradient": pg,
        "variants": variants,
    }
    (output_dir / "canary_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_run_script(output_dir, variants, args.n_cores)
    write_export_script(output_dir, variants, inflow, time_name)
    write_readme(output_dir, pg, variants)

    print(f"canary={output_dir}")
    print(f"variants={len(variants)}")
    print(f"pg_method={pg['method']}")
    print(f"dp_dx={pg['dp_dx']:.6e} dp_dy={pg['dp_dy']:.6e}")
    print(f"run={output_dir / 'run_canary_local_of.sh'}")
    print(f"export_audit={output_dir / 'export_and_audit_canary.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
