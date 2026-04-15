"""
validate_debug_cases.py — Validate complex_terrain_v1 debug cases against expected gates.

Reads `debug_cases.yaml` (spec) and the grid.zarr outputs from each debug run,
then checks each validation gate (solver convergence, T/u at sensor, AR mesh,
inflow surface anchor, etc.). Writes a markdown report.

Usage
-----
    python services/module2a-cfd/validate_debug_cases.py \\
        --debug-dir data/campaign/complex_terrain_v1/debug \\
        --spec data/campaign/complex_terrain_v1/debug/debug_cases.yaml \\
        --output data/campaign/complex_terrain_v1/debug/validation_report.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


logger = logging.getLogger(__name__)


SENSOR_HEIGHT_DEFAULT_M = 11.0     # typical ICOS mast height above canopy


def load_grid_zarr(case_dir: Path):
    """Return zarr group for grid.zarr if exists, else None."""
    import zarr
    # Debug case has site_id-based subdir from run_multisite_campaign; look deep
    candidates = list(case_dir.glob("**/grid.zarr"))
    if not candidates:
        return None
    return zarr.open_group(str(candidates[0]), mode="r")


def load_inflow_json(case_dir: Path) -> Optional[dict]:
    candidates = list(case_dir.glob("**/inflow.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def interp_at_height(z_agl: np.ndarray, values: np.ndarray, height_m: float) -> float:
    """Linear interp of 1D profile at given AGL height."""
    if z_agl.ndim != 1 or values.ndim != 1:
        return float("nan")
    return float(np.interp(height_m, z_agl, values))


def check_convergence(attrs: dict) -> dict:
    """Check solver convergence attribute."""
    conv = attrs.get("solver.converged", False)
    res_U = attrs.get("solver.final_residual_U", np.nan)
    res_p = attrs.get("solver.final_residual_p", np.nan)
    return {
        "passed": bool(conv),
        "detail": f"residuals U={res_U:.2e}, p={res_p:.2e}",
    }


def check_AR(attrs: dict, max_AR: float = 10.0) -> dict:
    """Check mesh aspect ratio of first cell."""
    AR = attrs.get("mesh.source_cfd_AR_first", np.nan)
    if not np.isfinite(AR):
        return {"passed": False, "detail": "AR not stored in attrs"}
    passed = AR < max_AR
    return {
        "passed": bool(passed),
        "detail": f"AR_first={AR:.2f} {'<' if passed else '>='} {max_AR}",
    }


def check_T_range(
    zarr_store,
    attrs: dict,
    expected_K_range: list[float],
    sensor_h: float = SENSOR_HEIGHT_DEFAULT_M,
) -> dict:
    """Check CFD T at sensor height is within range."""
    if "target/T" not in zarr_store:
        return {"passed": False, "detail": "target/T missing"}
    T = np.array(zarr_store["target/T"][:])      # (ny, nx, nz)
    z_agl = np.array(zarr_store["coords/z_agl"][:])
    T_center = T[T.shape[0] // 2, T.shape[1] // 2, :]
    T_sensor = interp_at_height(z_agl, T_center, sensor_h)
    t_min, t_max = expected_K_range
    passed = (t_min <= T_sensor <= t_max)
    return {
        "passed": passed,
        "detail": f"T@{sensor_h:.0f}m={T_sensor:.1f}K, expected [{t_min}, {t_max}]",
    }


def check_u_range(
    zarr_store,
    expected_ms_range: list[float],
    sensor_h: float = SENSOR_HEIGHT_DEFAULT_M,
) -> dict:
    """Check CFD wind magnitude at sensor height is within range."""
    if "target/U" not in zarr_store:
        return {"passed": False, "detail": "target/U missing"}
    U = np.array(zarr_store["target/U"][:])       # (ny, nx, nz, 3)
    z_agl = np.array(zarr_store["coords/z_agl"][:])
    u_center = U[U.shape[0] // 2, U.shape[1] // 2, :, 0]
    v_center = U[U.shape[0] // 2, U.shape[1] // 2, :, 1]
    spd_center = np.hypot(u_center, v_center)
    spd_sensor = interp_at_height(z_agl, spd_center, sensor_h)
    u_min, u_max = expected_ms_range
    passed = (u_min <= spd_sensor <= u_max)
    return {
        "passed": passed,
        "detail": f"|u|@{sensor_h:.0f}m={spd_sensor:.2f} m/s, expected [{u_min}, {u_max}]",
    }


def check_inflow_surface_anchor(inflow: Optional[dict], tolerance: float = 0.05) -> dict:
    """Check that inflow u/v at z=10m matches ERA5 u10/v10 within tolerance."""
    if inflow is None:
        return {"passed": False, "detail": "inflow.json missing"}
    u10 = inflow.get("u10_ms")
    v10 = inflow.get("v10_ms")
    z_m = inflow.get("z_m")
    u_ms = inflow.get("u_ms")
    v_ms = inflow.get("v_ms")
    if any(v is None for v in (u10, v10, z_m, u_ms, v_ms)):
        return {"passed": False, "detail": "u10/v10 or profile missing from inflow.json"}
    u10, v10 = float(u10), float(v10)
    z_arr = np.asarray(z_m)
    u_arr = np.asarray(u_ms)
    v_arr = np.asarray(v_ms)
    u_at_10 = float(np.interp(10.0, z_arr, u_arr))
    v_at_10 = float(np.interp(10.0, z_arr, v_arr))
    err_u = abs(u_at_10 - u10) / max(abs(u10), 0.5)
    err_v = abs(v_at_10 - v10) / max(abs(v10), 0.5)
    passed = err_u < tolerance and err_v < tolerance
    return {
        "passed": passed,
        "detail": f"u10 error={err_u:.2%}, v10 error={err_v:.2%} (tol {tolerance:.0%})",
    }


def check_inflow_T_anchor(inflow: Optional[dict], tolerance_K: float = 1.0) -> dict:
    """Check that inflow T at z=2m matches ERA5 t2m within tolerance."""
    if inflow is None:
        return {"passed": False, "detail": "inflow.json missing"}
    t2m = inflow.get("t2m_K")
    z_m = inflow.get("z_m")
    T_K = inflow.get("T_K")
    if any(v is None for v in (t2m, z_m, T_K)):
        return {"passed": False, "detail": "t2m or T profile missing"}
    t2m = float(t2m)
    z_arr = np.asarray(z_m)
    T_arr = np.asarray(T_K)
    T_at_2 = float(np.interp(2.0, z_arr, T_arr))
    err = abs(T_at_2 - t2m)
    passed = err <= tolerance_K
    return {
        "passed": passed,
        "detail": f"T(2m)={T_at_2:.2f}K vs t2m={t2m:.2f}K, err={err:.2f}K (tol {tolerance_K:.1f}K)",
    }


def validate_case(case_dir: Path, spec: dict) -> dict:
    """Run all gates for one debug case, return report dict."""
    grid = load_grid_zarr(case_dir)
    inflow = load_inflow_json(case_dir)

    expected = spec.get("expected_outcomes", {})

    report: dict[str, Any] = {
        "case_name": case_dir.name,
        "site_id": spec.get("site_id", ""),
        "timestamp": spec.get("timestamp", ""),
        "grid_zarr_found": grid is not None,
        "inflow_json_found": inflow is not None,
        "gates": {},
    }

    if grid is not None:
        attrs = dict(grid.attrs)
        report["gates"]["convergence"] = check_convergence(attrs)
        report["gates"]["AR"] = check_AR(attrs, max_AR=10.0)
        if "T_surface_K" in expected:
            report["gates"]["T_range"] = check_T_range(grid, attrs, expected["T_surface_K"])
        if "u10_ms_abs" in expected:
            report["gates"]["u_range"] = check_u_range(grid, expected["u10_ms_abs"])
    else:
        report["gates"]["convergence"] = {"passed": False, "detail": "grid.zarr not found"}

    report["gates"]["inflow_uv_anchor"] = check_inflow_surface_anchor(inflow)
    report["gates"]["inflow_T_anchor"] = check_inflow_T_anchor(inflow)

    report["all_passed"] = all(g["passed"] for g in report["gates"].values())
    return report


def render_markdown(reports: list[dict], out_path: Path) -> None:
    lines = [
        "# complex_terrain_v1 — Debug cases validation",
        f"Generated {dt.datetime.utcnow().isoformat()}Z",
        "",
    ]
    any_failed = False
    for r in reports:
        status = "✅ PASS" if r["all_passed"] else "❌ FAIL"
        if not r["all_passed"]:
            any_failed = True
        lines.append(f"## {r['case_name']}  {status}")
        lines.append(f"- site_id: `{r['site_id']}`  |  timestamp: `{r['timestamp']}`")
        lines.append(f"- grid.zarr found: {r['grid_zarr_found']}  |  inflow.json found: {r['inflow_json_found']}")
        lines.append("")
        lines.append("| Gate | Result | Detail |")
        lines.append("|------|--------|--------|")
        for g_name, g_res in r["gates"].items():
            mark = "✅" if g_res["passed"] else "❌"
            lines.append(f"| {g_name} | {mark} | {g_res.get('detail', '')} |")
        lines.append("")
    lines.append("---")
    lines.append(f"**Overall**: {'❌ FAIL — pipeline NOT ready for production' if any_failed else '✅ PASS — pipeline OK'}")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.spec) as f:
        spec = yaml.safe_load(f)

    reports: list[dict] = []
    for case_name, case_spec in spec.get("cases", {}).items():
        case_dir = args.debug_dir / case_name
        if not case_dir.exists():
            logger.warning("Case dir missing: %s", case_dir)
            reports.append({
                "case_name": case_name,
                "site_id": case_spec.get("site_id", ""),
                "timestamp": case_spec.get("timestamp", ""),
                "grid_zarr_found": False,
                "inflow_json_found": False,
                "gates": {"existence": {"passed": False, "detail": "case dir missing"}},
                "all_passed": False,
            })
            continue
        reports.append(validate_case(case_dir, case_spec))

    render_markdown(reports, args.output)
    logger.info("Report: %s", args.output)

    all_passed = all(r["all_passed"] for r in reports)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
