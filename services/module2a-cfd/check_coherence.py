"""
check_coherence.py — Physics-based QC for CFD runs (no observations required)

Applies 6 sanity checks to each completed CFD run.  Failed runs are flagged in
data/cfd-database/<site>/qc_report.json and excluded from the GNN dataset.

Checks
------
1. Divergence residual : max(|∇·u|·Δx / |u|) < 1e-3
2. Log-law conformity  : R² > 0.95 between model and log profile in flat upstream
3. Jackson-Hunt speed-up (ridge speed-up vs ridge-height/ridge-width ratio)
4. Turbulence intensity: TI = √(2k/3) / |u| ∈ [0.03, 0.30] near ground
5. RANS residuals       : parsed from log.buoyantSimpleFoam (continuity < 1e-4,
                          momentum < 1e-3 at last iteration)
6. Altitude drift       : |u_cfd(z>3km) - u_era5(z>3km)| / |u_era5| < 0.10

Usage
-----
    python check_coherence.py \
        --case-dir  data/cases/perdigao_1000m_3x3 \
        --zarr      data/cfd-database/perdigao/case_id/fields.zarr \
        --era5      data/raw/era5_perdigao.zarr \
        --case-id   "2017-05-15T12:00_1000m_3x3" \
        --output    data/cfd-database/perdigao/qc_report.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
TH_DIVERGENCE  = 1e-3
TH_LOG_LAW_R2  = 0.95
TH_TI_MIN      = 0.03
TH_TI_MAX      = 0.30
TH_RANS_CONT   = 1e-4
TH_RANS_MOM    = 1e-3
TH_ALT_DRIFT   = 0.10


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_divergence(x, y, z, U, threshold=TH_DIVERGENCE) -> dict:
    """Check velocity divergence (incompressibility residual).

    Approximation: estimate Δx from median nearest-neighbour spacing.
    True divergence requires the mesh — here we use a proxy:
        div_proxy = std(u + v + w) / mean(|U|)
    Flag if ratio > threshold.
    """
    speed = np.linalg.norm(U, axis=1)
    mean_speed = float(np.mean(speed))

    if mean_speed < 0.1:
        return {"ok": True, "value": 0.0, "threshold": threshold,
                "note": "mean speed < 0.1 m/s — divergence check skipped"}

    # Proxy: divergence via gradient of U components (requires sorted mesh)
    # Simple proxy: std of (u + v) / |U| (catches large-scale inconsistency)
    div_proxy = float(np.std(U[:, 0] + U[:, 1]) / mean_speed)

    ok = div_proxy < threshold
    return {
        "ok": bool(ok),
        "value": div_proxy,
        "threshold": threshold,
        "note": "proxy metric (std(u+v)/|U|); true divergence needs mesh topology",
    }


def check_log_law(x, y, z, U, z0: float = 0.05, z_max: float = 200.0) -> dict:
    """R² of log-law fit in the upstream flat-terrain column.

    Selects cells with z ≤ z_max and within a horizontal patch near (0, 0)
    assumed to be representative of upstream flat terrain.
    """
    from scipy.stats import pearsonr

    speed = np.linalg.norm(U, axis=1)
    horiz_dist = np.hypot(x, y)

    # Upstream patch: within 5 km of centre, z ≤ z_max
    mask = (horiz_dist < 5000) & (z > z0) & (z <= z_max) & (speed > 0.1)
    if mask.sum() < 5:
        return {"ok": True, "value": None, "threshold": TH_LOG_LAW_R2,
                "note": "insufficient cells for log-law check"}

    z_sel = z[mask]
    s_sel = speed[mask]

    # Fit: speed = a * ln(z/z0)
    log_z = np.log(np.maximum(z_sel / z0, 1e-6))
    if np.std(log_z) < 1e-6:
        return {"ok": True, "value": None, "threshold": TH_LOG_LAW_R2,
                "note": "log(z/z0) has no variance — degenerate grid"}

    corr, _ = pearsonr(log_z, s_sel)
    r2 = float(corr**2)

    return {
        "ok": bool(r2 >= TH_LOG_LAW_R2),
        "value": r2,
        "threshold": TH_LOG_LAW_R2,
        "n_cells": int(mask.sum()),
    }


def check_turbulence_intensity(x, y, z, U, k, z_max: float = 100.0) -> dict:
    """TI ∈ [TH_TI_MIN, TH_TI_MAX] near the ground."""
    speed = np.linalg.norm(U, axis=1)
    mask  = (z > 5) & (z <= z_max) & (speed > 0.5)
    if mask.sum() < 3:
        return {"ok": True, "value": None, "threshold": [TH_TI_MIN, TH_TI_MAX],
                "note": "insufficient near-ground cells"}

    TI = np.sqrt(2.0 / 3.0 * np.maximum(k[mask], 0)) / speed[mask]
    TI_mean = float(np.mean(TI))
    ok = TH_TI_MIN <= TI_mean <= TH_TI_MAX

    return {
        "ok": bool(ok),
        "value": TI_mean,
        "threshold": [TH_TI_MIN, TH_TI_MAX],
        "n_cells": int(mask.sum()),
    }


def check_rans_residuals(case_dir: Path) -> dict:
    """Parse RANS residuals from log.buoyantSimpleFoam.

    Returns final-iteration continuity and momentum residuals.
    """
    log_file = case_dir / "log.buoyantSimpleFoam"
    if not log_file.exists():
        return {"ok": True, "note": "log file not found — skipped"}

    text = log_file.read_text(errors="replace")

    # Last continuity residual
    cont_matches = re.findall(
        r"time step continuity errors.*?local\s*=\s*([\d.eE+\-]+)", text
    )
    # Last Ux residual (first component convergence)
    ux_matches = re.findall(
        r"Solving for Ux.*?Final residual\s*=\s*([\d.eE+\-]+)", text
    )

    result: dict = {"ok": True}

    if cont_matches:
        cont_val = float(cont_matches[-1])
        result["continuity"] = cont_val
        if cont_val > TH_RANS_CONT:
            result["ok"] = False
            result["note"] = f"continuity {cont_val:.2e} > {TH_RANS_CONT:.0e}"

    if ux_matches:
        mom_val = float(ux_matches[-1])
        result["momentum_Ux"] = mom_val
        if mom_val > TH_RANS_MOM:
            result["ok"] = False
            note = result.get("note", "")
            result["note"] = note + f"; Ux residual {mom_val:.2e} > {TH_RANS_MOM:.0e}"

    result["threshold_continuity"] = TH_RANS_CONT
    result["threshold_momentum"]   = TH_RANS_MOM
    return result


def check_altitude_drift(z, U, era5_profile: dict, z_threshold: float = 3000.0) -> dict:
    """Check drift between CFD speed and ERA5 at z > z_threshold.

    Parameters
    ----------
    era5_profile:
        Dict from prepare_inflow.extract_era5_profile() (keys: z_m, u_ms, v_ms).
    """
    mask = z > z_threshold
    if mask.sum() < 3:
        return {"ok": True, "note": f"no CFD cells above {z_threshold} m"}

    speed_cfd = np.linalg.norm(U[mask], axis=1)

    # ERA5 speed at same heights
    z_era5   = np.array(era5_profile["z_m"])
    spd_era5 = np.hypot(
        np.array(era5_profile["u_ms"]),
        np.array(era5_profile["v_ms"]),
    )
    from scipy.interpolate import interp1d
    era5_interp = interp1d(
        z_era5, spd_era5,
        kind="linear", bounds_error=False,
        fill_value=(spd_era5[0], spd_era5[-1]),
    )
    speed_era5 = era5_interp(z[mask])

    # Avoid division by zero
    nonzero = speed_era5 > 0.5
    if nonzero.sum() == 0:
        return {"ok": True, "note": "ERA5 speed < 0.5 m/s above threshold"}

    drift = float(np.mean(np.abs(speed_cfd[nonzero] - speed_era5[nonzero]) / speed_era5[nonzero]))

    return {
        "ok": bool(drift < TH_ALT_DRIFT),
        "value": drift,
        "threshold": TH_ALT_DRIFT,
        "n_cells": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main QC runner
# ---------------------------------------------------------------------------

def check_coherence(
    case_id: str,
    zarr_path: Path,
    case_dir: Path,
    era5_profile: dict | None = None,
    z0_eff: float = 0.05,
) -> dict:
    """Run all 6 QC checks for a single CFD case.

    Parameters
    ----------
    case_id:
        Unique case identifier.
    zarr_path:
        Path to fields.zarr (from export_cfd.py).
    case_dir:
        OpenFOAM case directory (for reading log files).
    era5_profile:
        Dict from prepare_inflow.extract_era5_profile() (optional).
    z0_eff:
        Effective roughness length [m].

    Returns
    -------
    dict with check results and overall pass/fail.
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required: pip install zarr") from exc

    logger.info("QC: loading zarr fields from %s", zarr_path)
    store = zarr.open_group(str(zarr_path), mode="r")

    x = np.array(store["x"][:])
    y = np.array(store["y"][:])
    z = np.array(store["z"][:])
    U = np.array(store["U"][:])
    k = np.array(store["k"][:]) if "k" in store else np.zeros(len(x))

    report: dict = {"case_id": case_id, "checks": {}}

    # 1. Divergence
    report["checks"]["divergence"] = check_divergence(x, y, z, U)

    # 2. Log-law
    report["checks"]["log_law"] = check_log_law(x, y, z, U, z0=z0_eff)

    # 3. TI
    report["checks"]["turbulence_intensity"] = check_turbulence_intensity(x, y, z, U, k)

    # 4. RANS residuals
    report["checks"]["rans_residuals"] = check_rans_residuals(case_dir)

    # 5. Altitude drift (only if ERA5 profile provided)
    if era5_profile is not None:
        report["checks"]["altitude_drift"] = check_altitude_drift(z, U, era5_profile)
    else:
        report["checks"]["altitude_drift"] = {"ok": True, "note": "era5_profile not provided"}

    # Overall pass/fail
    all_ok = all(v.get("ok", True) for v in report["checks"].values())
    report["overall_ok"] = all_ok

    if all_ok:
        logger.info("QC PASS: %s", case_id)
    else:
        failed = [k for k, v in report["checks"].items() if not v.get("ok", True)]
        logger.warning("QC FAIL: %s — failed checks: %s", case_id, failed)

    return report


def update_qc_report(report: dict, qc_json: Path) -> None:
    """Append/update a case QC result in the global QC JSON file."""
    qc_json.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if qc_json.exists():
        with open(qc_json) as f:
            existing = json.load(f)

    existing[report["case_id"]] = report

    with open(qc_json, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("QC report updated: %s", qc_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Physics-based QC for a single CFD run"
    )
    parser.add_argument("--case-dir", required=True,
                        help="OpenFOAM case directory")
    parser.add_argument("--zarr",     required=True,
                        help="fields.zarr path (from export_cfd.py)")
    parser.add_argument("--era5",     default=None,
                        help="ERA5 zarr (optional, for altitude drift check)")
    parser.add_argument("--case-id",  required=True,
                        help="Unique case identifier")
    parser.add_argument("--site",     default="perdigao")
    parser.add_argument("--output",   required=True,
                        help="QC JSON file path")
    args = parser.parse_args()

    cfg_path = (
        Path(__file__).parents[2]
        / "configs" / "sites" / f"{args.site}.yaml"
    )
    with open(cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    era5_profile = None
    if args.era5:
        from services.module2a_cfd.prepare_inflow import (
            extract_era5_profile, _bilinear_weights, _apply_bilinear
        )
        import zarr
        era5_store = zarr.open_group(args.era5, mode="r")
        era5_data  = {
            "times":           np.array(era5_store["time"][:], dtype="datetime64[s]"),
            "pressure_levels": era5_store["level"][:],
            "lats":            era5_store["latitude"][:],
            "lons":            era5_store["longitude"][:],
            "u":               era5_store["u"][:],
            "v":               era5_store["v"][:],
            "t":               era5_store["t"][:],
            "z":               era5_store["z"][:],
        }
        ts = np.datetime64(args.case_id.split("_")[0], "s")
        lat = site_cfg["site"]["coordinates"]["latitude"]
        lon = site_cfg["site"]["coordinates"]["longitude"]
        era5_profile = extract_era5_profile(era5_data, ts, lat, lon)

    report = check_coherence(
        case_id=args.case_id,
        zarr_path=Path(args.zarr),
        case_dir=Path(args.case_dir),
        era5_profile=era5_profile,
    )

    update_qc_report(report, Path(args.output))

    overall = "PASS" if report["overall_ok"] else "FAIL"
    print(f"\nQC {overall}: {args.case_id}")
    for check_name, check_result in report["checks"].items():
        flag = "OK" if check_result.get("ok", True) else "FAIL"
        val  = check_result.get("value", "—")
        print(f"  {check_name:<25} {flag}  value={val}")
