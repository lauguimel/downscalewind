#!/usr/bin/env python3
"""
sf_to_bbsf.py — Full pipeline: simpleFoam precursor → BBSF

Deletes and recreates the case from scratch every time.
No stale files, no ambiguity.

Usage:
    python sf_to_bbsf.py --config ../../configs/phase0_stability.yaml --case A \
        --sf-iter 3000 --bbsf-iter 10000 --nprocs 4

Steps:
    1. Generate OF case from config (via run_local_study --generate-only)
    2. Run simpleFoam precursor (serial or parallel)
    3. Convert SF fields → BBSF (p→p_rgh, add T, fix alphat)
    4. Run BBSF (parallel)
    5. Generate convergence report + plots
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("sf_to_bbsf")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CMU = 0.09
KAPPA = 0.41
G_ACC = 9.81


# ---------------------------------------------------------------------------
# Docker helper
# ---------------------------------------------------------------------------
def run_docker(case_dir: Path, command: str, nprocs: int = 1, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command in Docker."""
    image = "kraken/openfoam-ai:latest"

    if nprocs > 1:
        full_cmd = (
            f"foamDictionary system/decomposeParDict "
            f"-entry numberOfSubdomains -set {nprocs} && "
            f"decomposePar && "
            f'for d in processor*/; do ln -sf ../../constant/boundaryData "$d/constant/"; done && '
            f"mpirun --allow-run-as-root -np {nprocs} {command} -parallel"
        )
    else:
        full_cmd = command

    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{case_dir}:/case", image,
         "bash", "-c", f"cd /case && {full_cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return result


# ---------------------------------------------------------------------------
# Step 1: Generate case
# ---------------------------------------------------------------------------
def step_generate(config_path: Path, case_id: str, project_root: Path) -> Path:
    """Generate OF case as full BBSF (regardless of original config solver).

    Forces solver=buoyantBoussinesqSimpleFoam and thermal=true so the
    template generates all BBSF files (p_rgh, T, alphat with correct BCs).
    """
    logger.info("=== Step 1: Generate case (forced BBSF) ===")

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override case solver to BBSF
    if case_id in cfg.get("cases", {}):
        cfg["cases"][case_id]["solver"] = "buoyantBoussinesqSimpleFoam"
        cfg["cases"][case_id]["thermal"] = True

    # Write temporary config
    tmp_config = config_path.parent / f".tmp_bbsf_{config_path.name}"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    try:
        result = subprocess.run(
            [sys.executable, "run_local_study.py", str(tmp_config),
             "--generate-only", "--cases", case_id],
            capture_output=True, text=True,
            cwd=project_root / "services" / "module2a-cfd",
        )
        if result.returncode != 0:
            logger.error("Generate failed:\n%s", result.stderr[-500:])
            raise RuntimeError("Case generation failed")
    finally:
        tmp_config.unlink(missing_ok=True)

    cases_dir = project_root / "data" / "cases" / cfg["study"]["name"]
    case_dir = cases_dir / f"case_{case_id}"
    logger.info("Case dir: %s", case_dir)
    return case_dir


# ---------------------------------------------------------------------------
# Step 2: Run simpleFoam precursor
# ---------------------------------------------------------------------------
def step_sf_precursor(case_dir: Path, sf_iter: int, nprocs: int) -> None:
    """Run simpleFoam precursor on the BBSF case.

    The case is generated as a full BBSF case. We only change the solver
    to simpleFoam in controlDict. All BCs, fields, and settings remain
    identical. This ensures the SF-converged solution is directly
    compatible with BBSF (same mesh, same BCs, same reference frame).
    """
    logger.info("=== Step 2: simpleFoam precursor (%d iter, %d cores) ===", sf_iter, nprocs)

    # Mesh if needed
    if not (case_dir / "constant" / "polyMesh" / "points").exists():
        logger.info("Meshing with cartesianMesh...")
        mesh_image = "microfluidica/openfoam:latest"
        r = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64",
             "-v", f"{case_dir}:/case", mesh_image,
             "bash", "-c", "cd /case && cartesianMesh > log.cartesianMesh 2>&1"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"cartesianMesh failed: {r.stderr[-300:]}")
        logger.info("Mesh done")

    # Run init_from_era5 to write boundaryData + patch internalField
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    init_script = Path(__file__).parent / "init_from_era5.py"
    if init_script.exists() and inflow_json.exists():
        run_docker(case_dir, "postProcess -func writeCellCentres -time 0", nprocs=1)
        logger.info("Running init_from_era5 for boundaryData + internalField...")
        r = subprocess.run(
            [sys.executable, str(init_script),
             "--case-dir", str(case_dir),
             "--inflow", str(inflow_json)],
            capture_output=True, text=True, cwd=case_dir,
        )
        if r.returncode != 0:
            logger.warning("init_from_era5 failed: %s", r.stderr[-300:])
        else:
            logger.info("ERA5 height-varying profiles written")

    # Ensure div(phi,T) in fvSchemes (needed for scalarTransport)
    _fix_fvschemes(case_dir)

    # simpleFoam reads "p" not "p_rgh" — create 0/p from 0/p_rgh
    p_rgh_path = case_dir / "0" / "p_rgh"
    p_path = case_dir / "0" / "p"
    if p_rgh_path.exists() and not p_path.exists():
        p_text = p_rgh_path.read_text()
        p_text = p_text.replace("object p_rgh;", "object p;")
        p_text = p_text.replace("object  p_rgh;", "object  p;")
        p_path.write_text(p_text)
        logger.info("Created 0/p from 0/p_rgh for simpleFoam")

    # fvSolution: add entries missing for SF+scalarTransport(T)
    fvs_path = case_dir / "system" / "fvSolution"
    fvs_text = fvs_path.read_text()
    modified = False
    # Add p relaxation/residual alongside p_rgh
    if "p_rgh" in fvs_text and "\n        p " not in fvs_text:
        fvs_text = fvs_text.replace("p_rgh           0.4;", "p_rgh           0.4;\n        p               0.3;")
        fvs_text = fvs_text.replace("p_rgh           5e-4;", "p_rgh           5e-4;\n        p               1e-4;")
        modified = True
    # Add T solver if missing (needed by scalarTransport)
    if re.search(r'^\s+T\s*\{', fvs_text, re.M) is None:
        fvs_text = fvs_text.replace(
            "    epsilonFinal",
            "    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; maxIter 50; }\n"
            "    TFinal { $T; relTol 0; }\n\n    epsilonFinal",
        )
        modified = True
    if modified:
        fvs_path.write_text(fvs_text)
        logger.info("fvSolution: added p + T entries for simpleFoam")

    # Switch solver to simpleFoam in controlDict (everything else stays)
    cd_path = case_dir / "system" / "controlDict"
    cd_text = cd_path.read_text()
    cd_text = re.sub(r'^application\s+\w+;',
                     'application     simpleFoam;', cd_text, flags=re.M)
    cd_text = re.sub(r'^endTime\s+\d+;',
                     f'endTime         {sf_iter};', cd_text, flags=re.M)
    cd_text = re.sub(r'^startFrom\s+\w+;',
                     'startFrom       startTime;', cd_text, flags=re.M)
    cd_text = re.sub(r'^writeInterval\s+\d+;',
                     f'writeInterval   {sf_iter};', cd_text, flags=re.M)
    # Add scalarTransport to advect T as passive scalar in simpleFoam
    if "scalarTransport" not in cd_text:
        # Insert before the closing of the last functions block, or append
        cd_text += """
functions
{
    passiveT
    {
        type            scalarTransport;
        libs            (solverFunctionObjects);
        field           T;
        nCorr           0;
        resetOnStartUp  false;
    }
}
"""
    cd_path.write_text(cd_text)
    logger.info("controlDict: application → simpleFoam, endTime=%d, +scalarTransport(T)", sf_iter)

    t0 = time.time()
    result = run_docker(case_dir, "simpleFoam", nprocs=nprocs)
    wall = time.time() - t0

    if result.returncode != 0:
        logger.error("SF failed (%.0fs):\n%s", wall, result.stderr[-500:])
        raise RuntimeError("simpleFoam failed")

    logger.info("SF converged in %.0f s", wall)

    # Find latest time
    time_dirs = [d for d in case_dir.iterdir()
                 if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0]
    if not time_dirs:
        logger.info("Reconstructing SF fields...")
        recon_script = Path(__file__).parent / "reconstruct_fields.py"
        r = subprocess.run(
            [sys.executable, str(recon_script),
             "--case-dir", str(case_dir), "--time", "latest",
             "--write-foam", "--fields", "U", "k", "epsilon", "T", "p_rgh"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("reconstruct_fields.py: %s", r.stderr[-200:])
        time_dirs = [d for d in case_dir.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0]

    latest = max(time_dirs, key=lambda d: float(d.name))
    logger.info("SF latest time: %s", latest.name)


# ---------------------------------------------------------------------------
# Step 3: Switch to BBSF and continue
# ---------------------------------------------------------------------------
def step_switch_to_bbsf(case_dir: Path, bbsf_iter: int) -> None:
    """Switch solver from simpleFoam to BBSF and set startFrom latestTime.

    No field conversion needed — SF ran on the exact same case with the
    same BCs. The converged SF solution is directly usable by BBSF.
    Only changes: application name, endTime, startFrom.
    """
    logger.info("=== Step 3: Switch to BBSF ===")

    # Compute T_ref from SF-converged T field for transportProperties
    t_path = None
    time_dirs = sorted(
        [d for d in case_dir.iterdir()
         if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0],
        key=lambda d: float(d.name),
    )
    if time_dirs:
        t_path = time_dirs[-1] / "T"

    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    with open(inflow_json) as f:
        inflow = json.load(f)

    T_ref = float(inflow.get("T_ref", 288.15))
    if t_path and t_path.exists():
        t_text = t_path.read_text()
        m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', t_text)
        if m:
            start = m.end()
            end = t_text.index(')', start)
            T_vals = np.array([float(x) for x in t_text[start:end].split()])
            T_ref = float(T_vals.mean())
            logger.info("T_ref from SF-converged T: mean=%.2f K (min=%.1f, max=%.1f)",
                        T_ref, T_vals.min(), T_vals.max())

    _update_transport_properties(case_dir, T_ref)

    # Remove 0/p (simpleFoam artifact — BBSF uses p_rgh)
    p_path = case_dir / "0" / "p"
    if p_path.exists():
        p_path.unlink()
        logger.info("Removed 0/p (BBSF uses p_rgh)")

    # Copy BBSF-only fields (T, alphat, p_rgh) to SF latest time dir.
    # simpleFoam doesn't solve T/alphat/p_rgh so they're missing from
    # the converged time directory. BBSF needs them at startFrom latestTime.
    if time_dirs:
        latest_dir = time_dirs[-1]
        for field_name in ["T", "alphat", "p_rgh"]:
            src = case_dir / "0" / field_name
            dst = latest_dir / field_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.info("Copied 0/%s → %s/%s", field_name, latest_dir.name, field_name)
        # Also rename p → p_rgh in latest time dir if SF wrote p there
        p_in_latest = latest_dir / "p"
        p_rgh_in_latest = latest_dir / "p_rgh"
        if p_in_latest.exists() and not p_rgh_in_latest.exists():
            p_text = p_in_latest.read_text()
            p_text = p_text.replace("object p;", "object p_rgh;")
            p_text = p_text.replace("object  p;", "object  p_rgh;")
            p_rgh_in_latest.write_text(p_text)
            p_in_latest.unlink()
            logger.info("Renamed %s/p → p_rgh", latest_dir.name)

    # Clean SF postProcessing (so BBSF starts fresh monitoring)
    pp = case_dir / "postProcessing"
    if pp.exists():
        shutil.rmtree(pp)
        logger.info("Cleaned SF postProcessing")

    # Switch controlDict: BBSF, startFrom latestTime
    cd_path = case_dir / "system" / "controlDict"
    cd_text = cd_path.read_text()
    cd_text = re.sub(r'^application\s+\w+;',
                     'application     buoyantBoussinesqSimpleFoam;',
                     cd_text, flags=re.M)
    cd_text = re.sub(r'^startFrom\s+\w+;',
                     'startFrom       latestTime;', cd_text, flags=re.M)
    cd_text = re.sub(r'^endTime\s+\d+;',
                     f'endTime         {bbsf_iter};', cd_text, flags=re.M)
    cd_path.write_text(cd_text)
    logger.info("controlDict: application → BBSF, startFrom latestTime, endTime=%d", bbsf_iter)


# ---------------------------------------------------------------------------
# Step 4: Run BBSF
# ---------------------------------------------------------------------------
def step_bbsf(case_dir: Path, bbsf_iter: int, nprocs: int) -> None:
    """Run BBSF continuing from SF-converged solution.

    controlDict is already configured by step_switch_to_bbsf:
    application=BBSF, startFrom=latestTime, endTime set.
    """
    logger.info("=== Step 4: BBSF (endTime=%d, %d cores) ===", bbsf_iter, nprocs)

    # Ensure function objects for monitoring
    cd_path = case_dir / "system" / "controlDict"
    cd_text = cd_path.read_text()
    if "volAverages" not in cd_text:
        _add_function_objects(cd_path)

    t0 = time.time()
    result = run_docker(case_dir, "buoyantBoussinesqSimpleFoam", nprocs=nprocs,
                        timeout=14400)
    wall = time.time() - t0

    if result.returncode != 0:
        logger.warning("BBSF exited with error (%.0fs) — checking postProcessing", wall)

    logger.info("BBSF wall time: %.0f s", wall)


# ---------------------------------------------------------------------------
# Step 5: Convergence report
# ---------------------------------------------------------------------------
def step_report(case_dir: Path, sf_iter: int) -> None:
    """Generate convergence report and plots."""
    logger.info("=== Step 5: Convergence report ===")

    # After cleaning postProcessing between SF and BBSF, only BBSF data remains
    va_dir = case_dir / "postProcessing" / "volAverages" / "0"
    va_path = va_dir / "volFieldValue.dat"
    if not va_path.exists():
        logger.error("No volAverages found — cannot generate report")
        return
    logger.info("Reading volAverages from %s", va_path.name)

    lines = [l for l in va_path.read_text().splitlines() if l.strip() and not l.startswith('#')]

    iters, ux, uy, uz, k_arr, eps_arr, T_arr = [], [], [], [], [], [], []
    for line in lines:
        parts = line.split('\t')
        iters.append(int(parts[0].strip()))
        u_vals = [float(x) for x in parts[1].strip().strip('()').split()]
        ux.append(u_vals[0]); uy.append(u_vals[1]); uz.append(u_vals[2])
        k_arr.append(float(parts[2].strip()))
        eps_arr.append(float(parts[3].strip()))
        T_arr.append(float(parts[4].strip()) if len(parts) > 4 else 0.0)

    iters = np.array(iters)
    speed = np.sqrt(np.array(ux)**2 + np.array(uy)**2 + np.array(uz)**2)
    uz = np.array(uz); k_arr = np.array(k_arr); eps_arr = np.array(eps_arr); T_arr = np.array(T_arr)

    n = len(iters)
    k_final = k_arr[-1]
    dk_200 = abs(k_arr[-1] - k_arr[max(0, n-200)]) / max(k_final, 1e-10) * 100 if n > 200 else 999
    dk_500 = abs(k_arr[-1] - k_arr[max(0, n-500)]) / max(k_final, 1e-10) * 100 if n > 500 else 999
    dk_1000 = abs(k_arr[-1] - k_arr[max(0, n-1000)]) / max(k_final, 1e-10) * 100 if n > 1000 else 999

    converged = dk_500 < 5 and k_final < 5

    # Exponential fit for k plateau
    try:
        from scipy.optimize import curve_fit
        def exp_decay(x, k_inf, A, tau):
            return k_inf + A * np.exp(-x / tau)
        mask = iters > max(iters) * 0.3  # fit on last 70%
        popt, _ = curve_fit(exp_decay, iters[mask], k_arr[mask], p0=[3, 40, 2000], maxfev=5000)
        k_plateau = popt[0]
        iter_plateau = popt[2] * np.log(popt[1] / (0.02 * abs(popt[0]) + 1e-10))
        logger.info("k exponential fit: k_inf=%.2f, tau=%.0f iter", popt[0], popt[2])
    except Exception:
        k_plateau = k_final
        iter_plateau = iters[-1]

    # Write report
    report_path = case_dir / "convergence_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  CONVERGENCE REPORT — SF precursor → BBSF\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Total iterations  : {sf_iter} (SF) + {iters[-1]} (BBSF) = {sf_iter + iters[-1]}\n")
        f.write(f"  U mean final      : ({ux[-1]:.1f}, {uy[-1]:.1f}, {uz[-1]:.1f}) m/s  |U|={speed[-1]:.1f}\n")
        f.write(f"  Uz mean final     : {uz[-1]:.2f} m/s\n")
        f.write(f"  k mean final      : {k_final:.2f} m²/s²\n")
        f.write(f"  ε mean final      : {eps_arr[-1]:.4e} m²/s³\n")
        f.write(f"  T mean final      : {T_arr[-1]:.2f} K\n\n")
        f.write(f"  Δk (200 iter)     : {dk_200:.1f}%\n")
        f.write(f"  Δk (500 iter)     : {dk_500:.1f}%\n")
        f.write(f"  Δk (1000 iter)    : {dk_1000:.1f}%\n\n")
        f.write(f"  k plateau (fit)   : {k_plateau:.2f} m²/s² @ iter ~{iter_plateau:.0f}\n\n")
        f.write(f"  CONVERGED         : {'YES' if converged else 'NO'}\n")
        if not converged:
            if dk_500 >= 5:
                extra = int(iters[-1] * dk_500 / 5)
                f.write(f"  → Continuer ~{extra} iter supplémentaires\n")
            if k_final >= 5:
                f.write(f"  → k encore élevé ({k_final:.1f}), attendre plateau\n")

    logger.info("Report: %s", report_path)
    print(report_path.read_text())

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f'BBSF convergence — SF({sf_iter}) → BBSF({iters[-1]})\nk final={k_final:.2f}, Δk500={dk_500:.1f}%', fontsize=12)

    ax = axes[0, 0]
    ax.plot(iters, speed, 'b-', lw=1.5, label='vol mean |U|')
    ax.set_ylim(0, speed[-1] * 2.5)
    ax.set_ylabel('|U| [m/s]'); ax.set_title('Wind speed'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, uz, 'r-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylim(-5, 10)
    ax.set_ylabel('Uz [m/s]'); ax.set_title('Vertical velocity (vol mean)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, k_arr, 'g-', lw=1.5, label='vol mean k')
    ax.set_ylim(0, max(k_final * 3, 15))
    ax.set_ylabel('k [m²/s²]'); ax.set_title('TKE'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, eps_arr, 'm-', lw=1.5)
    ax.set_ylim(0, eps_arr[-1] * 3)
    ax.set_ylabel('ε [m²/s³]'); ax.set_title('Dissipation (vol mean)'); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(iters, T_arr, 'orange', lw=1.5, label='vol mean T')
    ax.set_ylim(T_arr[-1] - 2, T_arr[-1] + 2)
    ax.set_ylabel('T [K]'); ax.set_xlabel('Iteration'); ax.set_title('Temperature'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # k convergence zoom (last 50%)
    ax = axes[2, 1]
    half = max(1, n // 2)
    ax.plot(iters[half:], k_arr[half:], 'g-', lw=2, label='k (last 50%)')
    ax.set_ylabel('k [m²/s²]'); ax.set_xlabel('Iteration'); ax.set_title('k convergence (zoom)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = case_dir / "convergence_plot.png"
    plt.savefig(fig_path, dpi=150)
    logger.info("Plot: %s", fig_path)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _read_mesh_patches(case_dir: Path) -> list[str]:
    """Read patch names from constant/polyMesh/boundary."""
    bnd = (case_dir / "constant" / "polyMesh" / "boundary").read_text()
    return re.findall(r'^(\w+)\n\{', bnd, re.M)


def _write_sf_pressure(case_dir: Path):
    """Write 0/p for simpleFoam with SAME BCs as BBSF p_rgh.

    Using identical uniformMixed+pos(phi) on laterals ensures the
    converged SF pressure is in the same reference frame as p_rgh.
    This avoids a massive discontinuity when BBSF inherits the field.
    """
    patches = _read_mesh_patches(case_dir)
    is_octagonal = "lateral" in patches
    lateral_patches = [p for p in patches if p not in ("top", "terrain", "bottom")]

    bc_entries = []
    for p in lateral_patches:
        bc_entries.append(f"""    {p}
    {{
        type            uniformMixed;
        uniformValue    uniform 0;
        uniformGradient zero;
        uniformValueFraction
        {{
            type        expression;
            expression  "pos(phi)";
        }}
        value           uniform 0;
    }}""")
    if "top" in patches:
        bc_entries.append("    top { type zeroGradient; }")
    if "terrain" in patches:
        bc_entries.append("    terrain { type fixedFluxPressure; gradient uniform 0; value uniform 0; }")
    if "bottom" in patches:
        bc_entries.append("    bottom { type fixedFluxPressure; gradient uniform 0; value uniform 0; }")

    (case_dir / "0" / "p").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object p;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{chr(10).join(bc_entries)}
}}
""")
    # Remove p_rgh if exists
    (case_dir / "0" / "p_rgh").unlink(missing_ok=True)


def _write_sf_fvsolution(case_dir: Path):
    """Write fvSolution for simpleFoam precursor."""
    (case_dir / "system" / "fvSolution").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}

solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; maxIter 200; }
    pFinal { $p; relTol 0; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    UFinal { $U; relTol 0; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    kFinal { $k; relTol 0; }
    epsilon { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    epsilonFinal { $epsilon; relTol 0; }
    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    TFinal { $T; relTol 0; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell  0;
    pRefValue 0;
    residualControl { p 1e-4; U 1e-4; k 1e-4; epsilon 1e-4; T 1e-4; }
}

relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.7; epsilon 0.7; T 0.7; }
}
""")


def _write_sf_velocity(case_dir: Path):
    """Write 0/U for SF with simple inletOutlet BCs (no mappedFile)."""
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    with open(inflow_json) as f:
        inflow = json.load(f)

    ux = inflow["u_hub"] * inflow["flowDir_x"]
    uy = inflow["u_hub"] * inflow["flowDir_y"]
    patches = _read_mesh_patches(case_dir)
    laterals = [p for p in patches if p not in ("top", "terrain", "bottom")]

    bc = []
    for p in laterals:
        bc.append(f"""    {p}
    {{
        type            inletOutlet;
        inletValue      uniform ({ux:.6f} {uy:.6f} 0);
        value           uniform ({ux:.6f} {uy:.6f} 0);
    }}""")
    if "top" in patches:
        bc.append("    top { type slip; }")
    if "terrain" in patches:
        bc.append("    terrain { type noSlip; }")
    if "bottom" in patches:
        bc.append("    bottom { type noSlip; }")

    (case_dir / "0" / "U").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volVectorField; location "0"; object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({ux:.6f} {uy:.6f} 0);
boundaryField
{{
{chr(10).join(bc)}
}}
""")
    logger.info("U init: uniform (%.2f, %.2f, 0) m/s (inletOutlet BCs)", ux, uy)


def _write_sf_k_epsilon(case_dir: Path):
    """Write 0/k and 0/epsilon for SF with simple BCs."""
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    with open(inflow_json) as f:
        inflow = json.load(f)

    u_star = inflow["u_star"]
    k_init = u_star**2 / CMU**0.5
    y_wall = 10.0
    eps_init = CMU**0.75 * k_init**1.5 / (KAPPA * y_wall)

    patches = _read_mesh_patches(case_dir)
    laterals = [p for p in patches if p not in ("top", "terrain", "bottom")]

    dims = {"k": "[0 2 -2 0 0 0 0]", "epsilon": "[0 2 -3 0 0 0 0]"}
    for field, val, wall_type in [
        ("k", k_init, "kqRWallFunction"),
        ("epsilon", eps_init, "epsilonWallFunction"),
    ]:
        bc = []
        for p in laterals:
            bc.append(f"""    {p}
    {{
        type            inletOutlet;
        inletValue      uniform {val:.6e};
        value           uniform {val:.6e};
    }}""")
        if "top" in patches:
            bc.append(f"    top {{ type zeroGradient; }}")
        if "terrain" in patches:
            bc.append(f"    terrain {{ type {wall_type}; value uniform {val:.6e}; }}")
        if "bottom" in patches:
            bc.append(f"    bottom {{ type {wall_type}; value uniform {val:.6e}; }}")

        (case_dir / "0" / field).write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object {field};
}}
dimensions {dims[field]};
internalField uniform {val:.6e};
boundaryField
{{
{chr(10).join(bc)}
}}
""")

    logger.info("k=%.6f, eps=%.6e (y_wall=%.0fm, inletOutlet BCs)", k_init, eps_init, y_wall)


def _write_sf_nut(case_dir: Path):
    """Write 0/nut for SF with correct patches."""
    patches = _read_mesh_patches(case_dir)
    laterals = [p for p in patches if p not in ("top", "terrain", "bottom")]

    bc = []
    for p in laterals:
        bc.append(f"    {p} {{ type calculated; value uniform 0; }}")
    if "top" in patches:
        bc.append("    top { type calculated; value uniform 0; }")
    if "terrain" in patches:
        bc.append("    terrain { type nutkWallFunction; value uniform 0; }")
    if "bottom" in patches:
        bc.append("    bottom { type nutkWallFunction; value uniform 0; }")

    (case_dir / "0" / "nut").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object nut;
}}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{chr(10).join(bc)}
}}
""")


def _set_uniform_U(case_dir: Path):
    """Set U internalField to uniform ERA5 hub height."""
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    with open(inflow_json) as f:
        inflow = json.load(f)

    ux = inflow["u_hub"] * inflow["flowDir_x"]
    uy = inflow["u_hub"] * inflow["flowDir_y"]

    u_path = case_dir / "0" / "U"
    u_text = u_path.read_text()
    # Replace internalField (uniform or nonuniform)
    u_text = re.sub(
        r'internalField\s+(uniform\s+\([^)]+\)|nonuniform\s+List<vector>.*?\);)',
        f'internalField   uniform ( {ux:.6f} {uy:.6f} 0 );',
        u_text, flags=re.S,
    )
    u_path.write_text(u_text)
    logger.info("U init: uniform (%.2f, %.2f, 0) m/s", ux, uy)


def _set_uniform_k_epsilon(case_dir: Path):
    """Set k, epsilon to uniform values consistent with wall function."""
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    with open(inflow_json) as f:
        inflow = json.load(f)

    u_star = inflow["u_star"]
    k_init = u_star**2 / CMU**0.5
    y_wall = 10.0
    eps_init = CMU**0.75 * k_init**1.5 / (KAPPA * y_wall)

    for field, val in [("k", k_init), ("epsilon", eps_init)]:
        fpath = case_dir / "0" / field
        if not fpath.exists():
            continue
        text = fpath.read_text()
        text = re.sub(
            r'internalField\s+(uniform\s+[\d.e+-]+|nonuniform\s+List<scalar>.*?\);)',
            f'internalField   uniform {val:.6e};',
            text, flags=re.S,
        )
        fpath.write_text(text)

    logger.info("k init: %.6f m²/s², eps init: %.6e m²/s³ (y_wall=%.0fm)", k_init, eps_init, y_wall)


def _write_p_rgh(case_dir: Path, p_rgh: np.ndarray):
    """Write 0/p_rgh with BBSF BCs (dual of U)."""
    # Detect patch style
    u_text = (case_dir / "0" / "U").read_text()
    is_octagonal = "lateral" in u_text

    vals = "\n".join(f"{v:.6e}" for v in p_rgh)

    if is_octagonal:
        lateral_bc = """    lateral
    {
        type            uniformMixed;
        uniformValue    uniform 0;
        uniformGradient zero;
        uniformValueFraction
        {
            type        expression;
            expression  "pos(phi)";
        }
        value           uniform 0;
    }"""
    else:
        lateral_bc = "\n".join([f"""    {face}
    {{
        type            uniformMixed;
        uniformValue    uniform 0;
        uniformGradient zero;
        uniformValueFraction
        {{
            type        expression;
            expression  "pos(phi)";
        }}
        value           uniform 0;
    }}""" for face in ["west", "east", "south", "north"]])

    patches = _read_mesh_patches(case_dir)
    extra_bcs = []
    if "top" in patches:
        extra_bcs.append("    top { type zeroGradient; }")
    if "terrain" in patches:
        extra_bcs.append("    terrain { type fixedFluxPressure; gradient uniform 0; value uniform 0; }")
    if "bottom" in patches:
        extra_bcs.append("    bottom { type fixedFluxPressure; gradient uniform 0; value uniform 0; }")

    (case_dir / "0" / "p_rgh").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object p_rgh;
}}
dimensions [0 2 -2 0 0 0 0];
internalField nonuniform List<scalar>
{len(p_rgh)}
(
{vals}
)
;
boundaryField
{{
{lateral_bc}
{chr(10).join(extra_bcs)}
}}
""")


def _write_T(case_dir: Path, T_ref: float, inflow: dict):
    """Write 0/T with uniform T_ref and mappedFile BCs."""
    u_text = (case_dir / "0" / "U").read_text()
    is_octagonal = "lateral" in u_text

    if is_octagonal:
        lateral_bc = f"""    lateral
    {{
        type            inletOutlet;
        inletValue      uniform {T_ref:.2f};
        value           uniform {T_ref:.2f};
    }}"""
    else:
        lateral_bc = "\n".join([f"""    {face}
    {{
        type            inletOutlet;
        inletValue      uniform {T_ref:.2f};
        value           uniform {T_ref:.2f};
    }}""" for face in ["west", "east", "south", "north"]])

    patches = _read_mesh_patches(case_dir)
    extra_bcs = []
    if "top" in patches:
        extra_bcs.append("    top { type zeroGradient; }")
    if "terrain" in patches:
        extra_bcs.append("    terrain { type zeroGradient; }")
    if "bottom" in patches:
        extra_bcs.append("    bottom { type zeroGradient; }")

    (case_dir / "0" / "T").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object T;
}}
dimensions [0 0 0 1 0 0 0];
internalField uniform {T_ref:.2f};
boundaryField
{{
{lateral_bc}
{chr(10).join(extra_bcs)}
}}
""")


def _update_transport_properties(case_dir: Path, T_ref: float):
    """Update TRef and beta in transportProperties."""
    tp_path = case_dir / "constant" / "transportProperties"
    if not tp_path.exists():
        return
    tp = tp_path.read_text()
    beta = 1.0 / T_ref
    tp = re.sub(r'(TRef\s+\[.*?\]\s+)[\d.]+', lambda m: m.group(1) + f"{T_ref:.2f}", tp)
    tp = re.sub(r'(beta\s+\[.*?\]\s+)[\d.e+-]+', lambda m: m.group(1) + f"{beta:.6e}", tp)
    tp_path.write_text(tp)
    logger.info("transportProperties: TRef=%.2f, beta=%.6e", T_ref, beta)


def _fix_alphat(case_dir: Path):
    """Write 0/alphat for BBSF (incompressible dimensions + Jayatilleke wall function)."""
    patches = _read_mesh_patches(case_dir)
    laterals = [p for p in patches if p not in ("top", "terrain", "bottom")]

    bc = []
    for p in laterals:
        bc.append(f"    {p} {{ type calculated; value uniform 0; }}")
    if "top" in patches:
        bc.append("    top { type calculated; value uniform 0; }")
    if "terrain" in patches:
        bc.append("    terrain { type alphatJayatillekeWallFunction; Prt 0.85; value uniform 0; }")
    if "bottom" in patches:
        bc.append("    bottom { type alphatJayatillekeWallFunction; Prt 0.85; value uniform 0; }")

    (case_dir / "0" / "alphat").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; location "0"; object alphat;
}}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{chr(10).join(bc)}
}}
""")


def _fix_fvschemes(case_dir: Path):
    """Add div(phi,T) to fvSchemes if missing."""
    fs_path = case_dir / "system" / "fvSchemes"
    text = fs_path.read_text()
    if "div(phi,T)" not in text:
        text = text.replace(
            "div(phi,epsilon)",
            "div(phi,epsilon)                bounded Gauss upwind;\n"
            "    div(phi,T)                      bounded Gauss linearUpwind grad(T)"
        ).replace(
            "div(phi,epsilon)                bounded Gauss upwind;\n"
            "    div(phi,epsilon)",
            "div(phi,epsilon)"
        )
        # Simpler: just append before closing brace
        if "div(phi,T)" not in text:
            text = text.replace(
                "div(phi,epsilon)                bounded Gauss upwind;",
                "div(phi,epsilon)                bounded Gauss upwind;\n"
                "    div(phi,T)                      bounded Gauss linearUpwind grad(T);"
            )
        fs_path.write_text(text)


def _write_bbsf_fvsolution(case_dir: Path):
    """Write fvSolution for BBSF."""
    (case_dir / "system" / "fvSolution").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}

solvers
{
    p_rgh { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.01; maxIter 200; }
    p_rghFinal { $p_rgh; relTol 0; }
    U { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    UFinal { $U; relTol 0; }
    k { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    kFinal { $k; relTol 0; }
    epsilon { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    epsilonFinal { $epsilon; relTol 0; }
    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-6; relTol 0.1; }
    TFinal { $T; relTol 0; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    kMin        1e-10;
    epsilonMin  1e-8;
    residualControl { p_rgh 5e-4; U 1e-4; k 1e-4; epsilon 1e-4; T 1e-4; }
}

relaxationFactors
{
    fields { p_rgh 0.7; }
    equations { U 0.5; T 0.5; "(k|epsilon)" 0.1; }
}
""")


def _write_fvoptions(case_dir: Path):
    """Write fvOptions with ambient turb sources (no Lmax)."""
    (case_dir / "constant" / "fvOptions").write_text("""FoamFile
{
    format ascii; class dictionary; object fvOptions;
}

ambientTurbSource
{
    type            atmAmbientTurbSource;
    atmAmbientTurbSourceCoeffs
    {
        selectionMode   all;
        kAmb            0.001;
        epsilonAmb      7.208e-08;
    }
}
// Lmax disabled: causes k oscillations → negative k → divergence at iter ~300-400
""")


def _add_function_objects(cd_path: Path):
    """Append monitoring function objects to controlDict."""
    text = cd_path.read_text()
    if "functions" in text:
        return
    text += """

functions
{
    fieldMinMax
    {
        type            fieldMinMax;
        libs            (fieldFunctionObjects);
        writeControl    timeStep;
        writeInterval   1;
        fields          ( U k p_rgh T );
    }
    volAverages
    {
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    timeStep;
        writeInterval   1;
        operation       volAverage;
        regionType      all;
        writeFields     false;
        fields          ( U k epsilon T );
    }
}
"""
    cd_path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="SF precursor → BBSF pipeline")
    parser.add_argument("--config", required=True, help="Study YAML config")
    parser.add_argument("--case", required=True, help="Case ID (e.g. A, smoke_local)")
    parser.add_argument("--sf-iter", type=int, default=3000, help="SF iterations")
    parser.add_argument("--bbsf-iter", type=int, default=10000, help="BBSF iterations")
    parser.add_argument("--nprocs", type=int, default=4, help="Docker parallel cores")
    args = parser.parse_args()

    project_root = Path(__file__).parents[2]
    config_path = project_root / args.config

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cases_dir = project_root / "data" / "cases" / cfg["study"]["name"]
    case_dir = cases_dir / f"case_{args.case}"

    # DELETE AND RECREATE FROM SCRATCH
    if case_dir.exists():
        logger.info("Deleting existing case: %s", case_dir)
        shutil.rmtree(case_dir)

    # Step 1: Generate BBSF case (full setup with all BCs)
    case_dir = step_generate(config_path, args.case, project_root)

    # Step 2: Run simpleFoam precursor (same case, just switch solver)
    step_sf_precursor(case_dir, args.sf_iter, args.nprocs)

    # Step 3: Switch to BBSF (startFrom latestTime, update T_ref)
    step_switch_to_bbsf(case_dir, args.sf_iter + args.bbsf_iter)

    # Step 4: Run BBSF (continues from SF-converged solution)
    step_bbsf(case_dir, args.sf_iter + args.bbsf_iter, args.nprocs)

    # Step 5: Report
    step_report(case_dir, args.sf_iter)


if __name__ == "__main__":
    main()
