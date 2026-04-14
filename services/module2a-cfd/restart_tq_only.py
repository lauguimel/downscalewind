"""
restart_tq_only.py — Restart CFD cases with corrected T/q inflow, keeping U/k/epsilon.

Use case: after fixing the T/q inflow bug in prepare_inflow.py (2026-04-14),
re-solve existing CFD cases to recover correct T and q, without redoing the
expensive u/v convergence that is already good.

Strategy:
  1. For each solved case (has time=<N_ITER>/U):
     - Regenerate inflow.json with the fixed prepare_inflow.py
     - Regenerate boundaryData for T, q (NOT U)
     - Copy U/k/epsilon/nut from <N_ITER>/ to 0/ as initial condition
     - Rewrite 0/T and 0/q from the new inflow
     - Re-run simpleFoam for a short number of iterations (~100) to converge T/q
     - Export updated Zarr

Usage (on Aqua):
    python services/module2a-cfd/restart_tq_only.py \\
        --cases-dir data/campaign/icos_fwi_v1/cases/FR-Pue \\
        --era5-zarr data/raw/era5_fr-pue.zarr \\
        --n-cores 12 --n-parallel 2 --n-iter-restart 100 \\
        --runtime apptainer \\
        --of-image ~/dsw/containers/openfoam_v2512.sif

Much faster than full rerun: ~15s per case instead of 45s.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("restart_tq")

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
CONTAINER_RUNTIME = os.environ.get("CONTAINER_RUNTIME", "docker")


def _container_cmd(image, mount_dir, cmd, workdir="/home/ofuser/run"):
    d = str(mount_dir.resolve())
    if CONTAINER_RUNTIME == "apptainer":
        return ["apptainer", "exec", "--cleanenv",
                "--bind", f"{d}:{workdir}", "--pwd", workdir, image] + cmd
    return ["docker", "run", "--rm", "-v", f"{d}:{workdir}", "-w", workdir, image] + cmd


OF_ESI_BASHRC = "source /usr/lib/openfoam/openfoam2512/etc/bashrc 2>/dev/null || true"


def restart_case(case_dir, era5_zarr, site_lat, site_lon,
                 n_iter_prev, n_iter_restart, n_cores, of_image):
    """Restart one case with corrected T/q.

    case_dir must contain constant/polyMesh + <n_iter_prev>/ with U,k,epsilon,nut.
    """
    log.info("  Restart %s", case_dir.name)

    # Check preconditions
    prev_time = case_dir / str(n_iter_prev)
    if not (prev_time / "U").exists():
        log.error("  %s: no %s/U found — skipping", case_dir.name, n_iter_prev)
        return False

    # Extract timestamp from case_id (e.g. case_ts000 → need external mapping)
    # The inflow.json already has case info; regenerate it
    # We need: --case <timestamp>, but we don't have it here.
    # Solution: read existing inflow.json metadata if present
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        log.error("  %s: no inflow.json — cannot regenerate", case_dir.name)
        return False

    # Read timestamp from existing inflow.json (we just need to regenerate with same ts)
    import json
    with open(inflow_json) as f:
        existing = json.load(f)
    ts = existing.get("timestamp")
    if not ts:
        log.error("  %s: inflow.json has no timestamp", case_dir.name)
        return False

    # Regenerate inflow.json with new prepare_inflow (includes surface anchoring)
    new_inflow = case_dir / "inflow_new.json"
    result = subprocess.run(
        [PYTHON, str(SCRIPTS_DIR / "prepare_inflow.py"),
         "--era5", str(era5_zarr),
         "--case", ts,
         "--lat", str(site_lat), "--lon", str(site_lon),
         "--output", str(new_inflow)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log.error("  prepare_inflow failed: %s", (result.stderr or "")[-200:])
        return False
    shutil.move(str(new_inflow), str(inflow_json))

    # Copy solved fields from <n_iter_prev>/ to 0/ as initial conditions for restart
    # (Keep U, k, epsilon, nut — overwrite T, q from new inflow via init_from_era5)
    zero_dir = case_dir / "0"
    for field in ("U", "k", "epsilon", "nut", "p"):
        src = prev_time / field
        if src.exists():
            shutil.copy(src, zero_dir / field)

    # Re-run init_from_era5 to rewrite 0/T, 0/q + boundaryData with new inflow
    result = subprocess.run(
        [PYTHON, str(SCRIPTS_DIR / "init_from_era5.py"),
         "--case-dir", str(case_dir),
         "--inflow", str(inflow_json),
         "--fields", "T", "q"],  # only rewrite T and q
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning("  init_from_era5 failed (may not support --fields flag): %s",
                    (result.stderr or "")[-200:])
        # Fall back: full init (will overwrite U too, but then we re-copy)
        result = subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / "init_from_era5.py"),
             "--case-dir", str(case_dir),
             "--inflow", str(inflow_json)],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("  init_from_era5 failed: %s", (result.stderr or "")[-200:])
            return False
        # Re-copy U/k/epsilon/nut on top
        for field in ("U", "k", "epsilon", "nut"):
            src = prev_time / field
            if src.exists():
                shutil.copy(src, zero_dir / field)

    # Clean old processor dirs + timestep outputs
    for d in case_dir.glob("processor*"):
        shutil.rmtree(d, ignore_errors=True)
    for t in case_dir.iterdir():
        if t.is_dir() and t.name.isdigit() and int(t.name) > 0:
            shutil.rmtree(t)

    # Set endTime = n_iter_restart in controlDict
    cd = case_dir / "system" / "controlDict"
    if cd.exists():
        txt = cd.read_text()
        import re
        txt = re.sub(r'endTime\s+\d+', f'endTime {n_iter_restart}', txt)
        cd.write_text(txt)

    # Set numberOfSubdomains
    dpd = case_dir / "system" / "decomposeParDict"
    if dpd.exists():
        import re
        txt = dpd.read_text()
        txt = re.sub(r'numberOfSubdomains\s+\d+', f'numberOfSubdomains  {n_cores}', txt)
        dpd.write_text(txt)

    # decomposePar → mpirun → reconstruct
    def _of(cmd, timeout=300):
        if cmd[0] != "bash":
            cmd = ["bash", "-c", f"{OF_ESI_BASHRC} && {' '.join(cmd)}"]
        else:
            cmd = list(cmd)
            if CONTAINER_RUNTIME == "apptainer" and len(cmd) >= 3 and cmd[1] == "-c":
                cmd[2] = f"{OF_ESI_BASHRC} && {cmd[2]}"
        return subprocess.run(_container_cmd(of_image, case_dir, cmd),
                              capture_output=True, text=True, timeout=timeout)

    # decomposePar
    r = _of(["decomposePar", "-force"], 120)
    if r.returncode != 0:
        log.error("  decomposePar failed")
        return False

    # Symlink boundaryData into processor dirs
    if (case_dir / "constant" / "boundaryData").exists():
        for pd in sorted(case_dir.glob("processor*")):
            bd_dst = pd / "constant" / "boundaryData"
            if not bd_dst.exists():
                try:
                    bd_dst.symlink_to((case_dir / "constant" / "boundaryData").resolve())
                except PermissionError:
                    pass

    # mpirun simpleFoam
    mpi_flags = ["--oversubscribe", "--allow-run-as-root"] if CONTAINER_RUNTIME == "apptainer" else []
    r = _of(["mpirun"] + mpi_flags + ["-np", str(n_cores), "simpleFoam", "-parallel"], 900)
    (case_dir / "log.simpleFoam_restart").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        log.error("  simpleFoam failed: %s", (r.stderr or "")[-200:])
        return False

    # Reconstruct
    r = subprocess.run(
        [PYTHON, str(SCRIPTS_DIR / "reconstruct_fields.py"),
         "--case-dir", str(case_dir), "--time", "latest",
         "--write-foam", "--fields", "U", "T", "q", "k", "epsilon", "nut", "p"],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("  reconstruct failed: %s", (r.stderr or "")[-200:])
        return False

    for pd in sorted(case_dir.glob("processor*")):
        shutil.rmtree(pd, ignore_errors=True)
    return True


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", type=Path, required=True,
                    help="Directory containing case_ts* subdirs")
    ap.add_argument("--era5-zarr", type=Path, required=True)
    ap.add_argument("--site-lat", type=float, required=True)
    ap.add_argument("--site-lon", type=float, required=True)
    ap.add_argument("--n-iter-prev", type=int, default=500,
                    help="Number of iterations from previous solve (time dir)")
    ap.add_argument("--n-iter-restart", type=int, default=100,
                    help="Additional iterations for T/q convergence")
    ap.add_argument("--n-cores", type=int, default=12)
    ap.add_argument("--of-image", required=True)
    ap.add_argument("--runtime", choices=["docker", "apptainer"], default="docker")
    args = ap.parse_args()

    # Apply runtime
    globals()["CONTAINER_RUNTIME"] = args.runtime

    cases = sorted([d for d in args.cases_dir.iterdir()
                    if d.is_dir() and d.name.startswith("case_")])
    log.info("Found %d cases in %s", len(cases), args.cases_dir)

    n_ok = 0
    for c in cases:
        if restart_case(c, args.era5_zarr, args.site_lat, args.site_lon,
                        args.n_iter_prev, args.n_iter_prev + args.n_iter_restart,
                        args.n_cores, args.of_image):
            n_ok += 1
    log.info("Done: %d/%d cases restarted successfully", n_ok, len(cases))


if __name__ == "__main__":
    main()
