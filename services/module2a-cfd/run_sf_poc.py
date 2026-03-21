"""
run_sf_poc.py — Orchestrate SF PoC campaign: 25 ERA5 timestamps × 1 terrain.

Generates one mesh (cfMesh, cylindrical domain), then for each timestamp:
  1. Prepare inflow from ERA5 at that timestamp
  2. Generate case directory (same mesh, different BCs)
  3. Initialize fields (init_from_era5)
  4. potentialFoam (div-free projection)
  5. simpleFoam (200-500 iter)

Usage
-----
    cd services/module2a-cfd
    python run_sf_poc.py \
        --config     ../../configs/poc_sf_cylinder.yaml \
        --timestamps ../../data/campaign/sf_poc/timestamps.csv \
        --output     ../../data/cases/sf_poc/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_mesh import generate_mesh
from prepare_inflow import prepare_inflow

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("run_sf_poc")


# ---------------------------------------------------------------------------
# Docker helpers (reused from run_local_study.py)
# ---------------------------------------------------------------------------

def _detect_docker_config(cfg: dict) -> tuple[str, str | None, str, str | None]:
    """Detect best Docker images for meshing and solving."""
    import platform as plat
    is_arm = plat.machine() in ("arm64", "aarch64")

    mesh_image = "microfluidica/openfoam:latest"
    mesh_platform = "linux/amd64" if is_arm else None

    solver_image = cfg["docker"].get("image", "microfluidica/openfoam:latest")
    solver_platform = None

    if is_arm:
        try:
            result = subprocess.run(
                ["docker", "inspect", "kraken/openfoam-ai:latest"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                solver_image = "kraken/openfoam-ai:latest"
                log.info("Using native ARM64 image for solver",
                         extra={"image": solver_image})
        except Exception:
            solver_platform = "linux/amd64"

    return mesh_image, mesh_platform, solver_image, solver_platform


def run_docker(
    case_dir: Path,
    command: str,
    image: str = "microfluidica/openfoam:latest",
    timeout: int = 3600,
    platform: str | None = None,
) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command inside Docker with the case dir mounted."""
    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend([
        "-v", f"{case_dir.resolve()}:/case",
        "-w", "/case",
        image,
        "bash", "-c", f"cd /case && {command}",
    ])
    log.info("Docker: %s [%s]", command.split()[0], case_dir.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("Docker failed: %s", result.stderr[-300:] if result.stderr else "")
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_cases(
    cfg: dict,
    timestamps: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate one case directory per timestamp with per-case inflow."""
    site_cfg_path = ROOT / "configs" / "sites" / f"{cfg['study']['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    srtm_tif = ROOT / "data" / "raw" / f"srtm_{cfg['study']['site']}_30m.tif"
    era5_zarr = ROOT / "data" / "raw" / f"era5_{cfg['study']['site']}.zarr"
    study = cfg["study"]

    # Get the first (and only) case config as template
    case_template_id = list(cfg["cases"].keys())[0]
    case_template = cfg["cases"][case_template_id]

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]

    case_dirs = {}
    for _, row in timestamps.iterrows():
        ts = pd.Timestamp(row["datetime"])
        case_id = f"ts_{ts.strftime('%Y%m%d_%H%M')}"
        case_dir = output_dir / case_id

        if case_dir.exists() and (case_dir / "system" / "controlDict").exists():
            log.info("Case exists: %s", case_id)
            case_dirs[case_id] = case_dir
            continue

        log.info("Generating case: %s (%.1f m/s, %.0f°)",
                 case_id, row["speed_10m"], row["direction_deg"])

        # Prepare inflow for this timestamp
        inflow_json = output_dir / "inflow" / f"{case_id}.json"
        inflow_json.parent.mkdir(parents=True, exist_ok=True)

        if not inflow_json.exists():
            prepare_inflow(
                era5_zarr=era5_zarr,
                timestamp=ts.isoformat(),
                site_lat=site_lat,
                site_lon=site_lon,
                output_json=inflow_json,
            )

        # Generate full case (templates rendered with this inflow)
        generate_mesh(
            site_cfg=site_cfg,
            resolution_m=study.get("resolution_m", 1000),
            context_cells=study.get("context_cells", 1),
            output_dir=case_dir,
            srtm_tif=srtm_tif,
            inflow_json=inflow_json,
            domain_km=study.get("domain_km", 10),
            domain_type=study.get("domain_type", "cylinder"),
            solver_name=case_template["solver"],
            thermal=case_template.get("thermal", False),
            coriolis=case_template.get("coriolis", False),
            canopy_enabled=case_template.get("canopy", False),
            n_iter=study.get("n_iterations", 500),
            write_interval=study.get("write_interval", 100),
        )

        # Copy inflow JSON + init script into case dir
        shutil.copy2(inflow_json, case_dir / "inflow.json")
        init_script = Path(__file__).parent / "init_from_era5.py"
        shutil.copy2(init_script, case_dir / "init_from_era5.py")

        case_dirs[case_id] = case_dir

    return case_dirs


def mesh_first_and_copy(case_dirs: dict, cfg: dict) -> None:
    """Run cartesianMesh on the first case, copy polyMesh to the rest."""
    mesh_image, mesh_platform, _, _ = _detect_docker_config(cfg)
    first_id = list(case_dirs.keys())[0]
    first_dir = case_dirs[first_id]
    poly_mesh = first_dir / "constant" / "polyMesh"

    if poly_mesh.exists() and (poly_mesh / "points").exists():
        log.info("Mesh already exists, skipping cartesianMesh")
    else:
        log.info("Running cartesianMesh on %s...", first_id)
        result = run_docker(
            first_dir, "cartesianMesh",
            image=mesh_image, platform=mesh_platform,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cartesianMesh failed: {result.stderr[-500:]}")

        # checkMesh
        result = run_docker(
            first_dir, "checkMesh -latestTime",
            image=mesh_image, platform=mesh_platform,
        )
        log.info("checkMesh:\n%s", result.stdout[-500:] if result.stdout else "")

    # Copy polyMesh to other cases
    for case_id, case_dir in case_dirs.items():
        if case_id == first_id:
            continue
        dst = case_dir / "constant" / "polyMesh"
        if dst.exists() and (dst / "points").exists():
            continue
        log.info("Copying polyMesh → %s", case_id)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(poly_mesh, dst)


def init_fields(case_dirs: dict, cfg: dict) -> None:
    """Initialize fields from inflow profile for each case."""
    _, _, solver_image, solver_platform = _detect_docker_config(cfg)

    for case_id, case_dir in case_dirs.items():
        u_file = case_dir / "0" / "U"
        if u_file.exists() and "nonuniform" in u_file.read_text()[:3000]:
            log.info("Fields already initialized: %s", case_id)
            continue

        log.info("Initializing fields: %s", case_id)

        # writeCellCentres
        run_docker(
            case_dir, "postProcess -func writeCellCentres -time 0",
            image=solver_image, platform=solver_platform,
        )

        # init_from_era5.py (runs locally with numpy/scipy)
        result = subprocess.run(
            [sys.executable, "init_from_era5.py",
             "--case-dir", ".", "--inflow", "inflow.json"],
            cwd=case_dir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("init_from_era5 failed for %s: %s", case_id, result.stderr[-300:])
            raise RuntimeError(f"init_from_era5 failed for {case_id}")


def solve_all(case_dirs: dict, cfg: dict) -> pd.DataFrame:
    """Run potentialFoam + simpleFoam for each case. Return convergence report."""
    _, _, solver_image, solver_platform = _detect_docker_config(cfg)
    nprocs = cfg["docker"].get("nprocs", 1)
    records = []

    for case_id, case_dir in case_dirs.items():
        # Skip already-solved cases
        time_dirs = [
            d for d in case_dir.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit()
            and float(d.name) > 0
        ]
        if time_dirs:
            log.info("Already solved: %s", case_id)
            records.append({
                "case_id": case_id,
                "status": "CACHED",
                "wall_time_s": 0,
            })
            continue

        log.info("Solving: %s", case_id)
        t0 = time.time()

        # potentialFoam
        pot = run_docker(
            case_dir,
            "potentialFoam -writePhi > log.potentialFoam 2>&1",
            image=solver_image, platform=solver_platform, timeout=300,
        )
        if pot.returncode != 0:
            log.error("potentialFoam FAILED: %s", case_id)
            records.append({
                "case_id": case_id,
                "status": "FAILED_POTENTIAL",
                "wall_time_s": time.time() - t0,
            })
            continue

        # simpleFoam
        if nprocs > 1:
            command = (
                f"foamDictionary system/decomposeParDict "
                f"-entry numberOfSubdomains -set {nprocs} && "
                f"decomposePar && "
                f"mpirun --allow-run-as-root -np {nprocs} simpleFoam -parallel && "
                f"reconstructPar -latestTime"
            )
        else:
            command = "simpleFoam > log.simpleFoam 2>&1"

        result = run_docker(
            case_dir, command,
            image=solver_image, platform=solver_platform, timeout=3600,
        )
        wall_time = time.time() - t0

        if result.returncode != 0:
            status = "FAILED"
            # Save log for debugging
            (case_dir / "log.simpleFoam").write_text(
                result.stdout + "\n" + result.stderr
            )
        else:
            status = "CONVERGED"
            if result.stdout:
                (case_dir / "log.simpleFoam").write_text(result.stdout)

        log.info("%s: %s (%.0f s)", case_id, status, wall_time)
        records.append({
            "case_id": case_id,
            "status": status,
            "wall_time_s": wall_time,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run SF PoC campaign")
    parser.add_argument("--config", required=True, help="YAML config (poc_sf_cylinder.yaml)")
    parser.add_argument("--timestamps", required=True, help="timestamps.csv from sampler")
    parser.add_argument("--output", required=True, help="Output directory for cases")
    parser.add_argument("--mesh-only", action="store_true", help="Stop after meshing")
    parser.add_argument("--skip-mesh", action="store_true", help="Skip mesh generation")
    parser.add_argument("--skip-solve", action="store_true", help="Skip solver")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamps = pd.read_csv(args.timestamps)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("SF PoC campaign: %d timestamps, output=%s", len(timestamps), output_dir)

    # Step 1: Generate case directories
    case_dirs = generate_cases(cfg, timestamps, output_dir)
    log.info("Generated %d cases", len(case_dirs))

    # Step 2: Mesh (once, then copy)
    if not args.skip_mesh:
        mesh_first_and_copy(case_dirs, cfg)

    if args.mesh_only:
        log.info("--mesh-only: stopping after mesh")
        return

    # Step 3: Initialize fields
    init_fields(case_dirs, cfg)

    # Step 4: Solve
    if not args.skip_solve:
        report = solve_all(case_dirs, cfg)
        report_path = output_dir / "convergence_report.csv"
        report.to_csv(report_path, index=False)
        log.info("Convergence report saved: %s", report_path)

        # Summary
        n_ok = (report["status"] == "CONVERGED").sum()
        n_cached = (report["status"] == "CACHED").sum()
        n_fail = len(report) - n_ok - n_cached
        log.info("Results: %d converged, %d cached, %d failed / %d total",
                 n_ok, n_cached, n_fail, len(report))


if __name__ == "__main__":
    main()
