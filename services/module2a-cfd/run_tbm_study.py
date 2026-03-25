"""
run_tbm_study.py — Run CFD study with terrainBlockMesher structured mesh.

Pipeline:
  1. Prepare inflow (ERA5)
  2. Generate case (templates with section_0..7 lateral patches)
  3. Generate terrain STL
  4. Run terrainBlockMesher (Docker OF2.4) → polyMesh
  5. checkMesh (Docker OF2512)
  6. Init fields from ERA5 (auto-detects section_* patches)
  7. simpleFoam (Docker OF2512)
  8. Auto-evaluate (profiles + metrics + DB)

Usage
-----
    cd services/module2a-cfd
    python run_tbm_study.py ../../configs/poc_tbm_smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from generate_mesh import generate_mesh, dem_to_stl
from generate_mesh_tbm import generate_mesh_tbm
from prepare_inflow import prepare_inflow

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("run_tbm_study")


def run_docker(
    case_dir: Path, command: str,
    image: str = "microfluidica/openfoam:latest",
    timeout: int = 3600, platform: str | None = None,
) -> subprocess.CompletedProcess:
    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend(["-v", f"{case_dir.resolve()}:/case", "-w", "/case",
                image, "bash", "-c", f"cd /case && {command}"])
    log.info("Docker: %s [%s]", command.split()[0], case_dir.name)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run CFD study with terrainBlockMesher")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--cases", nargs="*", help="Run only these cases")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    study = cfg["study"]
    tbm_cfg = cfg["terrainBlockMesher"]
    docker_cfg = cfg["docker"]

    site_cfg_path = ROOT / "configs" / "sites" / f"{study['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]

    cases_dir = ROOT / "data" / "cases" / study["name"]
    cases_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Prepare inflow ---
    log.info("=== Step 1: Prepare inflow ===")
    era5_zarr = ROOT / "data" / "raw" / f"era5_{study['site']}.zarr"
    inflow_json = cases_dir / "inflow.json"

    if not inflow_json.exists():
        prepare_inflow(
            era5_zarr=era5_zarr,
            timestamp=study["timestamp"],
            site_lat=site_lat,
            site_lon=site_lon,
            output_json=inflow_json,
        )
    log.info("Inflow ready: %s", inflow_json)

    # --- Step 2: Generate terrain STL ---
    log.info("=== Step 2: Generate terrain STL ===")
    srtm_tif = ROOT / "data" / "raw" / f"srtm_{study['site']}_30m.tif"
    stl_path = cases_dir / "terrain.stl"

    if not stl_path.exists():
        radius_m = tbm_cfg.get("cylinder", {}).get("radius", 7000)
        half_m = radius_m * 1.1  # 10% margin
        DEG_LAT = 1.0 / 111_000.0
        DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))
        bounds = (
            site_lon - half_m * DEG_LON,
            site_lat - half_m * DEG_LAT,
            site_lon + half_m * DEG_LON,
            site_lat + half_m * DEG_LAT,
        )
        # Use fine resolution for STL (terrain detail)
        dem_to_stl(
            srtm_tif=srtm_tif,
            out_stl=stl_path,
            bounds_lonlat=bounds,
            resolution_m=100,  # coarse for smoke test
            site_lat=site_lat,
            site_lon=site_lon,
        )
    log.info("Terrain STL: %s", stl_path)

    # --- Step 3: Generate cases ---
    cases_to_run = args.cases or list(cfg["cases"].keys())

    for case_id in cases_to_run:
        case_cfg = cfg["cases"][case_id]
        case_dir = cases_dir / f"case_{case_id}"
        log.info("=== Case: %s ===", case_id)

        # Generate OF case structure (templates with section_* patches)
        if not (case_dir / "system" / "controlDict").exists():
            log.info("Generating case templates...")
            # lateral_patches for TBM: section_0..7
            n_sections = tbm_cfg.get("cylinder", {}).get("n_sections", 8)
            lateral_patches = [f"section_{i}" for i in range(n_sections)]

            generate_mesh(
                site_cfg=site_cfg,
                resolution_m=1000,  # doesn't matter for TBM
                context_cells=1,
                output_dir=case_dir,
                srtm_tif=None,  # skip STL generation (done separately)
                inflow_json=inflow_json,
                domain_km=tbm_cfg.get("cylinder", {}).get("radius", 7000) * 2 / 1000,
                domain_type="cylinder",
                solver_name=case_cfg["solver"],
                thermal=case_cfg.get("thermal", False),
                coriolis=case_cfg.get("coriolis", True),
                canopy_enabled=case_cfg.get("canopy", False),
                transport_T=case_cfg.get("transport_T", study.get("transport_T", False)),
                transport_q=case_cfg.get("transport_q", study.get("transport_q", False)),
                n_iter=study.get("n_iterations", 500),
                write_interval=study.get("write_interval", 100),
                lateral_patches=lateral_patches,
            )

        # --- Step 4: Mesh with TBM ---
        poly_mesh = case_dir / "constant" / "polyMesh"
        if not args.skip_mesh and not (poly_mesh / "points").exists():
            log.info("Running terrainBlockMesher...")
            generate_mesh_tbm(
                stl_path=stl_path,
                case_dir=case_dir,
                config=tbm_cfg,
                docker_image=docker_cfg.get("tbm_image", "terrainblockmesher:of24"),
                keep_tmp=True,
            )

            # checkMesh with OF2512
            log.info("checkMesh...")
            result = run_docker(
                case_dir, "checkMesh -latestTime 2>&1 | tail -10",
                image=docker_cfg.get("solver_image", "microfluidica/openfoam:latest"),
            )
            log.info("checkMesh:\n%s", result.stdout)

        # --- Step 5: Init fields ---
        u_file = case_dir / "0" / "U"
        if u_file.exists() and "nonuniform" not in u_file.read_text()[:3000]:
            log.info("Initializing fields from ERA5...")
            # writeCellCentres
            run_docker(
                case_dir, "postProcess -func writeCellCentres -time 0 > /dev/null 2>&1",
                image=docker_cfg.get("solver_image", "microfluidica/openfoam:latest"),
            )
            # Copy init script + inflow
            init_script = Path(__file__).parent / "init_from_era5.py"
            shutil.copy2(init_script, case_dir / "init_from_era5.py")
            shutil.copy2(inflow_json, case_dir / "inflow.json")
            # Run init (detects section_* patches automatically)
            result = subprocess.run(
                [sys.executable, "init_from_era5.py",
                 "--case-dir", ".", "--inflow", "inflow.json"],
                cwd=case_dir, capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.error("init_from_era5 failed: %s", result.stderr[-300:])
                raise RuntimeError(f"init failed for {case_id}")
            log.info("Fields initialized")

        # --- Step 6: Solve ---
        if not args.skip_solve:
            time_dirs = [d for d in case_dir.iterdir()
                         if d.is_dir() and d.name.replace(".", "").isdigit()
                         and float(d.name) > 0]
            if time_dirs:
                log.info("Already solved: %s", max(d.name for d in time_dirs))
            else:
                log.info("Running simpleFoam...")
                t0 = time.time()
                result = run_docker(
                    case_dir,
                    "simpleFoam > /case/log.simpleFoam 2>&1",
                    image=docker_cfg.get("solver_image", "microfluidica/openfoam:latest"),
                    timeout=3600,
                )
                wall = time.time() - t0
                if result.returncode != 0:
                    log.error("simpleFoam failed (%.0fs)", wall)
                    (case_dir / "log.simpleFoam").write_text(
                        result.stdout + "\n" + result.stderr)
                else:
                    log.info("Solved in %.0fs", wall)

        # --- Step 7: Evaluate ---
        log.info("Evaluating...")
        try:
            from evaluate_case import evaluate_single
            evaluate_single(case_dir, label=f"TBM {case_id}")
        except Exception as exc:
            log.warning("Evaluation failed: %s", exc)

    log.info("=== Study complete: %s ===", study["name"])


if __name__ == "__main__":
    main()
