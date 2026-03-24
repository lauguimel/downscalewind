"""
run_tbm_uga.py — End-to-end TBM pipeline: local mesh+init → UGA solve → local evaluate.

Usage
-----
    # Run a study config (physics sensitivity, convergence, etc.)
    python run_tbm_uga.py --config ../../configs/poc_tbm_convergence.yaml --nprocs 24

    # Multi-timestamp campaign (timestamps_csv in config)
    python run_tbm_uga.py --config ../../configs/poc_tbm_25ts.yaml --nprocs 24

    # Resume from a specific case
    python run_tbm_uga.py --config ../../configs/poc_tbm_25ts.yaml --skip-local --start-from ts05
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("run_tbm_uga")
ROOT = Path(__file__).resolve().parents[2]
UGA_BASE = "/home/guillaume/dsw/cases"


# ---------------------------------------------------------------------------
# Config expansion: timestamps_csv → cases
# ---------------------------------------------------------------------------

def expand_timestamps_csv(cfg: dict) -> None:
    """If study.timestamps_csv exists, generate cases from the CSV.

    Mutates cfg in place: adds entries to cfg["cases"].
    Each case inherits from cfg["defaults"] and gets its own timestamp.
    """
    study = cfg["study"]
    csv_rel = study.get("timestamps_csv")
    if not csv_rel:
        return

    csv_path = ROOT / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"timestamps_csv not found: {csv_path}")

    defaults = cfg.get("defaults", {})
    cases = cfg.setdefault("cases", {})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dt = row["datetime"]
            # case_id: ts00, ts01, ... (zero-padded for sort order)
            case_id = f"ts{i:02d}"
            cases[case_id] = {
                "timestamp": dt,
                "label": f"ERA5 {dt} ({float(row['speed_10m']):.1f} m/s, {float(row['direction_deg']):.0f}°)",
                **defaults,
            }

    log.info("Expanded %d timestamps from %s", len(cases), csv_path.name)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def generate_local(cfg: dict, cases_dir: Path) -> dict[str, Path]:
    """Generate all cases locally: STL + TBM mesh + templates + init.

    Optimizations for multi-timestamp campaigns:
    - Mesh is generated ONCE, then polyMesh is copied to subsequent cases
    - writeCellCentres runs once, Cx/Cy/Cz are copied
    - Per-case: inflow preparation + template rendering + init_from_era5
    """
    from generate_mesh_tbm import generate_mesh_tbm
    from generate_mesh import generate_mesh, dem_to_stl
    from prepare_inflow import prepare_inflow

    study = cfg["study"]
    site_cfg_path = ROOT / "configs" / "sites" / f"{study['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]
    era5_zarr = str(ROOT / "data" / "raw" / f"era5_{study['site']}.zarr")

    # Study-level inflow (if no per-case timestamps)
    study_ts = study.get("timestamp")
    study_inflow = cases_dir / "inflow.json"
    if study_ts and not study_inflow.exists():
        log.info("Generating study-level inflow...")
        prepare_inflow(
            era5_zarr=era5_zarr, timestamp=study_ts,
            site_lat=site_lat, site_lon=site_lon,
            output_json=study_inflow,
        )

    # STL (once, shared by all cases)
    stl_path = cases_dir / "terrain.stl"
    if not stl_path.exists():
        log.info("Generating terrain STL...")
        first_case = next(iter(cfg["cases"].values()))
        first_tbm = first_case.get("tbm", cfg.get("terrainBlockMesher", {}))
        radius = first_tbm.get("cylinder", {}).get("radius", 7000) * 1.1
        DEG_LAT = 1 / 111_000
        DEG_LON = 1 / (111_000 * np.cos(np.radians(site_lat)))
        dem_to_stl(
            srtm_tif=ROOT / "data" / "raw" / f"srtm_{study['site']}_30m.tif",
            out_stl=stl_path,
            bounds_lonlat=(site_lon - radius * DEG_LON, site_lat - radius * DEG_LAT,
                           site_lon + radius * DEG_LON, site_lat + radius * DEG_LAT),
            resolution_m=30, site_lat=site_lat, site_lon=site_lon,
        )

    # Reference mesh + cell centres (generated once, copied to all)
    ref_poly = None     # Path to first case's polyMesh
    ref_cc_dir = None   # Path to first case's 0/ with Cx, Cy, Cz

    case_dirs = {}
    for case_id, case_cfg in cfg["cases"].items():
        case_dir = cases_dir / f"case_{case_id}"
        tbm_cfg = case_cfg.get("tbm", cfg.get("terrainBlockMesher", {}))
        log.info("=== %s: %s ===", case_id, case_cfg.get("label", case_id))

        # 1. TBM mesh (generate once, copy for others)
        poly_dir = case_dir / "constant" / "polyMesh"
        if not (poly_dir / "points").exists():
            if ref_poly is not None:
                log.info("  Copying mesh from reference case...")
                poly_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(ref_poly, poly_dir)
            else:
                log.info("  Meshing (TBM Docker)...")
                generate_mesh_tbm(stl_path=stl_path, case_dir=case_dir,
                                  config=tbm_cfg, keep_tmp=True)
        if ref_poly is None:
            ref_poly = poly_dir

        # 2. Per-case inflow (if case has timestamp, else use study-level)
        case_ts = case_cfg.get("timestamp", study_ts)
        inflow_json = case_dir / "inflow.json"
        if not inflow_json.exists():
            if case_ts and case_ts != study_ts:
                log.info("  Preparing inflow for %s...", case_ts)
                inflow_json.parent.mkdir(parents=True, exist_ok=True)
                prepare_inflow(
                    era5_zarr=era5_zarr, timestamp=case_ts,
                    site_lat=site_lat, site_lon=site_lon,
                    output_json=inflow_json,
                )
            elif study_inflow.exists():
                shutil.copy2(study_inflow, inflow_json)

        # 3. z0 WorldCover (if requested, before templates)
        z0_mapped = case_cfg.get("z0_mapped", False)
        if z0_mapped:
            z0_bd = case_dir / "constant" / "boundaryData" / "terrain" / "0" / "z0"
            if not z0_bd.exists():
                from generate_z0_field import generate_z0_field
                wc_tif = ROOT / "data" / "raw" / f"worldcover_{study['site']}.tif"
                log.info("  Generating z0 from WorldCover...")
                generate_z0_field(case_dir, wc_tif, site_lat, site_lon)

        # 4. Templates (always re-render — inflow differs per case)
        n_sec = tbm_cfg.get("cylinder", {}).get("n_sections", 8)
        lateral_patches = [f"section_{i}" for i in range(n_sec)]
        if not (case_dir / "system" / "controlDict").exists():
            log.info("  Generating OF templates...")
            generate_mesh(
                site_cfg=site_cfg, resolution_m=1000, context_cells=1,
                output_dir=case_dir, srtm_tif=None, inflow_json=inflow_json,
                domain_km=tbm_cfg.get("cylinder", {}).get("radius", 7000) * 2 / 1000,
                domain_type="cylinder",
                solver_name=case_cfg.get("solver", "simpleFoam"),
                thermal=case_cfg.get("thermal", False),
                coriolis=case_cfg.get("coriolis", study.get("coriolis", True)),
                transport_T=case_cfg.get("transport_T", study.get("transport_T", False)),
                n_iter=study.get("n_iterations", 500),
                write_interval=study.get("write_interval", 500),
                lateral_patches=lateral_patches,
                z0_mapped=z0_mapped,
            )

        # 4. Ensure decomposeParDict exists
        decompose_dict = case_dir / "system" / "decomposeParDict"
        if not decompose_dict.exists():
            decompose_dict.write_text(
                'FoamFile { version 2.0; format ascii; class dictionary; '
                'object decomposeParDict; }\n'
                'numberOfSubdomains 24;\nmethod scotch;\n'
            )

        # 5. Copy helper scripts
        for script in ["init_from_era5.py", "reconstruct_fields.py"]:
            src = Path(__file__).parent / script
            if src.exists():
                shutil.copy2(src, case_dir / script)

        # 6. writeCellCentres (once) + init_from_era5 (per case)
        u_file = case_dir / "0" / "U"
        if not u_file.exists() or "nonuniform" not in u_file.read_text()[:3000]:
            # writeCellCentres: run once, copy for subsequent cases
            cx_file = case_dir / "0" / "Cx"
            if not cx_file.exists():
                if ref_cc_dir is not None and (ref_cc_dir / "Cx").exists():
                    log.info("  Copying cell centres from reference...")
                    for cc in ["Cx", "Cy", "Cz"]:
                        shutil.copy2(ref_cc_dir / cc, case_dir / "0" / cc)
                else:
                    log.info("  writeCellCentres (Docker)...")
                    subprocess.run(
                        ["docker", "run", "--rm",
                         "-v", f"{case_dir.resolve()}:/case", "-w", "/case",
                         "microfluidica/openfoam:latest", "bash", "-c",
                         "postProcess -func writeCellCentres -time 0 > /dev/null 2>&1"],
                        capture_output=True, timeout=300,
                    )
            if ref_cc_dir is None:
                ref_cc_dir = case_dir / "0"

            log.info("  init_from_era5...")
            subprocess.run(
                [sys.executable, "init_from_era5.py",
                 "--case-dir", ".", "--inflow", "inflow.json"],
                cwd=case_dir, capture_output=True, timeout=120,
            )

        case_dirs[case_id] = case_dir
        log.info("  Ready")

    return case_dirs


# ---------------------------------------------------------------------------
# UGA remote helpers
# ---------------------------------------------------------------------------

def scp_to_uga(local_dir: Path, remote_dir: str):
    """Copy a case directory to UGA."""
    log.info("  scp → UGA:%s", remote_dir)
    subprocess.run(
        ["scp", "-rq", str(local_dir), f"UGA:{remote_dir}"],
        timeout=1800, check=True,
    )


def ssh_run(cmd: str, timeout: int = 600) -> str:
    """Run a command on UGA via SSH. Returns stdout."""
    result = subprocess.run(
        ["ssh", "UGA", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip()


def solve_on_uga(case_id: str, remote_case: str, nprocs: int):
    """Solve a case on UGA via Docker.

    After decomposePar, symlinks constant/boundaryData into each processor
    directory so MappedFile BCs can find the points in parallel.
    """
    log.info("  Solving %s on UGA (%d cores)...", case_id, nprocs)

    if nprocs > 1:
        # Parallel: decomposePar + mpirun (--memory=16g prevents OOM on large meshes)
        cmd = (
            f'docker run --rm --cpus={nprocs} --memory=16g '
            f'-v {remote_case}:/case -w /case '
            f'microfluidica/openfoam:latest bash -c "'
            f'foamDictionary system/decomposeParDict -entry numberOfSubdomains -set {nprocs} && '
            f'rm -rf processor* && '
            f'decomposePar -force > /dev/null 2>&1 && '
            f'for d in processor*/; do ln -sf ../../constant/boundaryData \\$d/constant/boundaryData; done && '
            f'mpirun --allow-run-as-root -np {nprocs} simpleFoam -parallel '
            f'> /case/log.simpleFoam 2>&1"'
        )
    else:
        # Serial
        cmd = (
            f'docker run --rm --cpus=4 '
            f'-v {remote_case}:/case -w /case '
            f'microfluidica/openfoam:latest bash -c "'
            f'simpleFoam > /case/log.simpleFoam 2>&1"'
        )

    ssh_run(cmd, timeout=3600)

    # Verify success
    clock = ssh_run(f"grep ClockTime {remote_case}/log.simpleFoam | tail -1")
    ux = ssh_run(f"grep Ux {remote_case}/log.simpleFoam | tail -1")

    if not clock:
        log.error("  %s: solver produced no output — check log on UGA", case_id)
    else:
        log.info("  %s: %s | %s", case_id, clock, ux[:80] if ux else "no Ux")


def reconstruct_on_uga(remote_case: str, nprocs: int, n_iter: int = 500):
    """Reconstruct parallel results on UGA (if parallel)."""
    if nprocs <= 1:
        log.info("  Serial run — no reconstruction needed")
        return
    log.info("  Reconstructing on UGA...")
    out = ssh_run(
        f"cd {remote_case} && python3 reconstruct_fields.py "
        f"--case-dir . --time {n_iter} --write-foam 2>&1 | tail -2",
        timeout=300,
    )
    log.info("  %s", out)


def rsync_results(remote_case: str, local_case: Path, n_iter: int = 500):
    """Rsync solver results + logs from UGA."""
    log.info("  rsync ← UGA")
    local_case = Path(local_case)
    # Sync final time directory + log
    for sub in [f"{n_iter}/", "log.simpleFoam"]:
        src = f"UGA:{remote_case}/{sub}"
        dst = local_case / sub
        dst_parent = dst if sub.endswith("/") else dst.parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-az", src, str(dst)],
            capture_output=True, timeout=300,
        )
    # Sync polyMesh if not present locally
    if not (local_case / "constant" / "polyMesh" / "points").exists():
        subprocess.run(
            ["rsync", "-az", f"UGA:{remote_case}/constant/polyMesh/",
             str(local_case / "constant" / "polyMesh/")],
            capture_output=True, timeout=300,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="TBM pipeline: local mesh → UGA solve → evaluate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--nprocs", type=int, default=24, help="Cores per case on UGA")
    parser.add_argument("--skip-local", action="store_true", help="Skip local generation")
    parser.add_argument("--skip-solve", action="store_true", help="Skip UGA solve")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Skip cases before this case_id (for resuming)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Expand timestamps CSV into cases
    expand_timestamps_csv(cfg)

    study = cfg["study"]
    n_iter = study.get("n_iterations", 500)
    cases_dir = ROOT / "data" / "cases" / study["name"]
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate locally
    if not args.skip_local:
        log.info("===== Step 1: Local generation (%d cases) =====", len(cfg["cases"]))
        case_dirs = generate_local(cfg, cases_dir)
    else:
        case_dirs = {
            cid: cases_dir / f"case_{cid}" for cid in cfg["cases"]
        }

    # Step 2: Copy to UGA + solve
    if not args.skip_solve:
        remote_study = f"{UGA_BASE}/{study['name']}"
        ssh_run(f"mkdir -p {remote_study}")

        skip = args.start_from is not None
        for case_id, local_case in case_dirs.items():
            if skip:
                if case_id == args.start_from:
                    skip = False
                else:
                    log.info("  Skipping %s (before --start-from %s)", case_id, args.start_from)
                    continue

            remote_case = f"{remote_study}/case_{case_id}"
            nprocs = cfg["cases"][case_id].get("nprocs", args.nprocs)

            log.info("===== %s (→ UGA, %d cores) =====", case_id, nprocs)

            # Clean remote if root-owned
            ssh_run(
                f'docker run --rm -v {remote_study}:/d microfluidica/openfoam:latest '
                f'rm -rf /d/case_{case_id} 2>/dev/null; true'
            )

            scp_to_uga(local_case, f"{remote_study}/")
            solve_on_uga(case_id, remote_case, nprocs)

            # Reconstruct if parallel
            reconstruct_on_uga(remote_case, nprocs, n_iter)

            # Rsync results back
            rsync_results(remote_case, local_case, n_iter)

    # Step 3: Evaluate
    log.info("===== Step 3: Evaluate =====")
    try:
        from evaluate_case import evaluate_batch
        evaluate_batch(cases_dir)
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)

    log.info("===== DONE: %s =====", study["name"])


if __name__ == "__main__":
    main()
