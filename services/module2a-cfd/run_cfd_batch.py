"""
run_cfd_batch.py — Parametric CFD batch runner (local + HPC)

Generates and runs a grid of CFD cases from configs/training/cfd_grid.yaml.

Two modes
---------
  --mode local  : multiprocessing.Pool on N cores, runs sequentially per case
                  → 240 PoC runs on 8 cores, ~10–80h depending on resolution
  --mode hpc    : generates one Slurm script per run (or array job)
                  → submit with --submit for immediate launch

Local PoC grid (from cfd_grid.yaml, no perturbations):
  16 directions × 5 speeds × 3 stabilities = 240 runs

HPC production grid (full, with perturbations):
  32 directions × 10 speeds × 6 stabilities × ~50 perturbations ≈ 100k runs

Usage (local)
-------------
    python run_cfd_batch.py \
        --mode local \
        --site perdigao \
        --era5 data/raw/era5_perdigao.zarr \
        --srtm data/raw/srtm_perdigao_30m.tif \
        --z0map data/raw/z0_perdigao.tif \
        --grid configs/training/cfd_grid.yaml \
        --resolution-m 1000 \
        --context-cells 3 \
        --output data/cfd-database/perdigao \
        --n-processes 4

Usage (HPC — generate Slurm scripts only)
-----------------------------------------
    python run_cfd_batch.py \
        --mode hpc \
        --grid configs/training/cfd_grid.yaml \
        --resolution-m 1000 \
        --context-cells 3 \
        --output data/cfd-database/perdigao \
        --slurm-dir jobs/slurm \
        --partition compute \
        --time 01:00:00
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import textwrap
import time
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case parameter generation
# ---------------------------------------------------------------------------

def generate_case_params(grid_cfg: dict, perturbations: bool = False) -> list[dict]:
    """Expand the CDF grid configuration into a flat list of case parameter dicts.

    Parameters
    ----------
    grid_cfg:
        Parsed configs/training/cfd_grid.yaml.
    perturbations:
        If True, include all perturbation combinations (HPC mode).
        If False, use only the nominal values (PoC local mode, 240 runs).

    Returns
    -------
    List of dicts, one per case, with keys:
        case_id, direction_deg, speed_ms, stability,
        delta_T_grad, delta_ustar_frac, delta_dir_deg
    """
    directions = grid_cfg.get("directions_deg", [0, 45, 90, 135, 180, 225, 270, 315])
    speeds     = grid_cfg.get("speeds_ms",      [5, 8, 10, 12, 15])
    stabs      = grid_cfg.get("stabilities",    ["neutral"])

    if perturbations:
        T_grads   = grid_cfg.get("perturbations", {}).get("delta_T_grad_Km", [0])
        u_fracs   = grid_cfg.get("perturbations", {}).get("delta_ustar_frac", [1.0])
        dir_delts = grid_cfg.get("perturbations", {}).get("delta_dir_deg",   [0])
    else:
        T_grads   = [0]
        u_fracs   = [1.0]
        dir_delts = [0]

    cases = []
    for direction, speed, stability, dT, du, ddir in itertools.product(
        directions, speeds, stabs, T_grads, u_fracs, dir_delts
    ):
        eff_dir = (direction + ddir) % 360
        case_id = (
            f"dir{int(eff_dir):03d}_spd{int(speed*10):03d}_"
            f"{stability[:3]}_dT{int(dT):+d}_du{int(du*100):03d}"
        )
        cases.append({
            "case_id":          case_id,
            "direction_deg":    eff_dir,
            "speed_ms":         speed * du,
            "stability":        stability,
            "delta_T_grad":     dT,
            "delta_ustar_frac": du,
            "delta_dir_deg":    ddir,
        })

    return cases


# ---------------------------------------------------------------------------
# Local run worker
# ---------------------------------------------------------------------------

def _run_case_worker(args: tuple) -> dict:
    """Multiprocessing worker: run one CFD case end-to-end.

    Parameters (packed as tuple for Pool.map compatibility):
        (case_params, common_args)
    """
    case_params, common = args

    from prepare_inflow import prepare_inflow
    from generate_mesh import generate_mesh
    from openfoam_runner import OpenFOAMRunner
    from export_cfd import export_cfd
    from check_coherence import check_coherence, update_qc_report

    case_id   = case_params["case_id"]
    output    = Path(common["output"])
    case_dir  = output / "cases" / case_id
    site_cfg  = common["site_cfg"]
    site_lat  = site_cfg["site"]["coordinates"]["latitude"]
    site_lon  = site_cfg["site"]["coordinates"]["longitude"]
    n_cores   = common["n_cores_per_case"]
    res_m     = common["resolution_m"]
    ctx       = common["context_cells"]

    result = {**case_params, "ok": False, "error": None, "cpu_time_s": None}
    t0 = time.perf_counter()

    try:
        # 1 — inflow profile
        inflow_json = output / "inflow" / f"{case_id}.json"
        if not inflow_json.exists():
            prepare_inflow(
                era5_zarr=common["era5_zarr"],
                timestamp=case_params.get("era5_timestamp", "2017-05-15T12:00"),
                site_lat=site_lat,
                site_lon=site_lon,
                z0_tif=common.get("z0_tif"),
                output_json=inflow_json,
            )

        # 2 — mesh
        generate_mesh(
            site_cfg=site_cfg,
            resolution_m=res_m,
            context_cells=ctx,
            output_dir=case_dir,
            srtm_tif=common.get("srtm_tif"),
            inflow_json=inflow_json,
        )

        # 3 — solver
        runner = OpenFOAMRunner(case_dir, n_cores=n_cores)
        runner.run_case(inflow_json=inflow_json)

        # 4 — export
        towers_yaml = (
            Path(__file__).parents[2]
            / "configs" / "sites" / "perdigao_towers.yaml"
        )
        export_cfd(
            case_dir=case_dir,
            towers_yaml=towers_yaml,
            site_cfg=site_cfg,
            case_id=case_id,
            output_dir=output / "results",
            metadata={**case_params, "resolution_m": res_m, "context_cells": ctx},
        )

        # 5 — QC
        zarr_path = output / "results" / case_id / "fields.zarr"
        qc = check_coherence(
            case_id=case_id,
            zarr_path=zarr_path,
            case_dir=case_dir,
        )
        update_qc_report(qc, output / "qc_report.json")

        result["ok"]          = qc["overall_ok"]
        result["cpu_time_s"]  = time.perf_counter() - t0

    except Exception as exc:
        result["error"]       = str(exc)
        result["cpu_time_s"]  = time.perf_counter() - t0
        logger.error("[%s] ERROR: %s", case_id, exc)

    return result


# ---------------------------------------------------------------------------
# Slurm script generation (HPC mode)
# ---------------------------------------------------------------------------

def generate_slurm_scripts(
    cases: list[dict],
    common: dict,
    slurm_dir: Path,
    partition: str = "compute",
    time_limit: str = "01:00:00",
    n_cores: int = 8,
    submit: bool = False,
) -> None:
    """Generate one Slurm batch script per case (or an array job).

    Parameters
    ----------
    cases:
        List of case parameter dicts.
    common:
        Common arguments (era5, srtm, z0, site, resolution_m, etc.).
    slurm_dir:
        Directory to write Slurm scripts.
    partition, time_limit:
        Slurm resource parameters.
    n_cores:
        Cores per job.
    submit:
        If True, submit all scripts via sbatch.
    """
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_paths = []

    for case_params in cases:
        case_id = case_params["case_id"]
        script  = slurm_dir / f"{case_id}.sh"

        python_args = " ".join([
            f"--case-id {case_id}",
            f"--resolution-m {common['resolution_m']}",
            f"--context-cells {common['context_cells']}",
            f"--era5 {common['era5_zarr']}",
            f"--output {common['output']}",
            f"--n-cores {n_cores}",
        ])
        if common.get("srtm_tif"):
            python_args += f" --srtm {common['srtm_tif']}"
        if common.get("z0_tif"):
            python_args += f" --z0map {common['z0_tif']}"

        content = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={case_id[:20]}
            #SBATCH --partition={partition}
            #SBATCH --ntasks={n_cores}
            #SBATCH --cpus-per-task=1
            #SBATCH --time={time_limit}
            #SBATCH --output=logs/{case_id}_%j.out
            #SBATCH --error=logs/{case_id}_%j.err

            set -euo pipefail

            # Activate conda/venv environment
            # source /path/to/env/bin/activate

            cd {Path(__file__).parents[2].resolve()}

            python services/module2a-cfd/run_cfd_batch.py \\
                --mode single-case \\
                {python_args}
        """)

        script.write_text(content)
        script_paths.append(script)

    logger.info("Generated %d Slurm scripts in %s", len(script_paths), slurm_dir)

    if submit:
        import subprocess
        (slurm_dir / "logs").mkdir(exist_ok=True)
        for sp in script_paths:
            subprocess.run(["sbatch", str(sp)], check=True)
        logger.info("Submitted %d jobs via sbatch", len(script_paths))


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def write_batch_results(results: list[dict], output_dir: Path) -> None:
    """Append batch run results to a JSON log."""
    results_json = output_dir / "batch_results.json"
    existing = []
    if results_json.exists():
        with open(results_json) as f:
            existing = json.load(f)

    existing.extend(results)

    with open(results_json, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    n_ok   = sum(1 for r in results if r.get("ok"))
    n_fail = len(results) - n_ok
    logger.info("Batch results: %d OK, %d failed → %s", n_ok, n_fail, results_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Parametric CFD batch runner (local or HPC)"
    )
    parser.add_argument("--mode",   required=True,
                        choices=["local", "hpc", "single-case"],
                        help="local=multiprocessing, hpc=slurm scripts, "
                             "single-case=run one case (used by slurm scripts)")
    parser.add_argument("--site",   default="perdigao")
    parser.add_argument("--era5",   required=True,  help="ERA5 zarr store")
    parser.add_argument("--srtm",   default=None,   help="SRTM GeoTIFF")
    parser.add_argument("--z0map",  default=None,   help="z0 raster")
    parser.add_argument("--grid",   default="configs/training/cfd_grid.yaml",
                        help="CFD grid config (directions, speeds, stabilities)")
    parser.add_argument("--resolution-m",  type=float, default=1000.0)
    parser.add_argument("--context-cells", type=int,   default=3, choices=[1, 3, 5])
    parser.add_argument("--output",  required=True,  help="Output root directory")
    parser.add_argument("--n-processes",   type=int, default=1,
                        help="(local) Number of parallel Python processes")
    parser.add_argument("--n-cores",       type=int, default=8,
                        help="MPI cores per OpenFOAM case")
    parser.add_argument("--perturbations", action="store_true",
                        help="Include perturbations (HPC full grid)")
    # HPC-specific
    parser.add_argument("--slurm-dir",  default="jobs/slurm")
    parser.add_argument("--partition",  default="compute")
    parser.add_argument("--time",       default="01:00:00")
    parser.add_argument("--submit",     action="store_true",
                        help="(hpc) Submit Slurm jobs immediately")
    # single-case
    parser.add_argument("--case-id",    default=None)
    args = parser.parse_args()

    # Load site config
    cfg_path = (
        Path(__file__).parents[2]
        / "configs" / "sites" / f"{args.site}.yaml"
    )
    with open(cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    # Load grid
    grid_path = Path(args.grid)
    if grid_path.exists():
        with open(grid_path) as f:
            grid_cfg = yaml.safe_load(f)
    else:
        logger.warning("Grid config not found: %s — using defaults", grid_path)
        grid_cfg = {}

    common = {
        "site_cfg":         site_cfg,
        "era5_zarr":        args.era5,
        "srtm_tif":         args.srtm,
        "z0_tif":           args.z0map,
        "resolution_m":     args.resolution_m,
        "context_cells":    args.context_cells,
        "output":           args.output,
        "n_cores_per_case": args.n_cores,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- modes -------------------------------------------------------
    if args.mode == "local":
        cases = generate_case_params(grid_cfg, perturbations=args.perturbations)
        logger.info("Batch: %d cases, %d parallel process(es)", len(cases), args.n_processes)

        worker_args = [(c, common) for c in cases]

        if args.n_processes > 1:
            with mp.Pool(processes=args.n_processes) as pool:
                results = pool.map(_run_case_worker, worker_args)
        else:
            results = [_run_case_worker(a) for a in worker_args]

        write_batch_results(results, output_dir)

    elif args.mode == "hpc":
        cases = generate_case_params(grid_cfg, perturbations=args.perturbations)
        logger.info("HPC: generating %d Slurm scripts", len(cases))
        generate_slurm_scripts(
            cases=cases,
            common=common,
            slurm_dir=Path(args.slurm_dir),
            partition=args.partition,
            time_limit=args.time,
            n_cores=args.n_cores,
            submit=args.submit,
        )

    elif args.mode == "single-case":
        if not args.case_id:
            parser.error("--case-id is required for single-case mode")
        case_params = {"case_id": args.case_id}
        result = _run_case_worker((case_params, common))
        print(json.dumps(result, indent=2, default=str))
        if not result.get("ok"):
            raise SystemExit(1)
