"""
run_local_study.py — Local physics-progressive CFD study via Docker

Orchestrates case generation, meshing, solving, and validation for a
multi-case study on local machine using Docker (microfluidica/openfoam:latest).

Usage:
    cd services/module2a-cfd
    python run_local_study.py ../../configs/local_physics_study.yaml
    python run_local_study.py ../../configs/local_physics_study.yaml --cases A C
    python run_local_study.py ../../configs/local_physics_study.yaml --skip-mesh
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import yaml

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.logging_config import get_logger
from generate_mesh import generate_mesh
from prepare_inflow import prepare_inflow
from generate_campaign import build_parametric_inflow

log = get_logger("run_local_study")


# ---------------------------------------------------------------------------
# Docker execution
# ---------------------------------------------------------------------------

def run_docker(
    case_dir: Path,
    command: str,
    image: str = "microfluidica/openfoam:latest",
    timeout: int = 3600,
    platform: str | None = None,
) -> subprocess.CompletedProcess:
    """Run an OpenFOAM command inside Docker with the case dir mounted."""
    cmd = [
        "docker", "run", "--rm",
    ]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend([
        "-v", f"{case_dir.resolve()}:/case",
        "-w", "/case",
        image,
        "bash", "-c", f"cd /case && {command}",
    ])
    log.info("Docker run", extra={"command": command, "case": case_dir.name})
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        log.error("Docker failed", extra={
            "command": command,
            "stderr": result.stderr[-500:] if result.stderr else "(empty)",
        })
    return result


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_prepare_inflow(cfg: dict, output_json: Path) -> dict:
    """Generate inflow profile from ERA5 for the study timestamp."""
    if output_json.exists():
        log.info("Inflow already exists, loading", extra={"path": str(output_json)})
        return json.loads(output_json.read_text())

    site_cfg_path = ROOT / "configs" / "sites" / f"{cfg['study']['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    era5_zarr = ROOT / "data" / "raw" / f"era5_{cfg['study']['site']}.zarr"
    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]

    inflow = prepare_inflow(
        era5_zarr=era5_zarr,
        timestamp=cfg["study"]["timestamp"],
        site_lat=site_lat,
        site_lon=site_lon,
        output_json=output_json,
    )
    log.info("Inflow prepared", extra={
        "u_hub": f"{inflow['u_hub']:.1f} m/s",
        "wind_dir": f"{inflow['wind_dir']:.0f}°",
    })
    return inflow


def step_prepare_parametric_inflows(cfg: dict, cases_dir: Path) -> dict[str, Path]:
    """Generate one inflow JSON per case using parametric (no ERA5) profiles.

    Reads speed_ms and direction_deg from cfg['study']['inflow'], and
    stability preset from each case's 'stability' key.

    Returns {case_id: Path(inflow.json)}.
    """
    inflow_dir = cases_dir / "inflow_profiles"
    inflow_dir.mkdir(parents=True, exist_ok=True)

    inflow_cfg = cfg["study"]["inflow"]
    speed_ms = float(inflow_cfg["speed_ms"])
    direction_deg = float(inflow_cfg["direction_deg"])

    inflow_jsons: dict[str, Path] = {}
    for case_id, case_cfg in cfg["cases"].items():
        json_path = inflow_dir / f"inflow_{case_id}.json"
        if json_path.exists():
            log.info("Parametric inflow exists", extra={"case": case_id})
            inflow_jsons[case_id] = json_path
            continue

        stability = case_cfg.get("stability", "neutral")
        inflow_data = build_parametric_inflow(
            speed_ms=speed_ms,
            direction_deg=direction_deg,
            stability=stability,
        )
        json_path.write_text(__import__("json").dumps(inflow_data, indent=2))
        log.info("Parametric inflow written", extra={
            "case": case_id,
            "stability": stability,
            "speed_ms": speed_ms,
            "direction_deg": direction_deg,
        })
        inflow_jsons[case_id] = json_path

    return inflow_jsons


def step_generate_cases(
    cfg: dict,
    cases_dir: Path,
    inflow_jsons: "dict[str, Path] | Path",
) -> dict:
    """Generate OpenFOAM case directories for each study case.

    inflow_jsons may be either a single Path (shared across all cases, ERA5 mode)
    or a dict {case_id: Path} for parametric mode where each case has its own profile.
    """
    site_cfg_path = ROOT / "configs" / "sites" / f"{cfg['study']['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    srtm_tif = ROOT / "data" / "raw" / f"srtm_{cfg['study']['site']}_30m.tif"
    study = cfg["study"]
    case_dirs = {}

    for case_id, case_cfg in cfg["cases"].items():
        case_dir = cases_dir / f"case_{case_id}"
        if case_dir.exists():
            log.info("Case dir exists, skipping generation", extra={"case": case_id})
            case_dirs[case_id] = case_dir
            continue

        log.info("Generating case", extra={
            "case": case_id, "label": case_cfg["label"],
        })
        # Resolve per-case or shared inflow JSON
        inflow_json = (
            inflow_jsons[case_id] if isinstance(inflow_jsons, dict) else inflow_jsons
        )
        # flat: true → use flat terrain STL (srtm_tif=None) for box validation
        case_srtm = None if case_cfg.get("flat", False) else srtm_tif
        generate_mesh(
            site_cfg=site_cfg,
            resolution_m=study["resolution_m"],
            context_cells=study["context_cells"],
            output_dir=case_dir,
            srtm_tif=case_srtm,
            inflow_json=inflow_json,
            domain_km=study["domain_km"],
            solver_name=case_cfg["solver"],
            thermal=case_cfg.get("thermal", False),
            coriolis=case_cfg.get("coriolis", False),
            canopy_enabled=case_cfg.get("canopy", False),
            n_iter=study.get("n_iterations", 2000),
            write_interval=study.get("write_interval", 200),
        )
        case_dirs[case_id] = case_dir

    return case_dirs


def _detect_docker_config(cfg: dict) -> tuple[str, str | None, str, str | None]:
    """Detect best Docker images for meshing and solving.

    Returns (mesh_image, mesh_platform, solver_image, solver_platform).
    cfMesh is only in microfluidica/openfoam (amd64).
    Native ARM64 solver available in kraken/openfoam-ai.
    """
    import platform as plat
    is_arm = plat.machine() in ("arm64", "aarch64")

    mesh_image = "microfluidica/openfoam:latest"
    mesh_platform = "linux/amd64" if is_arm else None

    solver_image = cfg["docker"].get("image", "microfluidica/openfoam:latest")
    solver_platform = None

    # Prefer native ARM64 for solver
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


def step_mesh(case_dirs: dict, cfg: dict) -> None:
    """Run cartesianMesh on the first case, copy polyMesh to the rest."""
    mesh_image, mesh_platform, _, _ = _detect_docker_config(cfg)
    first_id = list(case_dirs.keys())[0]
    first_dir = case_dirs[first_id]
    poly_mesh = first_dir / "constant" / "polyMesh"

    if poly_mesh.exists() and (poly_mesh / "points").exists():
        log.info("Mesh already exists, skipping cartesianMesh")
    else:
        log.info("Running cartesianMesh", extra={
            "case": first_id, "image": mesh_image, "platform": mesh_platform,
        })
        result = run_docker(
            first_dir, "cartesianMesh",
            image=mesh_image, platform=mesh_platform, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cartesianMesh failed: {result.stderr[-300:]}")

        result = run_docker(
            first_dir, "checkMesh -latestTime",
            image=mesh_image, platform=mesh_platform,
        )
        log.info("checkMesh done", extra={
            "output": result.stdout[-200:] if result.stdout else "",
        })

    # Copy polyMesh to other cases
    for case_id, case_dir in case_dirs.items():
        if case_id == first_id:
            continue
        dst = case_dir / "constant" / "polyMesh"
        if dst.exists() and (dst / "points").exists():
            continue
        log.info("Copying polyMesh", extra={"from": first_id, "to": case_id})
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(poly_mesh, dst)


def step_init_fields(
    case_dirs: dict,
    inflow_jsons: "dict[str, Path] | Path",
    cfg: dict,
) -> None:
    """Initialize fields from inflow profile for each case.

    inflow_jsons may be a single shared Path (ERA5 mode) or
    a dict {case_id: Path} (parametric mode).
    """
    _, _, solver_image, solver_platform = _detect_docker_config(cfg)

    for case_id, case_dir in case_dirs.items():
        # Check if already initialized (internalField nonuniform in 0/U)
        u_file = case_dir / "0" / "U"
        if u_file.exists():
            u_text = u_file.read_text()[:5000]
            if "internalField   nonuniform" in u_text:
                log.info("Fields already initialized", extra={"case": case_id})
                continue

        log.info("Initializing fields", extra={"case": case_id})

        # Write cell centres
        run_docker(
            case_dir,
            "postProcess -func writeCellCentres -time 0",
            image=solver_image, platform=solver_platform,
        )

        # Resolve per-case or shared inflow JSON
        inflow_json = (
            inflow_jsons[case_id] if isinstance(inflow_jsons, dict) else inflow_jsons
        )

        # Copy inflow.json to case dir
        dst_inflow = case_dir / "inflow.json"
        if not dst_inflow.exists():
            shutil.copy2(inflow_json, dst_inflow)

        # Copy init_from_era5.py to case dir
        init_script = Path(__file__).parent / "init_from_era5.py"
        dst_init = case_dir / "init_from_era5.py"
        if not dst_init.exists():
            shutil.copy2(init_script, dst_init)

        # Run init_from_era5.py locally (needs numpy/scipy, not OpenFOAM)
        is_bbsf = "boussinesq" in cfg["cases"][case_id]["solver"].lower()
        init_cmd = [
            sys.executable, "init_from_era5.py",
            "--case-dir", ".", "--inflow", "inflow.json",
        ]
        if is_bbsf:
            init_cmd.append("--neutral-T-init")
        result = subprocess.run(
            init_cmd,
            cwd=case_dir,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("init_from_era5 failed", extra={
                "case": case_id, "stderr": result.stderr[-300:],
            })
            raise RuntimeError(f"init_from_era5.py failed for case {case_id}")

        log.info("Fields initialized", extra={"case": case_id})


def step_generate_lad(case_dirs: dict, cfg: dict) -> None:
    """Generate LAD field for canopy cases."""
    site_cfg_path = ROOT / "configs" / "sites" / f"{cfg['study']['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]
    landcover_tif = ROOT / cfg["validation"]["landcover_tif"]

    if not landcover_tif.exists():
        log.warning("Landcover TIF not found, skipping LAD generation",
                    extra={"path": str(landcover_tif)})
        return

    from generate_lad_field import generate_lad_field

    for case_id, case_cfg in cfg["cases"].items():
        if not case_cfg.get("canopy", False):
            continue

        case_dir = case_dirs[case_id]
        lad_file = case_dir / "0" / "LAD"
        if lad_file.exists():
            log.info("LAD already exists", extra={"case": case_id})
            continue

        # Ensure cell centres exist
        cx_file = case_dir / "0" / "Cx"
        if not cx_file.exists():
            _, _, solver_img, solver_plat = _detect_docker_config(cfg)
            run_docker(
                case_dir,
                "postProcess -func writeCellCentres -time 0",
                image=solver_img, platform=solver_plat,
            )

        log.info("Generating LAD field", extra={"case": case_id})
        stats = generate_lad_field(
            case_dir=case_dir,
            landcover_tif=landcover_tif,
            site_lat=site_lat,
            site_lon=site_lon,
        )
        log.info("LAD generated", extra={"case": case_id, **stats})


def step_precursor(case_dirs: dict, cfg: dict) -> None:
    """Copy converged SF solution as initial field for BBSF cases.

    BBSF is more stable when initialized from a converged SF solution
    (velocity/pressure already consistent, only thermal coupling to solve).
    """
    # Find a converged SF case to use as precursor
    sf_case_id = None
    sf_case_dir = None
    for cid, cdir in case_dirs.items():
        solver = cfg["cases"][cid]["solver"]
        if solver == "simpleFoam":
            # Check if it has converged results
            time_dirs = [d for d in cdir.iterdir()
                         if d.is_dir() and d.name.replace(".", "").isdigit()
                         and float(d.name) > 0]
            if time_dirs:
                sf_case_id = cid
                sf_case_dir = cdir
                break

    if sf_case_dir is None:
        return

    # Find the latest timestep
    time_dirs = [d for d in sf_case_dir.iterdir()
                 if d.is_dir() and d.name.replace(".", "").isdigit()
                 and float(d.name) > 0]
    latest = max(time_dirs, key=lambda d: float(d.name))

    for cid, cdir in case_dirs.items():
        solver = cfg["cases"][cid]["solver"]
        if "boussinesq" not in solver.lower():
            continue

        # Check if already has results
        bbsf_times = [d for d in cdir.iterdir()
                      if d.is_dir() and d.name.replace(".", "").isdigit()
                      and float(d.name) > 0]
        if bbsf_times:
            continue

        log.info("Precursor init", extra={
            "from": sf_case_id, "to": cid,
            "timestep": latest.name,
        })

        # Copy U, k, epsilon, nut, p_rgh from converged SF solution.
        # Both SF and BBSF use kinematic p_rgh [m2/s2]; when T ≈ T_ref the
        # buoyancy correction is negligible, so SF p_rgh is a valid BBSF start.
        for field in ["U", "k", "epsilon", "nut", "p_rgh"]:
            src = latest / field
            dst = cdir / "0" / field
            if src.exists() and dst.exists():
                shutil.copy2(src, dst)
                log.info("  Copied %s from %s/%s", field, sf_case_id, latest.name)

        # Note: T internalField is kept as uniform T_ref by init_from_era5.py
        # (--neutral-T-init flag in step_init_fields). No post-hoc reset needed.


def step_solve(case_dirs: dict, cfg: dict) -> dict:
    """Run the solver for each case and record timing."""
    _, _, solver_image, solver_platform = _detect_docker_config(cfg)
    nprocs = cfg["docker"].get("nprocs", 1)
    timings = {}

    # Sort: SF cases first, then BBSF (precursor order)
    sorted_cases = sorted(
        case_dirs.items(),
        key=lambda x: (0 if cfg["cases"][x[0]]["solver"] == "simpleFoam" else 1, x[0]),
    )

    for case_id, case_dir in sorted_cases:
        solver = cfg["cases"][case_id]["solver"]

        # For BBSF: copy precursor results from the first completed SF case
        if "boussinesq" in solver.lower():
            step_precursor({case_id: case_dir, **{k: v for k, v in case_dirs.items()
                           if cfg["cases"][k]["solver"] == "simpleFoam"}}, cfg)

        # Check if already solved (latestTime dir > 0)
        time_dirs = [d for d in case_dir.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and float(d.name) > 0]
        if time_dirs:
            log.info("Case already solved", extra={
                "case": case_id, "latest": max(d.name for d in time_dirs),
            })
            timings[case_id] = None
            continue

        log.info("Solving", extra={
            "case": case_id, "solver": solver,
            "image": solver_image,
        })
        t0 = time.time()

        if nprocs > 1:
            command = (
                f"foamDictionary system/decomposeParDict "
                f"-entry numberOfSubdomains -set {nprocs} && "
                f"decomposePar && "
                f"mpirun -np {nprocs} {solver} -parallel && "
                f"reconstructPar -latestTime"
            )
        else:
            command = solver

        result = run_docker(
            case_dir, command,
            image=solver_image, platform=solver_platform, timeout=3600,
        )
        wall_time = time.time() - t0
        timings[case_id] = wall_time

        if result.returncode != 0:
            log.error("Solver failed", extra={
                "case": case_id,
                "wall_time_s": f"{wall_time:.0f}",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            })
            # Write log for debugging
            log_file = case_dir / f"log.{solver}"
            log_file.write_text(result.stdout + "\n" + result.stderr)
        else:
            log.info("Solved", extra={
                "case": case_id,
                "wall_time_s": f"{wall_time:.0f}",
            })
            # Save solver log
            log_file = case_dir / f"log.{solver}"
            log_file.write_text(result.stdout)

    return timings


def step_export(case_dirs: dict, cfg: dict) -> dict:
    """Export CFD results at tower positions."""
    from export_cfd import export_cfd

    site_cfg_path = ROOT / "configs" / "sites" / f"{cfg['study']['site']}.yaml"
    towers_yaml = ROOT / "configs" / "sites" / "perdigao_towers.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    output_root = ROOT / cfg["validation"]["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)

    export_paths = {}
    for case_id, case_dir in case_dirs.items():
        label = cfg["cases"][case_id]["label"]
        csv_path = output_root / f"at_masts_{case_id}.csv"

        if csv_path.exists():
            log.info("Export already exists", extra={"case": case_id})
            export_paths[case_id] = csv_path
            continue

        log.info("Exporting", extra={"case": case_id})
        try:
            export_cfd(
                case_dir=case_dir,
                towers_yaml=towers_yaml,
                site_cfg=site_cfg,
                case_id=f"physics_{case_id}",
                output_dir=output_root,
                metadata={"label": label, "case_id": case_id},
            )
            # Move exported CSV to standard name
            src_csv = output_root / f"physics_{case_id}" / "at_masts.csv"
            if src_csv.exists():
                shutil.copy2(src_csv, csv_path)
            export_paths[case_id] = csv_path
        except Exception as e:
            log.error("Export failed", extra={"case": case_id, "error": str(e)})

    return export_paths


def step_compare(export_paths: dict, cfg: dict, timings: dict) -> None:
    """Compare CFD results with observations and produce figures."""
    from compare_cfd_obs import (
        load_cfd_masts, load_obs_snapshot,
        plot_multi_profiles, print_multi_summary, compare,
    )

    output_dir = ROOT / cfg["validation"]["output_dir"]
    obs_zarr = ROOT / cfg["validation"]["obs_zarr"]
    timestamp = cfg["study"]["timestamp"]
    towers = cfg["validation"].get("towers", ["tse04", "tse09", "tse13"])
    venkatraman_csv = ROOT / cfg["validation"].get("venkatraman_csv", "")

    # Load observations
    log.info("Loading observations", extra={"timestamp": timestamp})
    obs = load_obs_snapshot(obs_zarr, timestamp)

    # Load CFD results
    multi_cfd = {}
    multi_matched = {}
    for case_id, csv_path in export_paths.items():
        if not csv_path.exists():
            continue
        label = cfg["cases"][case_id]["label"]
        rows = load_cfd_masts(csv_path)
        multi_cfd[label] = rows
        matched = compare(rows, obs)
        multi_matched[label] = matched

    # Print summary table
    print_multi_summary(multi_matched)

    # Plot profiles with Venkatraman overlay
    log.info("Generating comparison figures")
    plot_multi_profiles(
        multi_cfd, obs, output_dir, towers=towers,
        venkatraman_csv=venkatraman_csv if venkatraman_csv.exists() else None,
    )

    # Save timing summary
    timing_file = output_dir / "timing.json"
    timing_data = {}
    for case_id, t in timings.items():
        label = cfg["cases"][case_id]["label"]
        timing_data[label] = {"wall_time_s": t, "case_id": case_id}
    timing_file.write_text(json.dumps(timing_data, indent=2))
    log.info("Timing saved", extra={"path": str(timing_file)})

    # Save metrics summary CSV
    metrics_file = output_dir / "summary_metrics.csv"
    with open(metrics_file, "w") as f:
        f.write("case,label,N,bias_ms,rmse_ms,hit_rate_pct,wall_time_s\n")
        for case_id in export_paths:
            label = cfg["cases"][case_id]["label"]
            m = multi_matched.get(label, {})
            if not m:
                continue
            import numpy as np
            cfd_s = np.array(m.get("cfd_speed", []))
            obs_s = np.array(m.get("obs_speed", []))
            if len(cfd_s) == 0:
                continue
            bias = float(np.mean(cfd_s - obs_s))
            rmse = float(np.sqrt(np.mean((cfd_s - obs_s) ** 2)))
            thr = np.maximum(2.0, 0.3 * obs_s)
            hr = float(100 * np.mean(np.abs(cfd_s - obs_s) < thr))
            wt = timings.get(case_id, "")
            f.write(f"{case_id},{label},{len(cfd_s)},{bias:.2f},{rmse:.2f},{hr:.1f},{wt}\n")
    log.info("Metrics saved", extra={"path": str(metrics_file)})


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--cases", multiple=True, default=None,
              help="Run only specific cases (e.g. --cases A --cases C)")
@click.option("--skip-mesh", is_flag=True, help="Skip mesh generation")
@click.option("--skip-solve", is_flag=True, help="Skip solver (export only)")
@click.option("--only-compare", is_flag=True, help="Only run comparison step")
def main(config, cases, skip_mesh, skip_solve, only_compare):
    """Run a local physics-progressive CFD study."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Filter cases if requested
    if cases:
        cfg["cases"] = {k: v for k, v in cfg["cases"].items() if k in cases}

    study_name = cfg["study"]["name"]
    cases_dir = ROOT / "data" / "cases" / study_name
    cases_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting local study", extra={
        "study_name": study_name,
        "cases": list(cfg["cases"].keys()),
        "resolution_m": cfg["study"]["resolution_m"],
    })

    # Inflow — ERA5 (single shared) or parametric (per-case)
    inflow_mode = cfg["study"].get("inflow_mode", "era5")
    if inflow_mode == "parametric":
        inflow_jsons = step_prepare_parametric_inflows(cfg, cases_dir)
    else:
        inflow_json = cases_dir / "inflow.json"
        step_prepare_inflow(cfg, inflow_json)
        inflow_jsons = inflow_json  # single Path, shared across cases

    # Generate cases
    case_dirs = step_generate_cases(cfg, cases_dir, inflow_jsons)

    if only_compare:
        if inflow_mode == "parametric":
            log.info("Parametric mode: skipping observation comparison")
            return
        export_paths = {}
        output_dir = ROOT / cfg["validation"]["output_dir"]
        for case_id in cfg["cases"]:
            csv_path = output_dir / f"at_masts_{case_id}.csv"
            if csv_path.exists():
                export_paths[case_id] = csv_path
        step_compare(export_paths, cfg, {k: None for k in cfg["cases"]})
        return

    # Mesh
    if not skip_mesh:
        step_mesh(case_dirs, cfg)

    # Init fields
    step_init_fields(case_dirs, inflow_jsons, cfg)

    # LAD for canopy cases
    step_generate_lad(case_dirs, cfg)

    # Precursor: init BBSF cases from converged SF solution
    step_precursor(case_dirs, cfg)

    # Solve
    timings = {}
    if not skip_solve:
        timings = step_solve(case_dirs, cfg)

    # Export
    export_paths = step_export(case_dirs, cfg)

    # Compare vs observations (ERA5 mode only — parametric has no matching timestamp)
    if inflow_mode != "parametric":
        step_compare(export_paths, cfg, timings)
    else:
        log.info("Parametric mode: skipping obs comparison (no ERA5 timestamp)")

    log.info("Study complete", extra={
        "cases_dir": str(cases_dir),
        "output_dir": cfg["validation"]["output_dir"],
    })


if __name__ == "__main__":
    main()
