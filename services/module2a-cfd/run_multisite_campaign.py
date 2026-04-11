"""
run_multisite_campaign.py — 100 sites × 15 timestamps = 1500-case campaign.

Runs entirely on UGA. Each site: extract STL → TBM mesh → 15 timestamps
(2 parallel solves × 24 cores) → export Zarr → delete raw OF to free disk.

Usage
-----
    python run_multisite_campaign.py --config campaign_1500.yaml
    python run_multisite_campaign.py --config campaign_1500.yaml --start-from site_00042
    python run_multisite_campaign.py --config campaign_1500.yaml --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

log = logging.getLogger("campaign")

# ---------------------------------------------------------------------------
# Defaults (overridable by YAML config)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
N_ITER = 500
N_CORES_PER_SOLVE = 24
N_PARALLEL_SOLVES = 2
OF_IMAGE = "microfluidica/openfoam:latest"
TBM_IMAGE = "terrainblockmesher:of24"
PYTHON = sys.executable
DOCKER_USER = f"{os.getuid()}:{os.getgid()}"

# Container runtime: set to "apptainer" on HPC clusters (Aqua).
# When using Apptainer, OF_IMAGE and TBM_IMAGE should point to .sif files.
CONTAINER_RUNTIME = os.environ.get("CONTAINER_RUNTIME", "docker")


def _container_cmd(image: str, mount_dir: Path, cmd: list[str],
                   workdir: str = "/home/ofuser/run") -> list[str]:
    """Build a container exec command for Docker or Apptainer."""
    d = str(mount_dir.resolve())
    if CONTAINER_RUNTIME == "apptainer":
        return [
            "apptainer", "exec", "--cleanenv",
            "--bind", f"{d}:{workdir}",
            "--pwd", workdir,
            image,
        ] + cmd
    else:
        return [
            "docker", "run", "--rm",
            "-v", f"{d}:{workdir}",
            "-w", workdir,
            image,
        ] + cmd

# TBM mesh parameters (same as Perdigão PoC)
# Must match PoC config (poc_tbm_25ts.yaml) → ~165k cells
DEFAULT_MESH = {
    "inner_size_m": 2000,
    "inner_blocks": 10,
    "cells_per_block_xy": 3,
    "cylinder_radius_m": 7000,
    "cylinder_sections": 8,
    "radial_cells": 20,
    "radial_grading": 20,
    "height_m": 5000,
    "cells_z": 50,
    "grading_z": 15,
    "max_dist_proj": 20000,
    "blend_distance_m": 5000,
    "p_above_z": 10000,
    "stl_resolution_m": 50,
}


# ===================================================================
# 1. STL extraction from regional DEM
# ===================================================================

def extract_stl(
    srtm_tif: Path,
    site_lat: float,
    site_lon: float,
    radius_m: float,
    stl_res_m: float,
    output_path: Path,
) -> float:
    """Extract terrain STL from regional SRTM for one site.

    Returns terrain_z_min (for TBM p_corner).
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds as window_from_bounds

    # Determine UTM zone from longitude
    utm_zone = int((site_lon + 180) / 6) + 1
    hemisphere = "north" if site_lat >= 0 else "south"
    utm_epsg = f"EPSG:326{utm_zone:02d}" if hemisphere == "north" else f"EPSG:327{utm_zone:02d}"

    transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    transformer_to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    x0, y0 = transformer_to_utm.transform(site_lon, site_lat)
    domain_m = 2 * radius_m + 2000  # STL slightly larger than cylinder
    half = domain_m / 2

    # Convert corners back to WGS84 for rasterio window
    lon_min, lat_min = transformer_to_wgs.transform(x0 - half, y0 - half)
    lon_max, lat_max = transformer_to_wgs.transform(x0 + half, y0 + half)

    nx = ny = int(domain_m / stl_res_m)

    with rasterio.open(srtm_tif) as src:
        window = window_from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        data = src.read(1, window=window)

        if data.size == 0 or np.all(data == src.nodata):
            raise ValueError(f"No SRTM data for site ({site_lat:.3f}, {site_lon:.3f})")

        # Resample to target resolution
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject, Resampling

        dst_transform = from_bounds(x0 - half, y0 - half, x0 + half, y0 + half, nx, ny)
        dst_array = np.zeros((ny, nx), dtype=np.float32)

        reproject(
            source=data,
            destination=dst_array,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=utm_epsg,
            resampling=Resampling.bilinear,
        )

    # Replace nodata with min valid elevation
    valid_mask = dst_array > -500
    if not valid_mask.any():
        raise ValueError(f"All nodata for site ({site_lat:.3f}, {site_lon:.3f})")
    z_min_valid = float(dst_array[valid_mask].min())
    dst_array[~valid_mask] = z_min_valid

    terrain_z_min = z_min_valid - 50

    # Write ASCII STL in local coordinates (centre = 0,0)
    xs = np.linspace(-half, half, nx)
    ys = np.linspace(-half, half, ny)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ntri = 0
    with open(output_path, "w") as f:
        f.write("solid terrain\n")
        for j in range(ny - 1):
            for i in range(nx - 1):
                p = [
                    (xs[i], ys[j], float(dst_array[j, i])),
                    (xs[i + 1], ys[j], float(dst_array[j, i + 1])),
                    (xs[i], ys[j + 1], float(dst_array[j + 1, i])),
                    (xs[i + 1], ys[j + 1], float(dst_array[j + 1, i + 1])),
                ]
                for tri in [(p[0], p[1], p[2]), (p[1], p[3], p[2])]:
                    f.write("  facet normal 0 0 1\n    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}\n")
                    f.write("    endloop\n  endfacet\n")
                    ntri += 1
        f.write("endsolid terrain\n")

    log.info("STL: %d triangles, %.1f MB, z=[%.0f, %.0f] → %s",
             ntri, output_path.stat().st_size / 1e6,
             dst_array.min(), dst_array.max(), output_path.name)

    return terrain_z_min


# ===================================================================
# 2. TBM mesh generation
# ===================================================================

def write_tbm_dict(mesh_cfg: dict, stl_name: str, case_dir: Path,
                   terrain_z_min: float) -> None:
    """Write terrainBlockMesherDict from mesh config."""
    m = mesh_cfg
    half = m["inner_size_m"] / 2
    n_blocks = m["inner_blocks"]
    cpb = m["cells_per_block_xy"]

    dict_content = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object terrainBlockMesherDict; }}

stl {{ {stl_name} {{ type triSurfaceMesh; }} }};

writeBlockMeshDict  true;
writePolyMesh       true;

blockManager
{{
    coordinates
    {{
        origin      (0 0 0);
        baseVectors (( 1 0 0 )( 0 1 0 )( 0 0 1 ));
    }}

    p_corner    ({-half:.0f} {-half:.0f} {terrain_z_min:.0f});
    dimensions  ({m["inner_size_m"]:.0f} {m["inner_size_m"]:.0f} {m["height_m"]:.0f});

    p_above     (0 0 {m["p_above_z"]});

    blocks      ({n_blocks} {n_blocks} 1);
    cells       ({cpb} {cpb} {m["cells_z"]});

    maxDistProj {m["max_dist_proj"]};

    gradingFactors  ( 1 1 {m["grading_z"]} );

    patch_name_west     west;
    patch_name_east     east;
    patch_name_north    north;
    patch_name_south    south;
    patch_name_sky      top;
    patch_name_ground   terrain;

    patch_type_west     patch;
    patch_type_east     patch;
    patch_type_north    patch;
    patch_type_south    patch;
    patch_type_sky      patch;
    patch_type_ground   wall;

    outerCylinder
    {{
        centrePoint                 (0 0 {terrain_z_min:.0f});
        radius                      {m["cylinder_radius_m"]};
        radialGrading               {m["radial_grading"]};
        radialBlockCells            {m["radial_cells"]};
        firstSectionStartDirection  (-1 1 0);
        numberOfSections            {m["cylinder_sections"]};

        blendingFunction
        {{
            type    distance;
            dMin    0;
            dMax    {m["blend_distance_m"]};

            transitionFunction
            {{
                type    linear;
            }}
        }}
    }}
}}

checkMesh           false;
checkMeshNoTopology false;
checkMeshAllGeometry false;
checkMeshAllTopology false;
"""
    system = case_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "terrainBlockMesherDict").write_text(dict_content)

    # Minimal OF system files (TBM needs these)
    for name, content in [
        ("controlDict",
         "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
         "application terrainBlockMesher; startFrom latestTime; startTime 0; stopAt endTime;\n"
         "endTime 0; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;\n"
         "writePrecision 10; writeCompression uncompressed; timeFormat general; timePrecision 6;\n"),
        ("fvSchemes",
         "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
         "ddtSchemes { default steadyState; }\n"
         "gradSchemes { default Gauss linear; }\n"
         "divSchemes { default none; }\n"
         "laplacianSchemes { default Gauss linear corrected; }\n"
         "interpolationSchemes { default linear; }\n"
         "snGradSchemes { default corrected; }\n"),
        ("fvSolution",
         "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"),
    ]:
        (system / name).write_text(content)


def run_tbm_mesh(mesh_dir: Path, mesh_cfg: dict) -> bool:
    """Run terrainBlockMesher via Docker. Returns True on success."""
    poly = mesh_dir / "constant" / "polyMesh"
    if poly.exists() and (poly / "points").exists():
        log.info("  polyMesh already exists, skipping mesher")
        return True

    t0 = time.time()
    chown_suffix = f"; chown -R {DOCKER_USER} /home/ofuser/run" if CONTAINER_RUNTIME == "docker" else ""
    cmd = _container_cmd(
        TBM_IMAGE, mesh_dir,
        ["bash", "-c",
         f"cd /home/ofuser/run && terrainBlockMesher; rc=$?{chown_suffix}; exit $rc"],
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.stdout:
        (mesh_dir / "log.terrainBlockMesher").write_text(result.stdout)

    if result.returncode != 0:
        log.error("  TBM failed (rc=%d): %s", result.returncode,
                  (result.stderr or "")[-300:])
        return False

    log.info("  Mesh done in %.0f s", time.time() - t0)
    return True


def run_write_cell_centres(mesh_dir: Path) -> bool:
    """Run postProcess -func writeCellCentres ONCE on mesh_dir.

    Needs a minimal 0/p file so OF recognises time=0.
    """
    cx = mesh_dir / "0" / "Cx"
    if cx.exists():
        return True

    # Create minimal 0/p so writeCellCentres has a time directory
    zero_dir = mesh_dir / "0"
    zero_dir.mkdir(exist_ok=True)
    (zero_dir / "p").write_text(
        "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
        "dimensions [0 2 -2 0 0 0 0];\n"
        "internalField uniform 0;\n"
        "boundaryField { \".*\" { type zeroGradient; } }\n"
    )

    chown_suffix = f"; chown -R {DOCKER_USER} /home/ofuser/run/0" if CONTAINER_RUNTIME == "docker" else ""
    cmd = _container_cmd(
        OF_IMAGE, mesh_dir,
        ["bash", "-c",
         f"cd /home/ofuser/run && postProcess -func writeCellCentres -time 0{chown_suffix}"],
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not cx.exists():
        log.error("  writeCellCentres failed: %s", (result.stderr or "")[-200:])
        return False
    return True


# ===================================================================
# 3. Per-timestamp case setup
# ===================================================================

def setup_case(
    case_dir: Path,
    mesh_dir: Path,
    era5_zarr: Path,
    timestamp: str,
    site_lat: float,
    site_lon: float,
    mesh_cfg: dict,
    n_iter: int,
) -> bool:
    """Set up one case: copy mesh, prepare inflow, render templates, init fields.

    Returns True on success.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy shared mesh + cell centres ---
    dst_poly = case_dir / "constant" / "polyMesh"
    if not dst_poly.exists():
        src_poly = mesh_dir / "constant" / "polyMesh"
        shutil.copytree(src_poly, dst_poly)
    for fname in ("Cx", "Cy", "Cz"):
        src = mesh_dir / "0" / fname
        dst = case_dir / "0" / fname
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # --- Prepare inflow ---
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        result = subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / "prepare_inflow.py"),
             "--era5", str(era5_zarr),
             "--case", timestamp,
             "--lat", str(site_lat),
             "--lon", str(site_lon),
             "--output", str(inflow_json)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.error("  prepare_inflow failed for %s: %s",
                      timestamp, (result.stderr or "")[-300:])
            return False

    with open(inflow_json) as f:
        inflow = json.load(f)

    # --- Render Jinja2 templates ---
    from jinja2 import Environment, FileSystemLoader

    template_dir = SCRIPTS_DIR / "templates" / "openfoam"
    n_sections = mesh_cfg.get("cylinder_sections", 8)
    lateral_patches = [f"section_{i}" for i in range(n_sections)]

    jinja_ctx = {
        "domain": {
            "octagonal": True,
            "lateral_patches": lateral_patches,
            "radius_m": mesh_cfg["cylinder_radius_m"],
            "z_max": mesh_cfg["height_m"],
        },
        "solver": {
            "name": "simpleFoam",
            "n_iter": n_iter,
            "n_cores": N_CORES_PER_SOLVE,
            "write_interval": n_iter,  # only write final step
            "transport_T": True,
            "transport_q": True,
        },
        "physics": {
            "coriolis": True,
            "T_ref_K": inflow.get("T_ref", 288.15),
        },
        "canopy": {"enabled": False},
        "inflow": inflow,
        "site": {"latitude": site_lat, "longitude": site_lon},
    }

    env = Environment(loader=FileSystemLoader(str(template_dir)),
                      keep_trailing_newline=True)
    skip = {"meshDict.j2", "terrainBlockMesherDict.j2"}
    for tmpl_path in sorted(template_dir.rglob("*.j2")):
        if tmpl_path.name in skip:
            continue
        rel = tmpl_path.relative_to(template_dir)
        out_file = case_dir / rel.with_suffix("")
        tmpl = env.get_template(str(rel))
        rendered = tmpl.render(**jinja_ctx)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(rendered)

    # --- Init from ERA5 ---
    init_script = SCRIPTS_DIR / "init_from_era5.py"
    result = subprocess.run(
        [PYTHON, str(init_script),
         "--case-dir", str(case_dir),
         "--inflow", str(inflow_json)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        log.error("  init_from_era5 failed for %s: %s",
                  case_dir.name, (result.stderr or "")[-300:])
        return False

    return True


# ===================================================================
# 4. Solve (parallel: decompose → mpirun → reconstruct)
# ===================================================================

def solve_case(case_dir: Path, n_iter: int, n_cores: int) -> bool:
    """Run one CFD solve. Returns True on success."""
    # Check if already solved
    final_time = case_dir / str(n_iter) / "U"
    if final_time.exists():
        log.info("  %s already solved, skipping", case_dir.name)
        return True

    t0 = time.time()

    # --- Pre-Docker setup (avoids entrypoint/env issues) ---
    # turbulenceProperties alias (OF ESI v2512 compat)
    mt = case_dir / "constant" / "momentumTransport"
    tp = case_dir / "constant" / "turbulenceProperties"
    if mt.exists() and not tp.exists():
        shutil.copy2(mt, tp)

    # Set numberOfSubdomains in decomposeParDict (avoid foamDictionary in Docker)
    dpd = case_dir / "system" / "decomposeParDict"
    if dpd.exists():
        txt = dpd.read_text()
        import re
        txt = re.sub(r'numberOfSubdomains\s+\d+', f'numberOfSubdomains  {n_cores}', txt)
        dpd.write_text(txt)

    # Helper: run OF command in container (Docker or Apptainer)
    def _docker_of(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run(
            _container_cmd(OF_IMAGE, case_dir, cmd),
            capture_output=True, text=True, timeout=timeout,
        )

    # Clean leftover processor dirs (may be root-owned from failed runs)
    _docker_of(["find", "/home/ofuser/run", "-maxdepth", "1",
                "-name", "processor*", "-exec", "rm", "-rf", "{}", "+"], timeout=30)

    # decomposePar
    result = _docker_of(["decomposePar", "-force"], timeout=120)
    (case_dir / "log.decomposePar").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        log.error("  decomposePar failed for %s", case_dir.name)
        return False

    # Symlink boundaryData into processor dirs
    if (case_dir / "constant" / "boundaryData").exists():
        for proc_dir in sorted(case_dir.glob("processor*")):
            bd_dst = proc_dir / "constant" / "boundaryData"
            if not bd_dst.exists():
                try:
                    bd_dst.symlink_to(
                        (case_dir / "constant" / "boundaryData").resolve())
                except PermissionError:
                    # Root-owned proc dir — symlink via Docker
                    _docker_of(["bash", "-c",
                        f"ln -sf ../../constant/boundaryData "
                        f"/home/ofuser/run/{proc_dir.name}/constant/"],
                        timeout=10)

    # mpirun simpleFoam (direct exec — entrypoint sets OF env)
    result = _docker_of(
        ["mpirun", "-np", str(n_cores), "simpleFoam", "-parallel"],
        timeout=3600)
    (case_dir / "log.simpleFoam").write_text(result.stdout + result.stderr)

    # chown all files back to user (Docker only — Apptainer preserves UID)
    if CONTAINER_RUNTIME == "docker":
        _docker_of(["chown", "-R", DOCKER_USER, "/home/ofuser/run"], timeout=60)

    wall = time.time() - t0

    if result.returncode != 0:
        log.error("  Solver FAILED for %s (%.0f s)", case_dir.name, wall)
        # Save what we can
        if result.stdout:
            (case_dir / "log.simpleFoam.stdout").write_text(result.stdout)
        return False

    log.info("  Solve %s: %.0f s", case_dir.name, wall)

    # Reconstruct fields (Python-based, avoids phi expression BC issue)
    result = subprocess.run(
        [PYTHON, str(SCRIPTS_DIR / "reconstruct_fields.py"),
         "--case-dir", str(case_dir),
         "--time", "latest",
         "--write-foam",
         "--fields", "U", "T", "q", "k", "epsilon", "nut", "p"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        log.error("  reconstruct_fields failed for %s: %s",
                  case_dir.name, (result.stderr or "")[-200:])
        return False

    # Clean processor dirs (user-owned thanks to --user)
    for proc_dir in sorted(case_dir.glob("processor*")):
        shutil.rmtree(proc_dir, ignore_errors=True)

    return True


def solve_batch(cases: list[Path], n_iter: int, n_cores: int,
                max_parallel: int) -> dict[str, bool]:
    """Run multiple solves in parallel. Returns {case_name: success}."""
    results = {}
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(solve_case, case, n_iter, n_cores): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case.name] = future.result()
            except Exception as e:
                log.error("  %s raised %s: %s", case.name, type(e).__name__, e)
                results[case.name] = False
    return results


# ===================================================================
# 5. Export Zarr + disk cleanup
# ===================================================================

def export_site(site_dir: Path, n_iter: int) -> bool:
    """Export solved cases to Zarr and delete raw OF dirs."""
    zarr_path = site_dir / f"{site_dir.name}.zarr"
    if zarr_path.exists():
        log.info("  Zarr already exists: %s", zarr_path)
        return True

    result = subprocess.run(
        [PYTHON, str(SCRIPTS_DIR / "export_campaign_zarr.py"),
         "--cases-dir", str(site_dir),
         "--output", str(zarr_path),
         "--time", str(n_iter)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        log.error("  export_campaign_zarr failed for %s: %s",
                  site_dir.name, (result.stderr or "")[-300:])
        return False

    if not zarr_path.exists():
        log.error("  Zarr NOT created for %s (export returned 0 but no output)",
                  site_dir.name)
        return False

    log.info("  Zarr exported: %s", zarr_path)

    # Delete raw OF case directories to free disk
    for case_dir in sorted(site_dir.glob("case_ts*")):
        shutil.rmtree(case_dir, ignore_errors=True)
    log.info("  Cleaned raw case dirs for %s", site_dir.name)

    return True


# ===================================================================
# 6. Main campaign loop
# ===================================================================

def load_run_matrix(csv_path: Path) -> dict[str, list[dict]]:
    """Load run_matrix.csv → {site_id: [row, ...]}."""
    sites = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["site_id"]
            sites.setdefault(sid, []).append(row)
    return sites


def process_site(
    site_id: str,
    runs: list[dict],
    campaign_dir: Path,
    srtm_tif: Path,
    era5_zarr: Path,
    mesh_cfg: dict,
    n_iter: int,
    n_cores: int,
    max_parallel: int,
    dry_run: bool = False,
) -> dict:
    """Process one site end-to-end. Returns status dict."""
    site_lat = float(runs[0]["lat"])
    site_lon = float(runs[0]["lon"])
    site_dir = campaign_dir / site_id
    mesh_dir = site_dir / "mesh"

    status = {
        "site_id": site_id,
        "lat": site_lat,
        "lon": site_lon,
        "n_timestamps": len(runs),
        "mesh_ok": False,
        "n_solved": 0,
        "n_failed": 0,
        "zarr_ok": False,
    }

    log.info("=" * 60)
    log.info("SITE %s (%.3f, %.3f) — %d timestamps",
             site_id, site_lat, site_lon, len(runs))
    log.info("=" * 60)

    if dry_run:
        log.info("  [DRY RUN] would process %d cases", len(runs))
        return status

    # ---- ERA5 resolution: use per-site Zarr if available ----
    # Convention: if era5_zarr parent dir contains era5_<site_id>.zarr, use it
    site_era5 = era5_zarr.parent / f"era5_{site_id.lower()}.zarr"
    if site_era5.exists() and site_era5 != era5_zarr:
        log.info("  Using per-site ERA5: %s", site_era5)
        era5_zarr = site_era5

    # ---- Step 1: STL ----
    stl_path = mesh_dir / "constant" / "triSurface" / "terrain.stl"
    try:
        if not stl_path.exists():
            terrain_z_min = extract_stl(
                srtm_tif, site_lat, site_lon,
                mesh_cfg["cylinder_radius_m"],
                mesh_cfg["stl_resolution_m"],
                stl_path,
            )
        else:
            # Read z_min from existing STL
            terrain_z_min = _read_stl_zmin(stl_path)
    except Exception as e:
        log.error("  STL extraction failed: %s", e)
        return status

    # ---- Step 2: TBM mesh ----
    write_tbm_dict(mesh_cfg, "terrain.stl", mesh_dir, terrain_z_min)
    if not run_tbm_mesh(mesh_dir, mesh_cfg):
        log.error("  Mesh FAILED for %s — skipping site", site_id)
        return status

    if not run_write_cell_centres(mesh_dir):
        log.error("  writeCellCentres FAILED for %s — skipping site", site_id)
        return status

    status["mesh_ok"] = True

    # ---- Step 3: Set up all cases ----
    case_dirs = []
    for i, run in enumerate(runs):
        ts = run["timestamp"]
        case_name = f"case_ts{i:03d}"
        case_dir = site_dir / case_name

        # Check if already solved
        if (case_dir / str(n_iter) / "U").exists():
            log.info("  %s already solved", case_name)
            case_dirs.append(case_dir)
            continue

        log.info("  Setting up %s (%s)", case_name, ts)
        try:
            ok = setup_case(
                case_dir, mesh_dir, era5_zarr, ts,
                site_lat, site_lon, mesh_cfg, n_iter,
            )
            if ok:
                case_dirs.append(case_dir)
            else:
                status["n_failed"] += 1
        except Exception as e:
            log.error("  Setup failed for %s: %s", case_name, e)
            status["n_failed"] += 1

    # ---- Step 4: Solve (batched, 4 at a time) ----
    to_solve = [c for c in case_dirs if not (c / str(n_iter) / "U").exists()]
    already_solved = len(case_dirs) - len(to_solve)

    if to_solve:
        log.info("  Solving %d cases (%d already done), %d parallel × %d cores",
                 len(to_solve), already_solved, max_parallel, n_cores)
        solve_results = solve_batch(to_solve, n_iter, n_cores, max_parallel)
        status["n_solved"] = sum(1 for v in solve_results.values() if v) + already_solved
        status["n_failed"] += sum(1 for v in solve_results.values() if not v)
    else:
        status["n_solved"] = already_solved
        log.info("  All %d cases already solved", already_solved)

    # ---- Step 5: Export Zarr + cleanup ----
    if status["n_solved"] > 0:
        status["zarr_ok"] = export_site(site_dir, n_iter)

    return status


def _read_stl_zmin(stl_path: Path) -> float:
    """Read z_min from an existing ASCII STL."""
    z_min = 1e9
    n = 0
    with open(stl_path) as f:
        for line in f:
            if "vertex" in line:
                parts = line.split()
                z = float(parts[3])
                if z < z_min:
                    z_min = z
                n += 1
                if n > 10000:
                    break
    return z_min - 50 if z_min < 1e9 else 0.0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("campaign.log", mode="a"),
        ],
    )

    parser = argparse.ArgumentParser(description="Multi-site CFD campaign (1500 cases)")
    parser.add_argument("--run-matrix", type=Path, required=True,
                        help="Path to run_matrix.csv")
    parser.add_argument("--srtm", type=Path, required=True,
                        help="Path to regional SRTM GeoTIFF (srtm_europe.tif)")
    parser.add_argument("--era5-zarr", type=Path, required=True,
                        help="Path to ERA5 Zarr store (shared for PoC)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Campaign output directory")
    parser.add_argument("--n-iter", type=int, default=N_ITER,
                        help=f"Solver iterations (default: {N_ITER})")
    parser.add_argument("--n-cores", type=int, default=N_CORES_PER_SOLVE,
                        help=f"Cores per solve (default: {N_CORES_PER_SOLVE})")
    parser.add_argument("--n-parallel", type=int, default=N_PARALLEL_SOLVES,
                        help=f"Parallel solves (default: {N_PARALLEL_SOLVES})")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume from this site_id (skip earlier sites)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sites without processing")
    parser.add_argument("--runtime", choices=["docker", "apptainer"],
                        default=None,
                        help="Container runtime (default: from CONTAINER_RUNTIME env var or 'docker'). "
                             "Use 'apptainer' on HPC.")
    parser.add_argument("--of-image", default=None,
                        help=f"OpenFOAM container image (default: {OF_IMAGE})")
    parser.add_argument("--tbm-image", default=None,
                        help=f"TBM container image (default: {TBM_IMAGE})")
    args = parser.parse_args()

    # Apply runtime overrides to module-level variables
    _mod = sys.modules[__name__]
    if args.runtime:
        _mod.CONTAINER_RUNTIME = args.runtime
    if args.of_image:
        _mod.OF_IMAGE = args.of_image
    if args.tbm_image:
        _mod.TBM_IMAGE = args.tbm_image

    # Load run matrix
    run_matrix = load_run_matrix(args.run_matrix)
    site_ids = sorted(run_matrix.keys())
    log.info("Loaded %d sites, %d total runs from %s",
             len(site_ids), sum(len(v) for v in run_matrix.values()),
             args.run_matrix)

    # Resume support
    if args.start_from:
        if args.start_from not in site_ids:
            log.error("Site %s not found in run matrix", args.start_from)
            sys.exit(1)
        idx = site_ids.index(args.start_from)
        site_ids = site_ids[idx:]
        log.info("Resuming from %s (%d sites remaining)", args.start_from, len(site_ids))

    # Mesh config
    mesh_cfg = dict(DEFAULT_MESH)

    args.output.mkdir(parents=True, exist_ok=True)

    # Campaign summary
    results = []
    t_campaign = time.time()

    for i, site_id in enumerate(site_ids):
        log.info("\n[%d/%d] Processing %s", i + 1, len(site_ids), site_id)
        t_site = time.time()

        try:
            status = process_site(
                site_id=site_id,
                runs=run_matrix[site_id],
                campaign_dir=args.output,
                srtm_tif=args.srtm,
                era5_zarr=args.era5_zarr,
                mesh_cfg=mesh_cfg,
                n_iter=args.n_iter,
                n_cores=args.n_cores,
                max_parallel=args.n_parallel,
                dry_run=args.dry_run,
            )
        except Exception as e:
            log.error("SITE %s CRASHED: %s", site_id, e, exc_info=True)
            status = {"site_id": site_id, "error": str(e)}

        status["wall_time_s"] = time.time() - t_site
        results.append(status)

        # Write incremental progress
        progress_path = args.output / "campaign_progress.json"
        with open(progress_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary
    wall_total = time.time() - t_campaign
    n_ok = sum(1 for r in results if r.get("zarr_ok"))
    n_solved = sum(r.get("n_solved", 0) for r in results)
    n_failed = sum(r.get("n_failed", 0) for r in results)

    log.info("=" * 60)
    log.info("CAMPAIGN COMPLETE")
    log.info("  Sites: %d/%d exported to Zarr", n_ok, len(results))
    log.info("  Cases: %d solved, %d failed", n_solved, n_failed)
    log.info("  Wall time: %.1f h", wall_total / 3600)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
