"""
run_tbm_case.py — End-to-end: terrainBlockMesher mesh → init → simpleFoam.

Reads a YAML config with all mesh parameters, generates STL from SRTM,
runs terrainBlockMesher in Docker, initializes fields from ERA5, solves.

Usage
-----
    cd services/module2a-cfd
    python run_tbm_case.py ../../configs/tbm_test.yaml
    python run_tbm_case.py ../../configs/tbm_test.yaml --mesh-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform as plat
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("run_tbm_case")


# ---------------------------------------------------------------------------
# Docker helper
# ---------------------------------------------------------------------------

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
        "-v", f"{case_dir.resolve()}:/home/ofuser/run",
        "-w", "/home/ofuser/run",
        image,
        "bash", "-c", f"cd /home/ofuser/run && {command}",
    ])
    log.info("Docker [%s]: %s", image.split("/")[-1].split(":")[0], command[:80])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("Docker failed (rc=%d): %s", result.returncode,
                  result.stderr[-300:] if result.stderr else "")
    return result


# ---------------------------------------------------------------------------
# STL generation
# ---------------------------------------------------------------------------

def generate_stl(site_cfg: dict, mesh_cfg: dict, output_path: Path) -> None:
    """Generate terrain STL from SRTM at specified resolution."""
    from pyproj import Transformer
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    lat0 = site_cfg["site"]["coordinates"]["latitude"]
    lon0 = site_cfg["site"]["coordinates"]["longitude"]
    radius = mesh_cfg["cylinder_radius_m"]
    res = mesh_cfg["stl_resolution_m"]
    domain_m = 2 * radius + 2000  # STL extends beyond cylinder

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    x0, y0 = transformer.transform(lon0, lat0)

    half = domain_m / 2
    nx = ny = int(domain_m / res)

    dst_transform = from_bounds(x0 - half, y0 - half, x0 + half, y0 + half, nx, ny)
    dst_array = np.zeros((ny, nx), dtype=np.float32)

    srtm_path = ROOT / "data" / "raw" / f"srtm_{site_cfg['site'].get('name', 'perdigao')}_30m.tif"
    if not srtm_path.exists():
        # fallback naming
        site_name = [k for k in site_cfg.get("site", {}).keys() if k != "coordinates"][0] if "site" in site_cfg else "perdigao"
        srtm_path = ROOT / "data" / "raw" / f"srtm_perdigao_30m.tif"

    with rasterio.open(srtm_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:32629",
            resampling=Resampling.bilinear,
        )

    log.info("STL terrain: %dx%d grid, elevation %.0f–%.0f m",
             nx, ny, dst_array.min(), dst_array.max())

    # Write STL in local coords (center = 0,0)
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
                    (xs[i+1], ys[j], float(dst_array[j, i+1])),
                    (xs[i], ys[j+1], float(dst_array[j+1, i])),
                    (xs[i+1], ys[j+1], float(dst_array[j+1, i+1])),
                ]
                for tri in [(p[0], p[1], p[2]), (p[1], p[3], p[2])]:
                    f.write("  facet normal 0 0 1\n    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}\n")
                    f.write("    endloop\n  endfacet\n")
                    ntri += 1
        f.write("endsolid terrain\n")

    log.info("STL: %d triangles, %.1f MB → %s",
             ntri, output_path.stat().st_size / 1e6, output_path.name)


# ---------------------------------------------------------------------------
# terrainBlockMesherDict generation
# ---------------------------------------------------------------------------

def write_tbm_dict(mesh_cfg: dict, stl_name: str, case_dir: Path,
                   terrain_z_min: float = 50.0) -> None:
    """Write terrainBlockMesherDict from mesh config."""
    m = mesh_cfg
    half = m["inner_size_m"] / 2
    n_blocks = m["inner_blocks"]
    cpb = m["cells_per_block_xy"]
    eff_res = m["inner_size_m"] / (n_blocks * cpb)

    log.info("Mesh: inner=%dm, %dx%d blocks, %d cells/block → %.0fm effective",
             m["inner_size_m"], n_blocks, n_blocks, cpb, eff_res)
    log.info("Mesh: height=%dm, %d vertical cells, grading=%d",
             m["height_m"], m["cells_z"], m["grading_z"])
    log.info("Mesh: cylinder R=%dm, %d radial cells, %d sections",
             m["cylinder_radius_m"], m["radial_cells"], m["cylinder_sections"])

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

checkMesh           true;
checkMeshNoTopology false;
checkMeshAllGeometry false;
checkMeshAllTopology false;
"""
    system = case_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "terrainBlockMesherDict").write_text(dict_content)

    # Minimal OF system files
    for name, content in [
        ("controlDict", "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
         "application terrainBlockMesher; startFrom latestTime; startTime 0; stopAt endTime;\n"
         "endTime 0; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;\n"
         "writePrecision 10; writeCompression uncompressed; timeFormat general; timePrecision 6;\n"),
        ("fvSchemes", "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"),
        ("fvSolution", "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"),
    ]:
        (system / name).write_text(content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_mesh(case_dir: Path, mesh_cfg: dict) -> None:
    """Run terrainBlockMesher in Docker."""
    image = mesh_cfg.get("docker_image", "terrainblockmesher:of24")
    is_arm = plat.machine() in ("arm64", "aarch64")
    platform = "linux/amd64" if is_arm else None

    poly = case_dir / "constant" / "polyMesh"
    if poly.exists() and (poly / "points").exists():
        log.info("polyMesh already exists, skipping mesher")
        return

    t0 = time.time()
    result = run_docker(case_dir, "terrainBlockMesher",
                        image=image, platform=platform, timeout=1800)

    if result.returncode != 0:
        # Save log for debugging
        (case_dir / "log.terrainBlockMesher").write_text(
            (result.stdout or "") + "\n" + (result.stderr or "")
        )
        raise RuntimeError(f"terrainBlockMesher failed (rc={result.returncode})")

    # Save log
    if result.stdout:
        (case_dir / "log.terrainBlockMesher").write_text(result.stdout)

    log.info("Mesh done in %.0f s", time.time() - t0)

    # Extract stats
    for line in (result.stdout or "").splitlines():
        if "cells:" in line or "non-orthogonality" in line or "Mesh OK" in line:
            log.info("  %s", line.strip())


def run_init(case_dir: Path, inflow_json: Path, solver_cfg: dict) -> None:
    """Render templates + init fields from ERA5."""
    solver_image = solver_cfg.get("image", "microfluidica/openfoam:latest")
    is_arm = plat.machine() in ("arm64", "aarch64")
    platform = "linux/amd64" if is_arm else None

    # writeCellCentres (needed for init_from_era5)
    run_docker(case_dir, "postProcess -func writeCellCentres -time 0",
               image=solver_image, platform=platform)

    # init_from_era5 (local Python)
    dst_inflow = case_dir / "inflow.json"
    if inflow_json.resolve() != dst_inflow.resolve():
        shutil.copy2(inflow_json, dst_inflow)
    init_script_src = Path(__file__).parent / "init_from_era5.py"
    init_script_dst = case_dir / "init_from_era5.py"
    if init_script_src.resolve() != init_script_dst.resolve():
        shutil.copy2(init_script_src, init_script_dst)

    result = subprocess.run(
        [sys.executable, "init_from_era5.py",
         "--case-dir", ".", "--inflow", "inflow.json"],
        cwd=case_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("init_from_era5 failed: %s", result.stderr[-500:])
        raise RuntimeError("init_from_era5 failed")
    log.info("Fields initialized from ERA5")


def run_solve(case_dir: Path, solver_cfg: dict, n_iter: int) -> None:
    """Run simpleFoam."""
    solver_image = solver_cfg.get("image", "microfluidica/openfoam:latest")
    is_arm = plat.machine() in ("arm64", "aarch64")
    platform = "linux/amd64" if is_arm else None

    t0 = time.time()
    result = run_docker(
        case_dir,
        f"{solver_cfg['name']} > log.{solver_cfg['name']} 2>&1",
        image=solver_image, platform=platform,
        timeout=3600,
    )
    wall = time.time() - t0

    if result.stdout:
        (case_dir / f"log.{solver_cfg['name']}").write_text(result.stdout)

    if result.returncode != 0:
        log.error("Solver failed (%.0f s)", wall)
    else:
        log.info("Solver done (%.0f s)", wall)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="terrainBlockMesher → init → solve")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--mesh-only", action="store_true", help="Stop after meshing")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    study = cfg["study"]
    mesh_cfg = cfg["mesh"]
    solver_cfg = cfg.get("solver", {"name": "simpleFoam", "image": "microfluidica/openfoam:latest"})

    # Output directory
    output_dir = args.output or ROOT / "data" / "cases" / study["name"]
    case_dir = output_dir / f"ts_{study['timestamp'].replace(':', '').replace('-', '').replace('T', '_')}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Site config
    site_cfg_path = ROOT / "configs" / "sites" / f"{study['site']}.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    # ---- Step 1: STL ----
    stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
    if not stl_path.exists():
        generate_stl(site_cfg, mesh_cfg, stl_path)
    else:
        log.info("STL exists, skipping generation")

    # Get terrain z_min for p_corner
    terrain_z_min = 50.0  # safe default below most terrain
    with open(stl_path) as f:
        zvals = []
        for line in f:
            if "vertex" in line:
                parts = line.split()
                zvals.append(float(parts[3]))
                if len(zvals) > 10000:
                    break
        if zvals:
            terrain_z_min = min(zvals) - 10
            log.info("Terrain z_min: %.0f m → p_corner.z = %.0f m", min(zvals), terrain_z_min)

    # ---- Step 2: terrainBlockMesherDict ----
    write_tbm_dict(mesh_cfg, "terrain.stl", case_dir, terrain_z_min)

    # ---- Step 3: Prepare inflow (needed for templates) ----
    from prepare_inflow import prepare_inflow
    era5_zarr = ROOT / "data" / "raw" / f"era5_{study['site']}.zarr"
    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        prepare_inflow(
            era5_zarr=era5_zarr,
            timestamp=study["timestamp"],
            site_lat=site_cfg["site"]["coordinates"]["latitude"],
            site_lon=site_cfg["site"]["coordinates"]["longitude"],
            output_json=inflow_json,
        )

    with open(inflow_json) as f:
        inflow = json.load(f)

    n_sections = mesh_cfg.get("cylinder_sections", 8)
    lateral_patches = [f"section_{i}" for i in range(n_sections)]

    template_dir = Path(__file__).parent / "templates" / "openfoam"
    jinja_ctx = {
        "domain": {
            "octagonal": True,
            "lateral_patches": lateral_patches,
            "radius_m": mesh_cfg["cylinder_radius_m"],
            "z_max": mesh_cfg["height_m"],
        },
        "solver": {
            "name": solver_cfg["name"],
            "n_iter": study.get("n_iterations", 300),
            "write_interval": study.get("write_interval", 100),
            "transport_T": solver_cfg.get("transport_T", study.get("transport_T", False)),
            "transport_q": solver_cfg.get("transport_q", study.get("transport_q", False)),
        },
        "physics": {
            "coriolis": solver_cfg.get("coriolis", False),
            "T_ref_K": inflow.get("T_ref", 288.15),
        },
        "canopy": {
            "enabled": solver_cfg.get("canopy", False),
        },
        "inflow": inflow,
    }
    # ---- Step 4: Mesh (writes minimal controlDict internally) ----
    run_mesh(case_dir, mesh_cfg)

    # ---- Step 5: Render solver templates (overwrites TBM's minimal controlDict) ----
    from jinja2 import Environment, FileSystemLoader
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    skip = {"meshDict.j2", "decomposeParDict.j2", "terrainBlockMesherDict.j2"}
    for tmpl_path in sorted(template_dir.rglob("*.j2")):
        if tmpl_path.name in skip:
            continue
        rel = tmpl_path.relative_to(template_dir)
        out_file = case_dir / rel.with_suffix("")
        tmpl = env.get_template(str(rel))
        rendered = tmpl.render(**jinja_ctx)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(rendered)

    # Create .foam for ParaView
    (case_dir / f"{case_dir.name}.foam").touch()

    if args.mesh_only:
        log.info("--mesh-only: stopping here. ParaView: %s/%s.foam",
                 case_dir, case_dir.name)
        return

    # ---- Step 6: Init fields ----
    run_init(case_dir, inflow_json, solver_cfg)

    # ---- Step 7: Solve ----
    run_solve(case_dir, solver_cfg, study.get("n_iterations", 300))

    log.info("Done. ParaView: %s/%s.foam", case_dir, case_dir.name)


if __name__ == "__main__":
    main()
