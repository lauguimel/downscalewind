"""
test_mesh_variant.py — Single-site mesh variant test.

Generates mesh + sets up case + solves for 1 site/timestamp with custom mesh_cfg.
Used to compare wall time + convergence between mesh resolutions.

Usage (on Aqua):
    python3 test_mesh_variant.py \\
        --site-id ct_c_morpho_0023 \\
        --lat 46.62386 --lon 11.8493 \\
        --timestamp 2017-06-17T12:00:00 \\
        --output-dir /scratch/maitreje/dsw/mesh_test/variant_B \\
        --inner-size 2000 --inner-blocks 10 \\
        --cells-z 80 --grading-z 30 \\
        --n-iter 1000 --n-cores 24
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Import from main campaign script
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_multisite_campaign import (
    DEFAULT_MESH, extract_stl, _read_stl_zmin, write_tbm_dict,
    run_tbm_mesh, run_write_cell_centres, solve_case,
    setup_case, _container_cmd, OF_IMAGE, TBM_IMAGE, PYTHON, log,
)
import run_multisite_campaign as RMC
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site-id", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--timestamp", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--era5-zarr", type=Path, required=True)
    p.add_argument("--srtm", type=Path, required=True)
    # Mesh overrides
    p.add_argument("--inner-size", type=float)
    p.add_argument("--inner-blocks", type=int)
    p.add_argument("--cells-per-block-xy", type=int)
    p.add_argument("--cylinder-radius", type=float)
    p.add_argument("--cells-z", type=int)
    p.add_argument("--grading-z", type=float)
    # Solve
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--n-cores", type=int, default=24)
    p.add_argument("--runtime", default="apptainer")
    p.add_argument("--of-image", default=None)
    p.add_argument("--tbm-image", default=None)
    # Solver overrides (post-template patching)
    p.add_argument("--p-solver", choices=["GAMG", "PCG"], default=None,
                   help="Override pressure solver")
    p.add_argument("--n-non-ortho", type=int, default=None,
                   help="Override SIMPLE.nNonOrthogonalCorrectors")
    p.add_argument("--simple-mode", choices=["consistent", "strict"], default=None,
                   help="consistent=SIMPLEC (relax U=0.7); strict=SIMPLE (relax U=0.3 like 9k)")
    args = p.parse_args()

    # Apply runtime
    RMC.CONTAINER_RUNTIME = args.runtime
    if args.of_image:
        RMC.OF_IMAGE = args.of_image
    if args.tbm_image:
        RMC.TBM_IMAGE = args.tbm_image
    RMC.GRID_EXPORT_ENABLED = False  # skip export for tests
    # Reload local references after monkey-patch
    global OF_IMAGE, TBM_IMAGE
    OF_IMAGE = RMC.OF_IMAGE
    TBM_IMAGE = RMC.TBM_IMAGE

    # Build mesh_cfg
    mesh_cfg = dict(DEFAULT_MESH)
    for key, val in [
        ("inner_size_m", args.inner_size),
        ("inner_blocks", args.inner_blocks),
        ("cells_per_block_xy", args.cells_per_block_xy),
        ("cylinder_radius_m", args.cylinder_radius),
        ("cells_z", args.cells_z),
        ("grading_z", args.grading_z),
    ]:
        if val is not None:
            mesh_cfg[key] = val

    log.info("Mesh config: %s", {k: mesh_cfg[k] for k in
        ("inner_size_m", "inner_blocks", "cells_per_block_xy",
         "cylinder_radius_m", "cells_z", "grading_z")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    site_dir = args.output_dir
    mesh_dir = site_dir / "mesh"
    case_dir = site_dir / "case_test"

    # ── Step 1: STL ──
    log.info("=== STL extraction ===")
    t0 = time.time()
    stl_path = mesh_dir / "constant" / "triSurface" / "terrain.stl"
    if not stl_path.exists():
        terrain_z_min = extract_stl(
            args.srtm, args.lat, args.lon,
            mesh_cfg["cylinder_radius_m"],
            mesh_cfg["stl_resolution_m"], stl_path)
    else:
        terrain_z_min = _read_stl_zmin(stl_path)
    log.info("STL done in %.1f s, z_min=%.1f", time.time() - t0, terrain_z_min)

    # ── Step 2: TBM mesh (uses run_tbm_mesh with proper env sourcing) ──
    log.info("=== terrainBlockMesher ===")
    t0 = time.time()
    write_tbm_dict(mesh_cfg, "terrain.stl", mesh_dir, terrain_z_min)
    if not run_tbm_mesh(mesh_dir, mesh_cfg):
        log.error("TBM failed")
        sys.exit(1)
    log.info("TBM done in %.1f s", time.time() - t0)

    # writeCellCentres (uses the proper helper)
    log.info("=== writeCellCentres ===")
    if not run_write_cell_centres(mesh_dir):
        log.error("writeCellCentres failed")
        sys.exit(1)

    # checkMesh
    cmd = _container_cmd(OF_IMAGE, mesh_dir,
        ["bash", "-c", "cd /home/ofuser/run && checkMesh -allTopology -allGeometry 2>&1 | grep -E 'cells:|Max non|Max skew|Max aspect|Mesh OK|Failed|small determinant|Concave'"])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log.info("=== checkMesh ===\n%s", r.stdout)

    # ── Step 3: Setup case ──
    log.info("=== setup_case ===")
    t0 = time.time()
    ok = setup_case(case_dir, mesh_dir, args.era5_zarr, args.timestamp,
                    args.lat, args.lon, mesh_cfg, args.n_iter)
    log.info("setup_case %s in %.1f s", "OK" if ok else "FAILED", time.time() - t0)
    if not ok:
        sys.exit(1)

    # ── Step 3.5: Solver overrides (post-template patching) ──
    fvsol = case_dir / "system" / "fvSolution"
    if fvsol.exists() and (args.p_solver or args.n_non_ortho is not None or args.simple_mode):
        txt = fvsol.read_text()
        import re
        if args.p_solver == "PCG":
            # Replace GAMG p block with PCG
            pcg_p = """    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
        maxIter         200;
    }"""
            txt = re.sub(r"    p\s*\{[^}]*?GAMG[^}]*?\}", pcg_p, txt, flags=re.DOTALL, count=1)
            log.info("Patched p solver to PCG")
        if args.n_non_ortho is not None:
            txt = re.sub(r"nNonOrthogonalCorrectors\s+\d+;",
                         f"nNonOrthogonalCorrectors {args.n_non_ortho};", txt, count=2)
            log.info("Patched nNonOrth = %d", args.n_non_ortho)
        if args.simple_mode == "strict":
            # Remove `consistent yes;` (so SIMPLE not SIMPLEC) and tighten relax
            txt = txt.replace("consistent      yes;", "consistent      no;")
            # Accept both legacy U=0.7 and intermediate U=0.5 base templates
            txt = re.sub(r"U\s+0\.[57];", "U               0.3;", txt)
            txt = re.sub(r"k\s+0\.[57];", "k               0.5;", txt)
            txt = re.sub(r"epsilon\s+0\.[57];", "epsilon         0.5;", txt)
            log.info("Patched to SIMPLE strict (consistent=no, relax 0.3/0.5)")
        fvsol.write_text(txt)

    # ── Step 4: Solve ──
    log.info("=== Solve simpleFoam (n_iter=%d, n_cores=%d) ===", args.n_iter, args.n_cores)
    t0 = time.time()
    ok = solve_case(case_dir, args.n_iter, args.n_cores)
    wall = time.time() - t0
    log.info("Solve done in %.1f s (ok=%s)", wall, ok)

    # Extract U_max trajectory
    fmm = case_dir / "postProcessing" / "fieldMinMax" / "0" / "fieldMinMax.dat"
    if fmm.exists():
        import re
        umax_traj = []
        for line in open(fmm):
            if "mag(U)" in line:
                m = re.match(r"^\s*(\d+)\s+mag\(U\)\s+\S+\s+\(\S+\s+\S+\s+\S+\)\s+\S+\s+(\S+)", line)
                if m:
                    umax_traj.append((int(m.group(1)), float(m.group(2))))
        if umax_traj:
            log.info("U_max trajectory (every 100 iter):")
            for i, u in umax_traj:
                if i % 100 == 0 or i < 3 or i == umax_traj[-1][0]:
                    log.info("  iter=%d U_max=%.3f", i, u)
            final_iter, final_umax = umax_traj[-1]
            summary = {
                "wall_time_s": wall, "rc": r.returncode,
                "final_iter": final_iter, "final_U_max": final_umax,
                "mesh_cfg": mesh_cfg,
            }
            (site_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            log.info("=== SUMMARY: wall=%.0fs U_max(@%d)=%.2f ===",
                     wall, final_iter, final_umax)


if __name__ == "__main__":
    main()
