"""
generate_mesh_tbm.py — Structured blockMesh via terrainBlockMesher (Docker legacy).

Generates a structured hexahedral mesh over terrain using the terrainBlockMesher
utility (Fraunhofer IWES) running inside an OpenFOAM 2.4.x Docker container.
The polyMesh output is cross-version compatible with OF v2406+.

Usage
-----
    from generate_mesh_tbm import generate_mesh_tbm

    generate_mesh_tbm(
        stl_path=Path("terrain.stl"),
        case_dir=Path("data/cases/my_case"),
        config={
            "origin": [0, 0, 0],
            "p_corner": [4000, 4000, 100],
            "dimensions": [1000, 1000, 500],
            "blocks": [10, 10],
            "cells_per_block": [3, 3, 10],
            "grading": [1, 1, 10],
            "cylinder": {"centre": [4500, 4500, 100], "radius": 3000},
        },
    )
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

log = logging.getLogger("generate_mesh_tbm")

TEMPLATE_DIR = Path(__file__).parent / "templates" / "openfoam"
TBM_IMAGE = "terrainblockmesher:of24"


def _render_tbm_dict(config: dict, stl_filename: str, output_path: Path) -> None:
    """Render terrainBlockMesherDict from Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR / "system")),
        keep_trailing_newline=True,
    )
    template = env.get_template("terrainBlockMesherDict.j2")

    ctx = {
        "stl_file": stl_filename,
        "origin": config.get("origin", [0, 0, 0]),
        "p_corner": config["p_corner"],
        "dimensions": config["dimensions"],
        "p_above": config.get("p_above", [0, 0, 10000]),
        "blocks": config.get("blocks", [10, 10]),
        "cells_per_block": config.get("cells_per_block", [3, 3, 10]),
        "max_dist_proj": config.get("max_dist_proj", 20000),
        "grading": config.get("grading", [1, 1, 10]),
        "check_mesh": config.get("check_mesh", "true"),
    }

    # Cylinder section (optional)
    if "cylinder" in config:
        ctx["cylinder"] = config["cylinder"]

    # Orthogonal splines (optional)
    if "orthogonal_splines" in config:
        ctx["orthogonal_splines"] = config["orthogonal_splines"]

    rendered = template.render(**ctx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    log.info("Rendered terrainBlockMesherDict → %s", output_path)


def _write_minimal_of_case(case_dir: Path) -> None:
    """Write minimal system/ files required by terrainBlockMesher."""
    system = case_dir / "system"
    system.mkdir(parents=True, exist_ok=True)

    # controlDict (minimal, required by OF runtime)
    (system / "controlDict").write_text("""\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     terrainBlockMesher;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         0;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  10;
writeCompression uncompressed;
timeFormat      general;
timePrecision   6;
runSubCycles    1;
""")

    # fvSchemes (minimal)
    (system / "fvSchemes").write_text("""\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
""")

    # fvSolution (minimal)
    (system / "fvSolution").write_text("""\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
""")


def _run_tbm_docker(
    case_dir: Path,
    image: str = TBM_IMAGE,
    timeout: int = 600,
    platform: str | None = None,
) -> subprocess.CompletedProcess:
    """Run terrainBlockMesher in Docker container."""
    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend([
        "-v", f"{case_dir.resolve()}:/case",
        "-w", "/case",
        image,
        "terrainBlockMesher",
    ])
    log.info("Running terrainBlockMesher in Docker [%s]", image)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("terrainBlockMesher failed:\n%s",
                  result.stderr[-500:] if result.stderr else result.stdout[-500:])
    else:
        log.info("terrainBlockMesher completed successfully")
    return result


def generate_mesh_tbm(
    stl_path: Path,
    case_dir: Path,
    config: dict,
    docker_image: str = TBM_IMAGE,
    timeout: int = 600,
    platform: str | None = None,
    keep_tmp: bool = False,
) -> Path:
    """Generate structured blockMesh via terrainBlockMesher Docker container.

    Parameters
    ----------
    stl_path : Path
        Path to terrain STL file.
    case_dir : Path
        Target OpenFOAM case directory (polyMesh will be placed here).
    config : dict
        terrainBlockMesher parameters. Required keys:
        - p_corner: [x, y, z] lower-left corner
        - dimensions: [dx, dy, dz] domain size
        Optional keys:
        - origin, p_above, blocks, cells_per_block, grading
        - cylinder: {centre, radius, radial_grading, radial_cells, ...}
        - orthogonal_splines: {normal_dist}
    docker_image : str
        Docker image with terrainBlockMesher compiled.
    timeout : int
        Max seconds for meshing.
    platform : str | None
        Docker --platform flag (e.g., "linux/amd64" for ARM emulation).
    keep_tmp : bool
        Keep temporary meshing directory for debugging.

    Returns
    -------
    Path
        Path to case_dir with polyMesh installed.
    """
    stl_path = Path(stl_path).resolve()
    case_dir = Path(case_dir).resolve()

    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    stl_filename = stl_path.name

    # Work in a temporary directory to avoid polluting the case
    with tempfile.TemporaryDirectory(prefix="tbm_") as tmp:
        tmp_case = Path(tmp) / "tbm_case"
        tmp_case.mkdir()

        # Setup case structure
        _write_minimal_of_case(tmp_case)
        _render_tbm_dict(config, stl_filename, tmp_case / "system" / "terrainBlockMesherDict")

        # Copy STL
        tri_dir = tmp_case / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_path, tri_dir / stl_filename)

        # Run terrainBlockMesher
        result = _run_tbm_docker(tmp_case, image=docker_image,
                                 timeout=timeout, platform=platform)
        if result.returncode != 0:
            if keep_tmp:
                debug_dir = case_dir.parent / f"{case_dir.name}_tbm_debug"
                shutil.copytree(tmp_case, debug_dir, dirs_exist_ok=True)
                log.error("Debug case saved: %s", debug_dir)
            raise RuntimeError(
                f"terrainBlockMesher failed (rc={result.returncode}). "
                f"stderr: {result.stderr[-300:] if result.stderr else 'empty'}"
            )

        # Verify polyMesh was created
        poly_mesh_src = tmp_case / "constant" / "polyMesh"
        if not (poly_mesh_src / "points").exists():
            # terrainBlockMesher may write blockMeshDict only — check
            bmdict = tmp_case / "constant" / "polyMesh" / "blockMeshDict"
            if not bmdict.exists():
                bmdict = tmp_case / "constant" / "blockMeshDict"
            if bmdict.exists():
                log.warning("terrainBlockMesher produced blockMeshDict but no polyMesh. "
                            "You may need to run blockMesh separately.")
            raise RuntimeError("polyMesh/points not found after terrainBlockMesher")

        # Copy polyMesh to target case
        poly_mesh_dst = case_dir / "constant" / "polyMesh"
        poly_mesh_dst.mkdir(parents=True, exist_ok=True)
        if poly_mesh_dst.exists():
            shutil.rmtree(poly_mesh_dst)
        shutil.copytree(poly_mesh_src, poly_mesh_dst)
        log.info("polyMesh copied → %s", poly_mesh_dst)

        # Also copy blockMeshDict if produced (useful for debugging)
        for bmd_candidate in [
            tmp_case / "constant" / "polyMesh" / "blockMeshDict",
            tmp_case / "constant" / "blockMeshDict",
        ]:
            if bmd_candidate.exists():
                shutil.copy2(bmd_candidate, case_dir / "constant" / "blockMeshDict")
                log.info("blockMeshDict copied for reference")
                break

    return case_dir


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate mesh via terrainBlockMesher")
    parser.add_argument("--stl", required=True, type=Path, help="Terrain STL file")
    parser.add_argument("--case", required=True, type=Path, help="Target case dir")
    parser.add_argument("--config", required=True, type=Path, help="YAML config for TBM params")
    parser.add_argument("--image", default=TBM_IMAGE, help="Docker image")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp dir on failure")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_mesh_tbm(
        stl_path=args.stl,
        case_dir=args.case,
        config=cfg.get("terrainBlockMesher", cfg),
        docker_image=args.image,
        keep_tmp=args.keep_tmp,
    )
