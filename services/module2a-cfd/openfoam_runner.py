"""
openfoam_runner.py — Docker wrapper for OpenFOAM ESI v2406

Container: opencfd/openfoam:v2406
  - cfMesh cartesianMesh (octree mesher, 2:1 transitions)
  - ABL boundary conditions (inletOutlet, epsilonWallFunction, kqRWallFunction, ...)
  - k-epsilon turbulence model (validated at Perdigão: Letzgus et al. WES 2023)
  - macOS Apple Silicon: requires --platform linux/amd64 (Rosetta 2, ~3-4x slowdown)

Reference: Letzgus et al. (WES 2023) used OpenFOAM v2012 with k-epsilon for Perdigão.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Container configuration
# ---------------------------------------------------------------------------
# microfluidica/openfoam — ESI OpenFOAM v2512 with cfMesh (cartesianMesh) included.
# Contains all ESI solvers + cfMesh utilities (surfaceGenerateBoundingBox, cartesianMesh).
OPENFOAM_IMAGE = "microfluidica/openfoam:latest"

# Source command for ESI OpenFOAM path convention.
OPENFOAM_INIT = (
    "source /usr/lib/openfoam/openfoam2512/etc/bashrc 2>/dev/null || "
    "source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null"
)

# Quality thresholds for checkMesh
MAX_NON_ORTHO = 50.0   # degrees — tightened for cfMesh octree
MAX_SKEWNESS  = 4.0    # relaxed for complex terrain with boundary layers


@dataclass
class MeshQuality:
    """Parsed output of checkMesh."""
    max_non_ortho: float = 0.0
    max_skewness: float = 0.0
    max_aspect_ratio: float = 0.0
    n_cells: int = 0
    ok: bool = True
    raw: str = ""


class OpenFOAMRunner:
    """Run OpenFOAM steps inside the ESI v2406 Docker container.

    Parameters
    ----------
    case_dir:
        Path to the OpenFOAM case directory on the host.
    n_cores:
        Number of MPI ranks for parallel runs.
    platform:
        Docker --platform flag.  Use "linux/amd64" on macOS Apple Silicon
        (Rosetta 2 emulation).
    image:
        Docker image to use (default: opencfd/openfoam:v2406).
    """

    def __init__(
        self,
        case_dir: Path | str,
        *,
        n_cores: int = 8,
        platform: str = "linux/amd64",
        image: str = OPENFOAM_IMAGE,
        timeout: int = 7200,  # 2 h per step
        runtime: str = "docker",  # "docker" or "apptainer"
    ) -> None:
        self.case_dir = Path(case_dir).resolve()
        self.n_cores = n_cores
        self.platform = platform
        self.image = image
        self.timeout = timeout
        self.runtime = runtime

        if not self.case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _docker_run(self, bash_cmd: str) -> subprocess.CompletedProcess:
        """Run *bash_cmd* inside the container; stream output to logger.

        Supports both Docker and Apptainer runtimes (set via self.runtime).
        """
        full_cmd = f"set -o pipefail; {OPENFOAM_INIT} && {bash_cmd}"

        if self.runtime == "apptainer":
            # Apptainer (Singularity successor) — used on HPC clusters.
            # --bind mounts the case dir, --pwd sets working directory,
            # --cleanenv avoids leaking host env into container.
            docker_argv = [
                "apptainer", "exec",
                "--cleanenv",
                "--bind", f"{self.case_dir}:/case",
                "--pwd", "/case",
                self.image,  # .sif file path for Apptainer
                "/bin/bash", "-c", full_cmd,
            ]
        else:
            docker_argv = [
                "docker", "run", "--rm",
                "--platform", self.platform,
                "--entrypoint", "/bin/bash",  # override container ENTRYPOINT
                "-v", f"{self.case_dir}:/case",
                "-w", "/case",
                self.image,
                "-c", full_cmd,
            ]
        logger.debug("%s cmd: %s", self.runtime, " ".join(docker_argv))

        result = subprocess.run(
            docker_argv,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.stdout:
            for line in result.stdout.splitlines():
                logger.debug("[OF] %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.debug("[OF-err] %s", line)

        if result.returncode != 0:
            raise RuntimeError(
                f"OpenFOAM command failed (exit {result.returncode}):\n"
                f"  cmd : {bash_cmd}\n"
                f"  stdout: {result.stdout[-2000:]}\n"
                f"  stderr: {result.stderr[-2000:]}"
            )
        return result

    # ------------------------------------------------------------------
    # Individual mesh/solver steps
    # ------------------------------------------------------------------

    def generate_bounding_box(self) -> None:
        """Generate closed FMS domain surface from terrain STL + bounding box.

        Uses cfMesh's surfaceGenerateBoundingBox utility.
        Reads domain bounds from system/meshDict (not needed as args — cfMesh
        infers them from the STL extent + the z_max from meshDict).
        """
        logger.info("surfaceGenerateBoundingBox — case: %s", self.case_dir)
        # Read domain bounds from meshDict or use the Allrun script
        self._docker_run(
            "cd constant/triSurface && "
            "surfaceGenerateBoundingBox terrain.stl domain.fms "
            "2>&1 | tee ../../log.surfaceGenerateBoundingBox"
        )

    def cartesian_mesh(self) -> None:
        """Generate octree mesh using cfMesh cartesianMesh.

        Reads configuration from system/meshDict.
        Replaces the old blockMesh + snappyHexMesh pipeline.
        """
        logger.info("cartesianMesh — case: %s", self.case_dir)
        self._docker_run("cartesianMesh 2>&1 | tee log.cartesianMesh")

    def check_mesh(self) -> MeshQuality:
        """Run checkMesh and parse quality metrics.

        Returns
        -------
        MeshQuality
            Parsed mesh quality metrics.

        Raises
        ------
        RuntimeError
            If skewness > MAX_SKEWNESS or non-orthogonality > MAX_NON_ORTHO.
        """
        logger.info("checkMesh — case: %s", self.case_dir)
        result = self._docker_run("checkMesh 2>&1 | tee log.checkMesh")
        quality = _parse_check_mesh(result.stdout)

        logger.info(
            "checkMesh: cells=%d, maxNonOrtho=%.1f°, maxSkewness=%.2f, "
            "maxAspectRatio=%.1f",
            quality.n_cells,
            quality.max_non_ortho,
            quality.max_skewness,
            quality.max_aspect_ratio,
        )

        if quality.max_non_ortho > MAX_NON_ORTHO:
            raise RuntimeError(
                f"Mesh non-orthogonality {quality.max_non_ortho:.1f}° "
                f"exceeds threshold {MAX_NON_ORTHO}°"
            )
        if quality.max_skewness > MAX_SKEWNESS:
            raise RuntimeError(
                f"Mesh skewness {quality.max_skewness:.2f} "
                f"exceeds threshold {MAX_SKEWNESS}"
            )
        quality.ok = True
        return quality

    def decompose_par(self, *, force: bool = True) -> None:
        """Decompose the case for parallel running."""
        flag = "-force" if force else ""
        logger.info("decomposePar %s — case: %s", flag, self.case_dir)
        # Symlink boundaryData into processor dirs (MappedFile resolves relative
        # to processor path in OF ESI, not global case root)
        symlink_cmd = (
            ' && if [ -d constant/boundaryData ]; then'
            ' for d in processor*/; do'
            ' ln -sf ../../constant/boundaryData "$d/constant/";'
            ' done; fi'
        )
        self._docker_run(
            f"decomposePar {flag} 2>&1 | tee log.decomposePar{symlink_cmd}"
        )

    def reconstruct_par(self, *, latest_time: bool = True) -> None:
        """Reconstruct the parallel case after the solver."""
        flag = "-latestTime" if latest_time else ""
        logger.info("reconstructPar %s — case: %s", flag, self.case_dir)
        self._docker_run(f"reconstructPar {flag} 2>&1 | tee log.reconstructPar")

    def potential_foam(self, *, parallel: bool = True) -> None:
        """Initialise U with divergence-free potential flow from boundary conditions."""
        if parallel and self.n_cores > 1:
            cmd = (
                f"mpirun --allow-run-as-root -np {self.n_cores} "
                "potentialFoam -parallel 2>&1 | tee log.potentialFoam"
            )
        else:
            cmd = "potentialFoam 2>&1 | tee log.potentialFoam"
        logger.info("potentialFoam (div-free init) — case: %s", self.case_dir)
        self._docker_run(cmd)

    def run_solver(
        self,
        solver: str = "simpleFoam",
        *,
        parallel: bool = True,
    ) -> None:
        """Run the specified OpenFOAM solver."""
        if parallel:
            cmd = (
                f"mpirun --allow-run-as-root -np {self.n_cores} "
                f"{solver} -parallel 2>&1 | tee log.{solver}"
            )
        else:
            cmd = f"{solver} 2>&1 | tee log.{solver}"

        logger.info(
            "%s (%s) — case: %s",
            solver,
            f"parallel {self.n_cores} cores" if parallel else "serial",
            self.case_dir,
        )
        self._docker_run(cmd)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def write_cell_centres(self) -> None:
        """Write cell centre coordinates (0/Cx, 0/Cy, 0/Cz) via postProcess."""
        logger.info("writeCellCentres — case: %s", self.case_dir)
        self._docker_run("postProcess -func writeCellCentres -time 0 2>&1 | tee log.writeCellCentres")

    def run_case(
        self,
        solver: str = "simpleFoam",
        *,
        skip_mesh: bool = False,
        inflow_json: str | Path | None = None,
    ) -> MeshQuality:
        """Run the full OpenFOAM pipeline for one case.

        Steps
        -----
        1. cartesianMesh (cfMesh octree, unless skip_mesh=True)
        2. checkMesh → raise if quality too low
        3. writeCellCentres + init_from_era5 (if inflow_json provided)
        4. decomposePar
        5. <solver> -parallel
        6. reconstructPar

        Parameters
        ----------
        solver:
            Solver binary (default: simpleFoam).
        skip_mesh:
            Skip mesh generation (useful when mesh is already generated).
        inflow_json:
            Path to inflow profile JSON.  If provided, initialises fields
            from ERA5 interpolation (replaces potentialFoam).

        Returns
        -------
        MeshQuality
            checkMesh result (useful for logging/convergence studies).
        """
        logger.info("=== run_case START: %s ===", self.case_dir.name)

        if not skip_mesh:
            self.cartesian_mesh()

        quality = self.check_mesh()

        # Initialise fields from ERA5 (replaces potentialFoam)
        if inflow_json is not None:
            self.write_cell_centres()
            from init_from_era5 import init_from_era5
            init_from_era5(case_dir=self.case_dir, inflow_json=inflow_json)
        else:
            # Fallback: potentialFoam if no inflow data available
            if self.n_cores > 1:
                self.decompose_par()
            self.potential_foam(parallel=(self.n_cores > 1))

        if self.n_cores > 1 and inflow_json is not None:
            # decomposePar after init (fields were written in serial)
            self.decompose_par()

        self.run_solver(solver, parallel=(self.n_cores > 1))

        if self.n_cores > 1:
            self.reconstruct_par()

        logger.info("=== run_case DONE: %s ===", self.case_dir.name)
        return quality


# ---------------------------------------------------------------------------
# checkMesh output parser
# ---------------------------------------------------------------------------

def _parse_check_mesh(output: str) -> MeshQuality:
    """Parse the textual output of checkMesh into a MeshQuality dataclass."""
    q = MeshQuality(raw=output)

    # Number of cells
    m = re.search(r"cells:\s+(\d+)", output)
    if m:
        q.n_cells = int(m.group(1))

    # Max non-orthogonality
    m = re.search(r"Max non-orthogonality\s*=\s*([\d.]+)", output)
    if m:
        q.max_non_ortho = float(m.group(1))

    # Max skewness
    m = re.search(r"Max skewness\s*=\s*([\d.]+)", output)
    if m:
        q.max_skewness = float(m.group(1))

    # Max aspect ratio
    m = re.search(r"Max aspect ratio\s*=\s*([\d.]+)", output)
    if m:
        q.max_aspect_ratio = float(m.group(1))

    return q


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run an OpenFOAM case via Docker (opencfd/openfoam:v2406)"
    )
    parser.add_argument("case_dir", help="Path to OpenFOAM case directory")
    parser.add_argument("--solver", default="simpleFoam")
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--skip-mesh", action="store_true",
                        help="Skip mesh generation (mesh already exists)")
    parser.add_argument("--platform", default="linux/amd64")
    args = parser.parse_args()

    runner = OpenFOAMRunner(
        args.case_dir,
        n_cores=args.n_cores,
        platform=args.platform,
    )
    quality = runner.run_case(solver=args.solver, skip_mesh=args.skip_mesh)
    print(f"cells={quality.n_cells}, maxNonOrtho={quality.max_non_ortho:.1f}°, "
          f"maxSkewness={quality.max_skewness:.2f}")
