"""
openfoam_runner.py — Docker wrapper for buoyantFoam (ESI OpenCFD v2412)

Container: opencfd/openfoam-default:v2412
  - ABL boundary conditions (atmBoundaryLayerInletVelocity, atmOmegaWallFunction, ...)
  - source /opt/openfoam*/etc/bashrc  (ESI path convention)
  - macOS Apple Silicon: requires --platform linux/amd64 (Rosetta 2, ~3-4x slowdown)

Reference: Neunaber et al. (WES 2023) used OpenFOAM v2012 (ESI branch) for Perdigão.
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
# Foundation OpenFOAM v10 (openfoam.org) — publicly available on Docker Hub.
# Supports all ABL BCs: atmBoundaryLayerInletVelocity, atmOmegaWallFunction, etc.
# Note: original plan used ESI opencfd/openfoam-default:v2412 but it is not on Docker Hub.
# Foundation v10 is equivalent for buoyantFoam + ABL BCs.
OPENFOAM_IMAGE = "openfoam/openfoam10-paraview510"

# Source command for Foundation OpenFOAM v10 path convention.
OPENFOAM_INIT = (
    "source /opt/openfoam10/etc/bashrc 2>/dev/null || "
    "source /opt/openfoam*/etc/bashrc 2>/dev/null || "
    "source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null"
)

# Quality thresholds for checkMesh
MAX_NON_ORTHO = 70.0   # degrees
MAX_SKEWNESS  = 10.0  # relaxed for complex terrain with addLayers


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
    """Run buoyantFoam steps inside the ESI OpenCFD Docker container.

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
        Docker image to use (default: opencfd/openfoam-default:v2412).
    """

    def __init__(
        self,
        case_dir: Path | str,
        *,
        n_cores: int = 8,
        platform: str = "linux/amd64",
        image: str = OPENFOAM_IMAGE,
        timeout: int = 7200,  # 2 h per step
    ) -> None:
        self.case_dir = Path(case_dir).resolve()
        self.n_cores = n_cores
        self.platform = platform
        self.image = image
        self.timeout = timeout

        if not self.case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _docker_run(self, bash_cmd: str) -> subprocess.CompletedProcess:
        """Run *bash_cmd* inside the container; stream output to logger."""
        full_cmd = f"set -o pipefail; {OPENFOAM_INIT} && {bash_cmd}"
        docker_argv = [
            "docker", "run", "--rm",
            "--platform", self.platform,
            "--entrypoint", "/bin/bash",  # override container ENTRYPOINT
            "-v", f"{self.case_dir}:/case",
            "-w", "/case",
            self.image,
            "-c", full_cmd,
        ]
        logger.debug("docker cmd: %s", " ".join(docker_argv))

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

    def block_mesh(self) -> None:
        """Generate the base Cartesian block mesh."""
        logger.info("blockMesh — case: %s", self.case_dir)
        self._docker_run("blockMesh 2>&1 | tee log.blockMesh")

    def surface_feature_extract(self) -> None:
        """Extract feature edges from terrain STL (required by snappyHexMesh).

        OF v10 uses 'surfaceFeatures' (replaces old 'surfaceFeatureExtract').
        """
        logger.info("surfaceFeatures — case: %s", self.case_dir)
        self._docker_run("surfaceFeatures 2>&1 | tee log.surfaceFeatures")

    def snappy_hex_mesh(self, *, overwrite: bool = True) -> None:
        """Run surfaceFeatureExtract + snappyHexMesh (terrain STL + refinement regions)."""
        self.surface_feature_extract()
        flag = "-overwrite" if overwrite else ""
        logger.info("snappyHexMesh %s — case: %s", flag, self.case_dir)
        self._docker_run(f"snappyHexMesh {flag} 2>&1 | tee log.snappyHexMesh")

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
        self._docker_run(f"decomposePar {flag} 2>&1 | tee log.decomposePar")

    def reconstruct_par(self, *, latest_time: bool = True) -> None:
        """Reconstruct the parallel case after the solver."""
        flag = "-latestTime" if latest_time else ""
        logger.info("reconstructPar %s — case: %s", flag, self.case_dir)
        self._docker_run(f"reconstructPar {flag} 2>&1 | tee log.reconstructPar")

    def potential_foam(self, *, parallel: bool = True) -> None:
        """Initialise U with divergence-free potential flow from boundary conditions.

        potentialFoam solves Laplace(phi) = 0 with the U BCs mapped to Neumann
        conditions on phi, then sets U = grad(phi).  The result satisfies:
          - U·n matches specified inflow at lateral faces
          - div(U) = 0 everywhere (divergence-free)
          - U·n = 0 at walls (no-penetration)

        This gives simpleFoam a physically consistent starting point.
        """
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
        """Run the specified OpenFOAM solver.

        Parameters
        ----------
        solver:
            Solver binary name (default: simpleFoam).
        parallel:
            If True, run with mpirun using self.n_cores.
        """
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
        skip_snappy: bool = False,
        inflow_json: str | Path | None = None,
    ) -> MeshQuality:
        """Run the full OpenFOAM pipeline for one case.

        Steps
        -----
        1. blockMesh
        2. snappyHexMesh (unless skip_snappy=True)
        3. checkMesh → raise if quality too low
        4. writeCellCentres + init_from_era5 (if inflow_json provided)
        5. decomposePar
        6. <solver> -parallel
        7. reconstructPar

        Parameters
        ----------
        solver:
            Solver binary (default: simpleFoam).
        skip_snappy:
            Skip snappyHexMesh (useful for pipeline test with 1×1 domain,
            blockMesh-only).
        inflow_json:
            Path to inflow profile JSON.  If provided, initialises fields
            from ERA5 interpolation (replaces potentialFoam).

        Returns
        -------
        MeshQuality
            checkMesh result (useful for logging/convergence studies).
        """
        logger.info("=== run_case START: %s ===", self.case_dir.name)

        self.block_mesh()

        if not skip_snappy:
            self.snappy_hex_mesh()

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
        description="Run an OpenFOAM case via Docker (opencfd/openfoam-default:v2412)"
    )
    parser.add_argument("case_dir", help="Path to OpenFOAM case directory")
    parser.add_argument("--solver", default="simpleFoam")
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--skip-snappy", action="store_true",
                        help="Skip snappyHexMesh (blockMesh-only test)")
    parser.add_argument("--platform", default="linux/amd64")
    args = parser.parse_args()

    runner = OpenFOAMRunner(
        args.case_dir,
        n_cores=args.n_cores,
        platform=args.platform,
    )
    quality = runner.run_case(solver=args.solver, skip_snappy=args.skip_snappy)
    print(f"cells={quality.n_cells}, maxNonOrtho={quality.max_non_ortho:.1f}°, "
          f"maxSkewness={quality.max_skewness:.2f}")
