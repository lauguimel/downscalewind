"""
run_cascade_study.py — Multi-resolution cascade CFD via hpc-foam.

Strategy: coarse simpleFoam → mapFields → fine buoyantSimpleFoam.
Each level is initialised from the converged solution of the previous level,
reducing convergence iterations by 2-3×.

Cascade levels (default):
  500m  simpleFoam         — fast RANS, ~12 min
  250m  simpleFoam         — init from 500m, ~20 min
  100m  buoyantSimpleFoam  — init from 250m (p×ρ + T), ~30 min
   40m  buoyantSimpleFoam  — init from 100m, ~2h

PBS chain dependencies: each level waits for the previous to complete
(afterok).  All levels can be submitted in one shot; the HPC scheduler
handles sequencing.

Uses hpc-foam primitives (CaseDatabase, CaseBatch, TransferManager) for
persistent tracking and PBS submission.

Usage
-----
    # Full cascade submit (generates cases locally, uploads, submits with deps)
    python run_cascade_study.py submit \\
        --inflow data/inflow_profiles/2017-05-11T12.json \\
        --hpc    configs/hpc/aqua.yaml \\
        --resolutions 500 250 100 40 \\
        --solvers simpleFoam simpleFoam buoyantSimpleFoam buoyantSimpleFoam

    # Timing experiment: 3 identical runs to measure real HPC wall time
    python run_cascade_study.py timing-test \\
        --inflow data/inflow_profiles/2017-05-11T12.json \\
        --hpc    configs/hpc/aqua.yaml \\
        --resolution 500 --solver simpleFoam --n-replicas 3

    # Monitor submitted jobs
    python run_cascade_study.py monitor --hpc configs/hpc/aqua.yaml

    # Download results for a level + prepare next level IC
    python run_cascade_study.py next-level \\
        --hpc configs/hpc/aqua.yaml --from-res 250 --to-res 100

    # List all cascade cases in the DB
    python run_cascade_study.py list
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "data" / "cascade" / "cases"
DB_PATH = PROJECT_ROOT / "data" / "cascade" / "cascade.db"


# ---------------------------------------------------------------------------
# Level specification
# ---------------------------------------------------------------------------

@dataclass
class CascadeLevel:
    """One resolution level in the cascade."""
    level_id: str            # e.g. "cascade_perdigao_500m"
    resolution_m: float
    solver_name: str         # "simpleFoam" or "buoyantSimpleFoam"
    thermal: bool
    case_dir: Path           # local case directory


# ---------------------------------------------------------------------------
# Case generation helpers
# ---------------------------------------------------------------------------

def _load_site_cfg(site: str = "perdigao") -> dict:
    path = PROJECT_ROOT / "configs" / "sites" / f"{site}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_level_case(
    level_id: str,
    resolution_m: float,
    solver_name: str,
    thermal: bool,
    inflow_json: Path,
    site: str = "perdigao",
    of_version: int = 9,
    n_iterations: int = 2000,
    write_interval: int = 200,
) -> Path:
    """Generate a single cascade level case directory."""
    from generate_mesh import generate_mesh

    case_dir = CASES_DIR / level_id
    if case_dir.exists():
        logger.info("Case %s already exists, skipping generation", level_id)
        return case_dir

    site_cfg = _load_site_cfg(site)
    srtm_tif = PROJECT_ROOT / "data" / "raw" / f"srtm_{site}_30m.tif"

    logger.info(
        "Generating %s: res=%gm, solver=%s, thermal=%s",
        level_id, resolution_m, solver_name, thermal,
    )
    generate_mesh(
        site_cfg=site_cfg,
        resolution_m=resolution_m,
        output_dir=case_dir,
        srtm_tif=srtm_tif if srtm_tif.exists() else None,
        inflow_json=inflow_json,
        solver_name=solver_name,
        thermal=thermal,
    )
    logger.info("Case ready: %s", case_dir)
    return case_dir


# ---------------------------------------------------------------------------
# mapFields helpers
# ---------------------------------------------------------------------------

def _mapfields_local(source_case: Path, target_case: Path) -> None:
    """Run OpenFOAM mapFields locally via Docker (same-solver levels only)."""
    logger.info("mapFields: %s → %s", source_case.name, target_case.name)
    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{source_case}:/source",
        "-v", f"{target_case}:/target",
        "opencfd/openfoam:v2406",
        "bash", "-c",
        "source /usr/lib/openfoam/openfoam2406/etc/bashrc && "
        "mapFields /source -case /target -sourceTime latestTime -consistent -noFunctionObjects",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("mapFields stderr: %s", result.stderr[-500:])
    else:
        logger.info("mapFields completed successfully")


# ---------------------------------------------------------------------------
# hpc-foam config builder
# ---------------------------------------------------------------------------

def _build_hpc_config(hpc_cfg: dict) -> "HPCConfig":
    from hpc_foam import HPCConfig
    conn = hpc_cfg["connection"]
    res = hpc_cfg["resources"]
    container = hpc_cfg.get("container", {})
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    return HPCConfig(
        hpc_host=conn["hpc_host"],
        username=conn["username"],
        remote_base_dir=Path(conn["remote_base_dir"]),
        local_base_dir=CASES_DIR,
        solver="simpleFoam",
        default_ncpus=res["default_ncpus"],
        default_mem=res["default_mem"],
        default_walltime=res["default_walltime"],
        apptainer_wrapper=container.get("apptainer_wrapper", ""),
    )


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

class CascadeStudy:
    """Manages a multi-resolution cascade of CFD simulations via hpc-foam."""

    def __init__(self, hpc_config_path: Path):
        with open(hpc_config_path) as f:
            self.hpc_cfg = yaml.safe_load(f)
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.config = _build_hpc_config(self.hpc_cfg)

        from hpc_foam import CaseDatabase, TransferManager, CaseBatch
        self.db = CaseDatabase(DB_PATH)
        self.transfer = TransferManager(self.config, db=self.db)
        self.batch = CaseBatch(self.config, db=self.db)

    # ------------------------------------------------------------------
    # submit — full cascade with PBS dependencies via CaseBatch.submit_cascade
    # ------------------------------------------------------------------

    def submit(
        self,
        inflow_json: Path,
        resolutions: list[float],
        solver_names: list[str],
        site: str = "perdigao",
        of_version: int = 9,
        n_iterations: int = 2000,
        write_interval: int = 200,
    ) -> list:
        """Generate, upload, and submit all levels with PBS afterok chain."""
        from hpc_foam import CaseState

        assert len(resolutions) == len(solver_names)

        # Build a shared cascade group ID from first level params
        inflow_stem = inflow_json.stem.replace(":", "").replace("-", "")
        cascade_group_id = f"cascade_{site}_{inflow_stem}"
        remote_base = self.config.remote_base_dir

        levels: list[CascadeLevel] = []
        case_dirs: list[Path] = []

        # 1. Generate all cases locally and register in DB
        for resolution_m, solver_name in zip(resolutions, solver_names):
            thermal = solver_name != "simpleFoam"
            level_id = f"cascade_{site}_{int(resolution_m)}m"

            case_dir = _generate_level_case(
                level_id=level_id,
                resolution_m=resolution_m,
                solver_name=solver_name,
                thermal=thermal,
                inflow_json=inflow_json,
                site=site,
                of_version=of_version,
                n_iterations=n_iterations,
                write_interval=write_interval,
            )

            level = CascadeLevel(
                level_id=level_id,
                resolution_m=resolution_m,
                solver_name=solver_name,
                thermal=thermal,
                case_dir=case_dir,
            )
            levels.append(level)
            case_dirs.append(case_dir)

            # Register in DB (idempotent via INSERT OR IGNORE)
            if not self.db.get_case(level_id):
                self.db.add_case(
                    level_id,
                    group_name=cascade_group_id,
                    parameters={
                        "resolution_m": resolution_m,
                        "solver": solver_name,
                        "thermal": thermal,
                        "site": site,
                    },
                    local_dir=str(case_dir),
                    remote_dir=str(remote_base / level_id),
                    solver=solver_name,
                )

        # 2. Upload all cases to HPC
        for case_dir in case_dirs:
            logger.info("Uploading %s …", case_dir.name)
            ok = self.transfer.upload_case(case_dir)
            if not ok:
                raise RuntimeError(f"Upload failed for {case_dir.name}")

        # 3. Build CaseBatch in cascade order and submit with afterok chain
        from hpc_foam import CaseBatch
        cascade_batch = CaseBatch(self.config, db=self.db)
        for level in levels:
            cascade_batch.add_case(remote_base / level.level_id, case_id=level.level_id)

        jobs = cascade_batch.submit_cascade(cascade_group_id=cascade_group_id)

        logger.info(
            "Cascade submitted: %d levels, job chain %s",
            len(levels), " → ".join(j.job_id for j in jobs),
        )
        return jobs

    # ------------------------------------------------------------------
    # timing-test — N replicas of the same case in parallel
    # ------------------------------------------------------------------

    def timing_test(
        self,
        inflow_json: Path,
        resolution_m: float,
        solver_name: str,
        n_replicas: int = 3,
        site: str = "perdigao",
        of_version: int = 9,
        n_iterations: int = 1000,
        write_interval: int = 100,
    ) -> list:
        """Submit N identical cases to measure real HPC wall time variability."""
        from hpc_foam import CaseBatch

        thermal = solver_name != "simpleFoam"
        remote_base = self.config.remote_base_dir
        timing_group = f"timing_{site}_{int(resolution_m)}m"

        timing_batch = CaseBatch(self.config, db=self.db)

        for i in range(1, n_replicas + 1):
            level_id = f"timing_{site}_{int(resolution_m)}m_r{i:02d}"

            case_dir = _generate_level_case(
                level_id=level_id,
                resolution_m=resolution_m,
                solver_name=solver_name,
                thermal=thermal,
                inflow_json=inflow_json,
                site=site,
                of_version=of_version,
                n_iterations=n_iterations,
                write_interval=write_interval,
            )

            if not self.db.get_case(level_id):
                self.db.add_case(
                    level_id,
                    group_name=timing_group,
                    parameters={
                        "resolution_m": resolution_m,
                        "solver": solver_name,
                        "thermal": thermal,
                        "replica": i,
                        "type": "timing",
                    },
                    local_dir=str(case_dir),
                    remote_dir=str(remote_base / level_id),
                    solver=solver_name,
                )

            logger.info("Uploading timing replica %d/%d: %s …", i, n_replicas, level_id)
            ok = self.transfer.upload_case(case_dir)
            if not ok:
                raise RuntimeError(f"Upload failed for {level_id}")

            timing_batch.add_case(remote_base / level_id, case_id=level_id)

        # All replicas run in parallel (no dependencies)
        jobs = timing_batch.submit_all()

        job_ids = [j.job_id for j in jobs]
        logger.info(
            "Timing test submitted: %d replicas of %gm %s — jobs: %s",
            n_replicas, resolution_m, solver_name, ", ".join(job_ids),
        )
        return jobs

    # ------------------------------------------------------------------
    # monitor
    # ------------------------------------------------------------------

    def monitor(self) -> None:
        """Print status of all cascade and timing jobs."""
        from hpc_foam import SSHExecutor

        # Show DB summary
        from hpc_foam import CaseState
        cases = self.db.list_cases()
        if not cases:
            print("No cases in database.")
            return

        print(f"\n{'Case ID':<45} {'Solver':<25} {'State':<15} {'Job ID':<15}")
        print("-" * 105)
        for c in sorted(cases, key=lambda x: x["case_id"]):
            print(
                f"{c['case_id']:<45} "
                f"{c.get('solver') or '—':<25} "
                f"{c['state']:<15} "
                f"{c.get('job_id') or '—':<15}"
            )

        # Live qstat
        ssh = SSHExecutor.from_config(self.config)
        result = ssh.run(
            f"qstat -u {self.hpc_cfg['connection']['username']} 2>/dev/null || echo '(no jobs)'",
            timeout=30,
        )
        print("\n--- qstat ---")
        print(result.stdout or "(no output)")

        # Cascade chains
        groups = self.db.list_cascade_groups()
        if groups:
            print("\n--- Cascade chains ---")
            for gid in groups:
                chain = self.db.get_cascade_chain(gid)
                print(f"\n{gid}:")
                for c in chain:
                    print(f"  L{c['cascade_level']}: {c['case_id']:<40} {c['state']}")

    # ------------------------------------------------------------------
    # next-level — download + convert IC for next resolution
    # ------------------------------------------------------------------

    def prepare_next_level(
        self,
        from_res: float,
        to_res: float,
        site: str = "perdigao",
        inflow_json: Path | None = None,
    ) -> None:
        """Download converged from_res case and set up to_res initial conditions.

        Pipeline:
          1. Download from_res case from HPC via TransferManager
          2. Run convert_fields if solver changes (incompressible → compressible)
          3. Run mapFields locally (Docker)
          4. The to_res case must already be generated locally
        """
        from convert_fields import convert_fields as _convert_fields

        from_id = f"cascade_{site}_{int(from_res)}m"
        to_id = f"cascade_{site}_{int(to_res)}m"
        from_case = CASES_DIR / from_id
        to_case = CASES_DIR / to_id

        if not to_case.exists():
            raise FileNotFoundError(f"Target case not found locally: {to_case}. Run generate first.")

        # 1. Download from HPC
        logger.info("Downloading %s from HPC …", from_id)
        ok = self.transfer.download_case(from_id, from_case)
        if not ok:
            raise RuntimeError(f"Download failed for {from_id}")
        logger.info("Downloaded %s", from_id)

        # 2. Field conversion if solver changes
        from_rec = self.db.get_case(from_id) or {}
        to_rec = self.db.get_case(to_id) or {}
        from_solver = from_rec.get("solver", "simpleFoam")
        to_solver = to_rec.get("solver", "simpleFoam")
        solver_change = (from_solver == "simpleFoam") and (to_solver == "buoyantSimpleFoam")

        if solver_change:
            logger.info("Solver change %s → %s: converting p and creating T", from_solver, to_solver)
            T_ref = 300.0
            if inflow_json and inflow_json.exists():
                with open(inflow_json) as f:
                    T_ref = float(json.load(f).get("T_ref", T_ref))
            _convert_fields(
                source_case=from_case,
                target_0=to_case / "0",
                rho_ref=1.225,
                T_ref=T_ref,
                inflow_json=inflow_json,
            )

        # 3. mapFields (U, k, epsilon and p if same solver)
        _mapfields_local(from_case, to_case)
        logger.info("IC prepared for %s", to_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-resolution cascade CFD via hpc-foam.")
    p.add_argument("--hpc", type=Path, default=PROJECT_ROOT / "configs" / "hpc" / "aqua.yaml",
                   help="HPC config YAML (default: configs/hpc/aqua.yaml).")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command", required=True)

    # submit
    s = sub.add_parser("submit", help="Generate, upload, and submit cascade levels.")
    s.add_argument("--inflow", required=True, type=Path)
    s.add_argument("--resolutions", nargs="+", type=float, default=[500, 250, 100, 40])
    s.add_argument("--solvers", nargs="+", type=str,
                   default=["simpleFoam", "simpleFoam", "buoyantSimpleFoam", "buoyantSimpleFoam"])
    s.add_argument("--site", default="perdigao")
    s.add_argument("--of-version", type=int, default=9)
    s.add_argument("--n-iter", type=int, default=2000)
    s.add_argument("--write-interval", type=int, default=200)

    # timing-test
    t = sub.add_parser("timing-test", help="Submit N replicas for wall-time calibration.")
    t.add_argument("--inflow", required=True, type=Path)
    t.add_argument("--resolution", type=float, default=500)
    t.add_argument("--solver", default="simpleFoam")
    t.add_argument("--n-replicas", type=int, default=3)
    t.add_argument("--site", default="perdigao")
    t.add_argument("--of-version", type=int, default=9)
    t.add_argument("--n-iter", type=int, default=1000)

    # monitor
    sub.add_parser("monitor", help="Show status of submitted jobs.")

    # next-level
    nl = sub.add_parser("next-level", help="Download converged level and prepare next IC.")
    nl.add_argument("--from-res", type=float, required=True)
    nl.add_argument("--to-res", type=float, required=True)
    nl.add_argument("--site", default="perdigao")
    nl.add_argument("--inflow", type=Path, default=None)

    # list
    sub.add_parser("list", help="List all cascade cases in the DB.")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    study = CascadeStudy(args.hpc)

    if args.command == "submit":
        assert len(args.resolutions) == len(args.solvers), \
            "--resolutions and --solvers must have the same number of values"
        study.submit(
            inflow_json=args.inflow,
            resolutions=args.resolutions,
            solver_names=args.solvers,
            site=args.site,
            of_version=args.of_version,
            n_iterations=args.n_iter,
            write_interval=args.write_interval,
        )

    elif args.command == "timing-test":
        study.timing_test(
            inflow_json=args.inflow,
            resolution_m=args.resolution,
            solver_name=args.solver,
            n_replicas=args.n_replicas,
            site=args.site,
            of_version=args.of_version,
            n_iterations=args.n_iter,
        )

    elif args.command == "monitor":
        study.monitor()

    elif args.command == "next-level":
        study.prepare_next_level(
            from_res=args.from_res,
            to_res=args.to_res,
            site=args.site,
            inflow_json=args.inflow,
        )

    elif args.command == "list":
        groups = study.db.list_cascade_groups()
        timing_cases = study.db.list_cases(group_name=None)
        timing_cases = [c for c in timing_cases if "timing" in c["case_id"]]

        if not groups and not timing_cases:
            print("No cases in database.")
        else:
            if groups:
                print("=== Cascade chains ===")
                for gid in groups:
                    chain = study.db.get_cascade_chain(gid)
                    print(f"\n{gid}:")
                    print(f"  {'Level ID':<45} {'Res':>6} {'Solver':<25} {'Job ID':<15} State")
                    print("  " + "-" * 100)
                    for c in chain:
                        params = c.get("parameters", {})
                        print(
                            f"  {c['case_id']:<45} "
                            f"{params.get('resolution_m', 0):>6.0f} "
                            f"{c.get('solver') or '—':<25} "
                            f"{c.get('job_id') or '—':<15} "
                            f"{c['state']}"
                        )
            if timing_cases:
                print("\n=== Timing replicas ===")
                print(f"{'Case ID':<45} {'Res':>6} {'Solver':<25} {'Job ID':<15} State")
                print("-" * 100)
                for c in sorted(timing_cases, key=lambda x: x["case_id"]):
                    params = c.get("parameters", {})
                    print(
                        f"{c['case_id']:<45} "
                        f"{params.get('resolution_m', 0):>6.0f} "
                        f"{c.get('solver') or '—':<25} "
                        f"{c.get('job_id') or '—':<15} "
                        f"{c['state']}"
                    )


if __name__ == "__main__":
    main()
