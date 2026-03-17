"""
run_convergence_study.py — Orchestrate convergence study on HPC via hpc_foam

Reads convergence_study.yaml + hpc/aqua.yaml and manages the full workflow:
  generate → upload → submit → monitor → download

Usage
-----
    # Generate all Phase 1 cases locally
    python run_convergence_study.py generate --phase mesh_convergence

    # Upload to HPC and submit
    python run_convergence_study.py submit --phase mesh_convergence

    # Monitor running jobs
    python run_convergence_study.py monitor

    # Download results
    python run_convergence_study.py download --phase mesh_convergence

    # Generate + submit in one step
    python run_convergence_study.py run --phase mesh_convergence

    # List all cases in the manifest
    python run_convergence_study.py list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# hpc_foam is installed from the IGNIS project
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "IGNIS" / "rheotool"))
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Case specification
# ---------------------------------------------------------------------------

@dataclass
class CaseSpec:
    """Specification for a single CFD case in the convergence study."""
    case_id: str                 # e.g. "rheotool_001"
    phase: str                   # e.g. "mesh_convergence"
    resolution_m: float
    domain_km: float
    context_cells: int
    direction_deg: float
    solver_name: str = "simpleFoam"
    thermal: bool = False
    coriolis: bool = True
    canopy: bool = False
    precursor: bool = False
    n_iterations: int = 2000
    write_interval: int = 200
    of_version: int = 9          # HPC default (OF9 Foundation)
    stability: str = "neutral"
    inlet_type: str = "idealized"
    notes: str = ""


# ---------------------------------------------------------------------------
# Study class
# ---------------------------------------------------------------------------

class ConvergenceStudy:
    """Orchestrator for the CFD convergence study."""

    def __init__(
        self,
        study_config_path: Path,
        hpc_config_path: Path,
    ):
        with open(study_config_path) as f:
            self.study_cfg = yaml.safe_load(f)
        with open(hpc_config_path) as f:
            self.hpc_cfg = yaml.safe_load(f)

        self.site = self.study_cfg["study"]["site"]
        self.prefix = self.study_cfg.get("case_prefix", "rheotool")
        self.n_iter = self.study_cfg["study"]["n_iterations"]
        self.write_interval = self.study_cfg["study"]["write_interval"]
        self.of_version = self.hpc_cfg.get("container", {}).get("of_version", 9)

        # Case output directory
        self.cases_dir = PROJECT_ROOT / "data" / "convergence" / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)

        # Manifest tracks case_id → conditions
        self.manifest_path = self.cases_dir / "convergence_study_manifest.json"
        self.manifest = self._load_manifest()

        # Counter for case IDs
        self._next_id = max(
            (int(k.split("_")[-1]) for k in self.manifest if k.startswith(self.prefix)),
            default=0,
        ) + 1

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _next_case_id(self) -> str:
        cid = f"{self.prefix}_{self._next_id:03d}"
        self._next_id += 1
        return cid

    # -----------------------------------------------------------------
    # Phase generators
    # -----------------------------------------------------------------

    def generate_phase_specs(self, phase_name: str) -> list[CaseSpec]:
        """Generate CaseSpec list for a given phase.

        Safe to call multiple times — resets the ID counter from manifest
        each time so repeated calls produce the same IDs.
        """
        # Reset counter from manifest to avoid drift from repeated calls
        self._next_id = max(
            (int(k.split("_")[-1]) for k in self.manifest if k.startswith(self.prefix)),
            default=0,
        ) + 1

        generators = {
            "mesh_convergence": self._gen_mesh_convergence,
            "domain_sensitivity": self._gen_domain_sensitivity,
            "physics_comparison": self._gen_physics_comparison,
            "precursor_stability": self._gen_precursor_stability,
        }
        if phase_name not in generators:
            raise ValueError(f"Unknown phase: {phase_name}. Choose from {list(generators)}")
        return generators[phase_name]()

    def _gen_mesh_convergence(self) -> list[CaseSpec]:
        cfg = self.study_cfg["mesh_convergence"]
        specs = []
        for res in cfg["resolutions_m"]:
            for d in cfg["directions_deg"]:
                specs.append(CaseSpec(
                    case_id=self._next_case_id(),
                    phase="mesh_convergence",
                    resolution_m=res,
                    domain_km=cfg["domain_km"],
                    context_cells=cfg["context_cells"],

                    direction_deg=d,
                    coriolis=True,
                    n_iterations=self.n_iter,
                    write_interval=self.write_interval,
                    of_version=self.of_version,
                ))
        return specs

    def _gen_domain_sensitivity(self) -> list[CaseSpec]:
        cfg = self.study_cfg["domain_sensitivity"]
        specs = []
        for dk in cfg["domain_sizes_km"]:
            for d in cfg["directions_deg"]:
                specs.append(CaseSpec(
                    case_id=self._next_case_id(),
                    phase="domain_sensitivity",
                    resolution_m=cfg["resolution_m"],
                    domain_km=dk,
                    context_cells=cfg["context_cells"],

                    direction_deg=d,
                    coriolis=True,
                    n_iterations=self.n_iter,
                    write_interval=self.write_interval,
                    of_version=self.of_version,
                ))
        return specs

    def _gen_physics_comparison(self) -> list[CaseSpec]:
        cfg = self.study_cfg["physics_comparison"]
        specs = []
        for config_name, pcfg in cfg["configs"].items():
            for d in cfg["directions_deg"]:
                specs.append(CaseSpec(
                    case_id=self._next_case_id(),
                    phase="physics_comparison",
                    resolution_m=cfg["resolution_m"],
                    domain_km=cfg["domain_km"],
                    context_cells=cfg["context_cells"],

                    direction_deg=d,
                    solver_name=pcfg["solver"],
                    thermal=pcfg.get("thermal", False),
                    coriolis=pcfg.get("coriolis", True),
                    canopy=pcfg.get("canopy", False),
                    precursor=pcfg.get("precursor", False),
                    n_iterations=self.n_iter,
                    write_interval=self.write_interval,
                    of_version=self.of_version,
                    notes=f"config_{config_name}",
                ))
        return specs

    def _gen_precursor_stability(self) -> list[CaseSpec]:
        cfg = self.study_cfg["precursor_stability"]
        specs = []
        for stab_name in cfg["stabilities"]:
            for inlet_type in cfg["inlets"]:
                specs.append(CaseSpec(
                    case_id=self._next_case_id(),
                    phase="precursor_stability",
                    resolution_m=cfg["resolution_m"],
                    domain_km=cfg["domain_km"],
                    context_cells=cfg["context_cells"],

                    direction_deg=cfg["direction_deg"],
                    solver_name="buoyantBoussinesqSimpleFoam",
                    thermal=True,
                    coriolis=True,
                    canopy=True,
                    stability=stab_name,
                    inlet_type=inlet_type,
                    n_iterations=self.n_iter,
                    write_interval=self.write_interval,
                    of_version=self.of_version,
                    notes=f"stability_{stab_name}_{inlet_type}",
                ))
        return specs

    # -----------------------------------------------------------------
    # Case preparation (local)
    # -----------------------------------------------------------------

    def prepare_case(self, spec: CaseSpec) -> Path:
        """Generate a complete OpenFOAM case directory for a CaseSpec.

        Pipeline: prepare_inflow → generate_mesh → (generate_lad_field if canopy)

        Returns the case directory path.
        """
        from generate_mesh import generate_mesh

        case_dir = self.cases_dir / spec.case_id
        if case_dir.exists():
            logger.info("Case %s already exists, skipping", spec.case_id)
            return case_dir

        # Load site config
        site_cfg_path = PROJECT_ROOT / "configs" / "sites" / f"{self.site}.yaml"
        with open(site_cfg_path) as f:
            site_cfg = yaml.safe_load(f)

        logger.info(
            "Preparing %s: res=%gm, domain=%gkm, dir=%g°, solver=%s",
            spec.case_id, spec.resolution_m, spec.domain_km,
            spec.direction_deg, spec.solver_name,
        )

        # 1. Prepare inflow profile
        #    Generate base profile from ERA5 once, then override wind direction
        #    for each case in the convergence matrix.
        import math

        inflow_dir = self.cases_dir / "inflow_profiles"
        inflow_dir.mkdir(exist_ok=True)

        site = site_cfg["site"]
        site_lat = site["coordinates"]["latitude"]
        site_lon = site["coordinates"]["longitude"]

        # Base profile (ERA5 direction, shared across all cases at this timestamp)
        base_inflow_json = inflow_dir / "inflow_base.json"
        if not base_inflow_json.exists():
            from prepare_inflow import prepare_inflow

            era5_zarr = PROJECT_ROOT / "data" / "raw" / f"era5_{self.site}.zarr"
            z0_tif = PROJECT_ROOT / "data" / "raw" / f"landcover_{self.site}.tif"

            prepare_inflow(
                era5_zarr=era5_zarr,
                timestamp=self.study_cfg["study"]["timestamp"],
                site_lat=site_lat,
                site_lon=site_lon,
                z0_tif=z0_tif if z0_tif.exists() else None,
                output_json=base_inflow_json,
            )

        # Per-direction profile: override flowDir and wind_dir from base
        inflow_json = inflow_dir / f"inflow_{spec.direction_deg:.0f}deg.json"
        if not inflow_json.exists():
            with open(base_inflow_json) as f:
                inflow_data = json.load(f)

            # Convert met direction (wind FROM, clockwise from N)
            # to flow unit vector (direction wind blows TOWARD)
            wind_toward_deg = (spec.direction_deg + 180.0) % 360.0
            wind_toward_rad = math.radians(wind_toward_deg)
            # Met convention: 0°=N, 90°=E → x=sin(θ), y=cos(θ)
            inflow_data["flowDir_x"] = math.sin(wind_toward_rad)
            inflow_data["flowDir_y"] = math.cos(wind_toward_rad)
            inflow_data["wind_dir"] = spec.direction_deg

            with open(inflow_json, "w") as f:
                json.dump(inflow_data, f, indent=2)
            logger.info("Inflow profile for %.0f°: %s", spec.direction_deg, inflow_json)

        # 2. Generate mesh + render templates
        srtm_path = PROJECT_ROOT / "data" / "raw" / f"srtm_{self.site}_30m.tif"
        if not srtm_path.exists():
            srtm_path = None

        generate_mesh(
            site_cfg=site_cfg,
            resolution_m=spec.resolution_m,
            context_cells=spec.context_cells,
            output_dir=case_dir,
            srtm_tif=srtm_path,
            inflow_json=inflow_json,
            domain_km=spec.domain_km,
            solver_name=spec.solver_name,
            thermal=spec.thermal,
        )

        # 3. If canopy, generate LAD field
        if spec.canopy:
            from generate_lad_field import generate_lad_field
            landcover_tif = PROJECT_ROOT / "data" / "raw" / f"landcover_{self.site}.tif"
            if landcover_tif.exists():
                site = site_cfg["site"]
                generate_lad_field(
                    case_dir=case_dir,
                    landcover_tif=landcover_tif,
                    site_lat=site["coordinates"]["latitude"],
                    site_lon=site["coordinates"]["longitude"],
                )
            else:
                logger.warning("Land cover raster not found: %s — skipping canopy", landcover_tif)

        # 4. Save to manifest
        self.manifest[spec.case_id] = {
            "phase": spec.phase,
            "resolution_m": spec.resolution_m,
            "domain_km": spec.domain_km,
            "direction_deg": spec.direction_deg,
            "solver": spec.solver_name,
            "thermal": spec.thermal,
            "coriolis": spec.coriolis,
            "canopy": spec.canopy,
            "precursor": spec.precursor,
            "stability": spec.stability,
            "inlet_type": spec.inlet_type,
            "notes": spec.notes,
        }
        self._save_manifest()

        logger.info("Case %s ready: %s", spec.case_id, case_dir)
        return case_dir

    def prepare_phase(self, phase_name: str) -> list[Path]:
        """Generate all cases for a phase."""
        specs = self.generate_phase_specs(phase_name)
        logger.info("Generating %d cases for phase '%s'", len(specs), phase_name)
        case_dirs = []
        for spec in specs:
            case_dirs.append(self.prepare_case(spec))
        self._save_manifest()
        return case_dirs

    # -----------------------------------------------------------------
    # HPC operations
    # -----------------------------------------------------------------

    def _get_hpc_config(self):
        """Create hpc_foam.HPCConfig from aqua.yaml."""
        from hpc_foam import HPCConfig

        conn = self.hpc_cfg["connection"]
        res = self.hpc_cfg["resources"]
        container = self.hpc_cfg.get("container", {})

        return HPCConfig(
            hpc_host=conn["hpc_host"],
            username=conn["username"],
            remote_base_dir=Path(conn["remote_base_dir"]),
            local_base_dir=self.cases_dir,
            solver=self.hpc_cfg.get("solver", {}).get("name", "simpleFoam"),
            default_ncpus=res["default_ncpus"],
            default_mem=res["default_mem"],
            default_walltime=res["default_walltime"],
            apptainer_wrapper=container.get("apptainer_wrapper", ""),
            parallel=self.hpc_cfg.get("solver", {}).get("parallel", True),
        )

    def upload_phase(self, phase_name: str) -> dict:
        """Upload all cases for a phase to HPC."""
        from hpc_foam import CaseTransferManager

        config = self._get_hpc_config()
        transfer = CaseTransferManager(config)

        case_names = [
            cid for cid, info in self.manifest.items()
            if info["phase"] == phase_name
        ]
        if not case_names:
            logger.warning("No cases found for phase '%s'", phase_name)
            return {}

        logger.info("Uploading %d cases for phase '%s'", len(case_names), phase_name)
        return transfer.upload_all_cases(case_names)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def clean_phase(self, phase_name: str, remote: bool = True) -> dict:
        """Clean everything for a phase: cancel jobs, delete local+remote dirs.

        Returns summary dict with counts.
        """
        import shutil

        summary = {"cancelled": 0, "local_deleted": 0, "remote_cleaned": False}

        # 1. Cancel running jobs
        summary["cancelled"] = self.cancel_phase(phase_name)

        # 2. Find case names for this phase
        case_names = [
            cid for cid, info in self.manifest.items()
            if info["phase"] == phase_name
        ]

        # 3. Delete local case directories
        for cname in case_names:
            case_dir = self.cases_dir / cname
            if case_dir.exists():
                shutil.rmtree(case_dir)
                summary["local_deleted"] += 1

        # 4. Remove from manifest
        for cname in case_names:
            self.manifest.pop(cname, None)
        self._save_manifest()

        # Reset ID counter
        self._next_id = max(
            (int(k.split("_")[-1]) for k in self.manifest if k.startswith(self.prefix)),
            default=0,
        ) + 1

        # 5. Clean remote directories
        if remote and case_names:
            try:
                from hpc_foam.ssh_executor import SSHExecutor
                config = self._get_hpc_config()
                ssh = SSHExecutor.from_config(config)
                remote_base = Path(self.hpc_cfg["connection"]["remote_base_dir"])
                rm_cmds = [f"rm -rf {remote_base / cn}" for cn in case_names]
                ssh.run(" && ".join(rm_cmds), timeout=60)
                summary["remote_cleaned"] = True
            except Exception as e:
                logger.warning("Remote cleanup failed: %s", e)
                summary["remote_cleaned"] = False

        # 6. Remove inflow profiles (they get regenerated)
        inflow_dir = self.cases_dir / "inflow_profiles"
        if inflow_dir.exists():
            shutil.rmtree(inflow_dir)

        logger.info("Cleaned phase '%s': %s", phase_name, summary)
        return summary

    def clean_all(self, remote: bool = True) -> dict:
        """Clean everything: all phases, all jobs, all local+remote dirs."""
        import shutil

        # Cancel all jobs first
        n_cancelled = self.cancel_all()

        # Delete all case directories
        case_names = list(self.manifest.keys())
        n_local = 0
        for cname in case_names:
            case_dir = self.cases_dir / cname
            if case_dir.exists():
                shutil.rmtree(case_dir)
                n_local += 1

        # Clear manifest
        self.manifest.clear()
        self._save_manifest()
        self._next_id = 1

        # Clean remote
        remote_cleaned = False
        if remote and case_names:
            try:
                from hpc_foam.ssh_executor import SSHExecutor
                config = self._get_hpc_config()
                ssh = SSHExecutor.from_config(config)
                remote_base = self.hpc_cfg["connection"]["remote_base_dir"]
                ssh.run(f"rm -rf {remote_base}/*", timeout=60)
                remote_cleaned = True
            except Exception as e:
                logger.warning("Remote cleanup failed: %s", e)

        # Remove inflow profiles
        inflow_dir = self.cases_dir / "inflow_profiles"
        if inflow_dir.exists():
            shutil.rmtree(inflow_dir)

        # Remove job status files
        for f in self.cases_dir.glob("*.json"):
            if f.name != "convergence_study_manifest.json":
                f.unlink()

        summary = {
            "cancelled": n_cancelled,
            "local_deleted": n_local,
            "remote_cleaned": remote_cleaned,
            "cases": len(case_names),
        }
        logger.info("Cleaned all: %s", summary)
        return summary

    def cancel_phase(self, phase_name: str) -> int:
        """Cancel all submitted jobs for a phase via qdel.

        Returns the number of jobs cancelled.
        """
        from hpc_foam.ssh_executor import SSHExecutor

        config = self._get_hpc_config()
        ssh = SSHExecutor.from_config(config)

        jobs_file = self.cases_dir / f"submitted_jobs_{phase_name}.json"
        if not jobs_file.exists():
            logger.info("No submitted jobs file for phase '%s'", phase_name)
            return 0

        with open(jobs_file) as f:
            data = json.load(f)

        job_ids = [j["job_id"] for j in data.get("jobs", [])]
        if not job_ids:
            return 0

        # Cancel all jobs in one SSH call (qdel ignores already-finished jobs)
        qdel_cmd = "qdel " + " ".join(job_ids) + " 2>/dev/null; echo done"
        result = ssh.run(qdel_cmd, timeout=30)
        logger.info("Cancelled %d jobs for phase '%s': %s",
                     len(job_ids), phase_name, result.stdout.strip())

        # Remove the jobs file so submit_phase starts fresh
        jobs_file.unlink()
        return len(job_ids)

    def cancel_all(self) -> int:
        """Cancel ALL submitted jobs across all phases."""
        from hpc_foam.ssh_executor import SSHExecutor

        config = self._get_hpc_config()
        ssh = SSHExecutor.from_config(config)

        all_ids = []
        for jf in sorted(self.cases_dir.glob("submitted_jobs_*.json")):
            with open(jf) as f:
                data = json.load(f)
            for j in data.get("jobs", []):
                all_ids.append(j["job_id"])

        if not all_ids:
            logger.info("No submitted jobs to cancel")
            return 0

        qdel_cmd = "qdel " + " ".join(all_ids) + " 2>/dev/null; echo done"
        result = ssh.run(qdel_cmd, timeout=30)
        logger.info("Cancelled %d jobs: %s", len(all_ids), result.stdout.strip())

        # Remove all job files
        for jf in self.cases_dir.glob("submitted_jobs_*.json"):
            jf.unlink()

        return len(all_ids)

    def submit_phase(self, phase_name: str) -> list:
        """Submit all cases for a phase to HPC PBS queue.

        Uses Allrun-based PBS scripts so that the full pipeline
        (cartesianMesh → checkMesh → decomposePar → solver → reconstructPar)
        runs on the HPC node.

        If jobs already exist for this phase, they are cancelled first (qdel).
        """
        from hpc_foam.pbs_templates import generate_allrun_pbs_script
        from hpc_foam.ssh_executor import SSHExecutor
        from hpc_foam.pbs_manager import PBSJob

        config = self._get_hpc_config()
        ssh = SSHExecutor.from_config(config)

        # Cancel any existing jobs for this phase before resubmitting
        jobs_file = self.cases_dir / f"submitted_jobs_{phase_name}.json"
        if jobs_file.exists():
            n_cancelled = self.cancel_phase(phase_name)
            if n_cancelled:
                logger.info("Cancelled %d old jobs before resubmitting", n_cancelled)

        case_names = [
            cid for cid, info in self.manifest.items()
            if info["phase"] == phase_name
        ]
        if not case_names:
            logger.warning("No cases found for phase '%s'", phase_name)
            return []

        remote_base = Path(self.hpc_cfg["connection"]["remote_base_dir"])
        res = self.hpc_cfg["resources"]
        container = self.hpc_cfg.get("container", {})
        wrapper = container.get("apptainer_wrapper", "")
        ncpus = res["default_ncpus"]
        walltime = res["default_walltime"]
        mem = res["default_mem"]

        # 1. Generate Allrun-based PBS scripts locally
        local_scripts = []
        for cname in case_names:
            remote_case = remote_base / cname
            script_content = generate_allrun_pbs_script(
                case_dir=str(remote_case),
                job_name=cname,
                ncpus=ncpus,
                mem=mem,
                walltime=walltime,
                wrapper=wrapper,
            )
            local_pbs = self.cases_dir / cname / "run.pbs"
            local_pbs.write_text(script_content)
            local_pbs.chmod(0o755)
            local_scripts.append((cname, local_pbs))
            logger.debug("Generated PBS script for %s", cname)

        # 2. Upload all PBS scripts in one rsync
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for cname, local_pbs in local_scripts:
                dest = tmpdir / cname
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_pbs, dest / "run.pbs")

            import subprocess
            rsync_cmd = [
                "rsync", "-avz", "-e", ssh.rsync_transport,
                "--include=*/", "--include=run.pbs", "--exclude=*",
                f"{tmpdir}/",
                f"{ssh.target}:{remote_base}/",
            ]
            subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)
        logger.info("Uploaded %d PBS scripts", len(local_scripts))

        # 3. Submit all jobs via single SSH session
        submit_cmds = []
        for cname in case_names:
            remote_case = remote_base / cname
            submit_cmds.append(f"cd {remote_case} && qsub run.pbs")

        all_commands = " && echo '---' && ".join(submit_cmds)
        result = ssh.run(all_commands, timeout=120)

        if result.returncode != 0:
            logger.error("Submit failed: %s", result.stderr)
            raise RuntimeError(f"PBS submission failed: {result.stderr}")

        # 4. Parse job IDs from qsub output
        import re
        jobs = []
        raw_ids = result.stdout.strip().split("---")
        for cname, raw_id in zip(case_names, raw_ids):
            job_id = raw_id.strip()
            # qsub returns e.g. "12345.pbs" — extract the numeric part
            m = re.match(r"(\d+\S*)", job_id)
            if m:
                job_id = m.group(1)
            jobs.append(PBSJob(job_id=job_id, name=cname))
            logger.info("Submitted %s → %s", cname, job_id)

        # 5. Save job IDs
        job_data = {
            "phase": phase_name,
            "jobs": [{"job_id": j.job_id, "case_name": j.name} for j in jobs],
        }
        jobs_file = self.cases_dir / f"submitted_jobs_{phase_name}.json"
        with open(jobs_file, "w") as f:
            json.dump(job_data, f, indent=2)

        logger.info("Submitted %d jobs → %s", len(jobs), jobs_file)
        return jobs

    def monitor(self, phase_name: str | None = None) -> dict:
        """Check status of all submitted jobs."""
        from hpc_foam import check_all_jobs_and_save_status

        config = self._get_hpc_config()

        # Collect all job files and build {job_id: case_name} dict
        jobs_dict: dict[str, str] = {}
        for jf in sorted(self.cases_dir.glob("submitted_jobs_*.json")):
            with open(jf) as f:
                data = json.load(f)
            if phase_name and data.get("phase") != phase_name:
                continue
            for j in data["jobs"]:
                jobs_dict[j["job_id"]] = j["case_name"]

        if not jobs_dict:
            logger.warning("No submitted jobs found")
            return {}

        status = check_all_jobs_and_save_status(
            config=config,
            submitted_jobs=jobs_dict,
            output_file=self.cases_dir / "job_status.json",
        )

        summary = status.get("summary", {})
        logger.info(
            "Job status: %d total — %d running, %d completed, %d failed",
            summary.get("total", len(jobs_dict)),
            summary.get("running", 0),
            summary.get("completed", 0),
            summary.get("to_restart", 0),
        )
        return status

    def download_phase(self, phase_name: str) -> dict:
        """Download results for completed cases in a phase."""
        from hpc_foam import CaseTransferManager

        config = self._get_hpc_config()
        transfer = CaseTransferManager(config)

        case_names = [
            cid for cid, info in self.manifest.items()
            if info["phase"] == phase_name
        ]
        if not case_names:
            logger.warning("No cases for phase '%s'", phase_name)
            return {}

        logger.info("Downloading %d cases for phase '%s'", len(case_names), phase_name)
        return transfer.download_results_batch(case_names)

    # -----------------------------------------------------------------
    # All-in-one
    # -----------------------------------------------------------------

    def run_phase(self, phase_name: str) -> None:
        """Generate, upload, and submit a full phase."""
        self.prepare_phase(phase_name)
        self.upload_phase(phase_name)
        self.submit_phase(phase_name)
        logger.info("Phase '%s' submitted. Use 'monitor' to track progress.", phase_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Convergence study orchestrator")
    parser.add_argument(
        "action",
        choices=["generate", "upload", "submit", "monitor", "download", "run", "list"],
        help="Action to perform",
    )
    parser.add_argument("--phase", default=None,
                        help="Phase name (mesh_convergence, domain_sensitivity, etc.)")
    parser.add_argument("--study-config",
                        default=str(PROJECT_ROOT / "configs" / "convergence_study.yaml"),
                        help="Study config YAML")
    parser.add_argument("--hpc-config",
                        default=str(PROJECT_ROOT / "configs" / "hpc" / "aqua.yaml"),
                        help="HPC config YAML")
    args = parser.parse_args()

    study = ConvergenceStudy(
        study_config_path=Path(args.study_config),
        hpc_config_path=Path(args.hpc_config),
    )

    if args.action == "list":
        print(f"Manifest: {len(study.manifest)} cases")
        for cid, info in sorted(study.manifest.items()):
            print(f"  {cid}: {info['phase']} | {info['resolution_m']}m "
                  f"{info['domain_km']}km {info['direction_deg']}° "
                  f"({info['solver']})")
        return

    if args.action in ("generate", "upload", "submit", "download", "run"):
        if not args.phase:
            parser.error(f"--phase is required for action '{args.action}'")

    if args.action == "generate":
        case_dirs = study.prepare_phase(args.phase)
        print(f"Generated {len(case_dirs)} cases")
    elif args.action == "upload":
        results = study.upload_phase(args.phase)
        ok = sum(v for v in results.values())
        print(f"Uploaded {ok}/{len(results)} cases")
    elif args.action == "submit":
        jobs = study.submit_phase(args.phase)
        print(f"Submitted {len(jobs)} jobs")
    elif args.action == "monitor":
        study.monitor(args.phase)
    elif args.action == "download":
        results = study.download_phase(args.phase)
        ok = sum(v for v in results.values())
        print(f"Downloaded {ok}/{len(results)} cases")
    elif args.action == "run":
        study.run_phase(args.phase)


if __name__ == "__main__":
    main()
