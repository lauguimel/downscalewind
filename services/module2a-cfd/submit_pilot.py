"""
submit_pilot.py — Register, upload and submit pilot cases to Aqua (QUT)

Uses hpc-sim components (SSHExecutor with ControlMaster, TransferManager,
PBSManager) for robust SSH multiplexing.

Usage:
    python submit_pilot.py --register          # Register in hpc-sim DB
    python submit_pilot.py --upload            # Upload cases via rsync
    python submit_pilot.py --submit            # Write PBS scripts + qsub
    python submit_pilot.py --status            # Check DB + remote status
    python submit_pilot.py --all               # Register + upload + submit
    python submit_pilot.py --dry-run --all     # Preview everything
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add hpc-sim to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HPCSIM_ROOT = PROJECT_ROOT.parent / "hpc-sim"
sys.path.insert(0, str(HPCSIM_ROOT / "src"))

from hpc_sim.core.config import HPCConfig
from hpc_sim.core.db import CaseDatabase
from hpc_sim.core.ssh import SSHExecutor
from hpc_sim.core.transfer import TransferManager
from hpc_sim.core.pbs import PBSManager
from hpc_sim.core.state import CaseState

# ── Config ──────────────────────────────────────────────────────────────
DB_PATH = HPCSIM_ROOT / "cases.db"
PILOT_DIR = PROJECT_ROOT / "data" / "campaign" / "pilot_hpc"
HPC_PROFILE = HPCSIM_ROOT / "configs" / "aqua.yaml"
WRAPPER = "OF9_RT.sh"

GROUPS = {
    "pilot_sf": {
        "manifest": "campaign_manifest_pilot_sf.json",
        "walltime": "4:00:00",
        "mem_gb": 8,
        "ncpus": 12,
    },
    "pilot_bbsf": {
        "manifest": "campaign_manifest_pilot_bbsf.json",
        "walltime": "5:00:00",
        "mem_gb": 8,
        "ncpus": 12,
    },
}

# ── PBS template ────────────────────────────────────────────────────────
PBS_TEMPLATE = """\
#!/bin/bash
#PBS -N {job_name}
#PBS -q workq
#PBS -l select=1:ncpus={ncpus}:mem={mem_gb}GB
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {job_name}.log

cd $PBS_O_WORKDIR

# Run inside OpenFOAM 9 container
{wrapper} ./Allrun {ncpus}
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_config() -> HPCConfig:
    return HPCConfig.load(str(HPC_PROFILE))


def _get_db() -> CaseDatabase:
    return CaseDatabase(str(DB_PATH))


# ── Commands ────────────────────────────────────────────────────────────

def register_cases(dry_run: bool = False) -> list[str]:
    """Register pilot cases in the hpc-sim database."""
    db = _get_db()
    config = _get_config()
    registered = []

    for group_name, group_cfg in GROUPS.items():
        manifest_path = PILOT_DIR / group_cfg["manifest"]
        manifest = json.loads(manifest_path.read_text())

        for case_id, params in manifest.items():
            local_dir = str(PILOT_DIR / case_id)
            existing = db.get_case(case_id)
            if existing:
                print(f"  {case_id}: already in DB (state={existing['state']})")
                registered.append(case_id)
                continue

            if dry_run:
                print(f"  [dry-run] Would register {case_id} (group={group_name})")
                registered.append(case_id)
                continue

            db.add_case(
                case_id,
                parameters=params,
                group_name=group_name,
                local_dir=local_dir,
                remote_dir=f"{config.remote_base_dir}/{case_id}",
                solver=params["solver"],
                ncpus=group_cfg["ncpus"],
                end_time=2000.0,
            )
            db.update_state(case_id, CaseState.GENERATED)
            print(f"  {case_id}: registered + GENERATED "
                  f"({params['direction_deg']}° {params['speed_ms']}m/s "
                  f"{params['stability']})")
            registered.append(case_id)

    return registered


def upload_cases(dry_run: bool = False) -> list[str]:
    """Upload pilot cases to Aqua via hpc-sim TransferManager."""
    db = _get_db()
    config = _get_config()
    tm = TransferManager(config, db=db)
    uploaded = []

    for group_name in GROUPS:
        cases = db.list_cases(group_name=group_name)
        for case in cases:
            case_id = case["case_id"]
            local_dir = Path(case["local_dir"])

            if not local_dir.exists():
                print(f"  {case_id}: SKIP — local dir not found")
                continue

            if dry_run:
                print(f"  [dry-run] Would upload {case_id}")
                uploaded.append(case_id)
                continue

            print(f"  {case_id}: uploading...", end=" ", flush=True)
            ok = tm.upload_case(local_dir)
            if ok:
                print("OK")
                uploaded.append(case_id)
            else:
                print("FAILED")

    return uploaded


def submit_cases(dry_run: bool = False) -> dict[str, str]:
    """Write PBS scripts remotely + submit via PBSManager.submit_batch()."""
    db = _get_db()
    config = _get_config()
    ssh = SSHExecutor.from_config(config)
    pbs = PBSManager(ssh)
    results = {}

    # Collect all scripts to submit in batch (single SSH session)
    scripts: dict[str, str] = {}  # case_id -> remote PBS script path

    for group_name, group_cfg in GROUPS.items():
        cases = db.list_cases(group_name=group_name)
        for case in cases:
            case_id = case["case_id"]
            if case["state"] not in ("generated", "uploaded"):
                print(f"  {case_id}: SKIP (state={case['state']})")
                continue

            remote_dir = case.get("remote_dir") or f"{config.remote_base_dir}/{case_id}"
            pbs_content = PBS_TEMPLATE.format(
                job_name=case_id,
                ncpus=group_cfg["ncpus"],
                mem_gb=group_cfg["mem_gb"],
                walltime=group_cfg["walltime"],
                wrapper=WRAPPER,
            )

            if dry_run:
                print(f"  [dry-run] Would submit {case_id} "
                      f"(walltime={group_cfg['walltime']}, ncpus={group_cfg['ncpus']})")
                continue

            # Write PBS script to remote via SSH (single command)
            ssh.run(f"chmod +x {remote_dir}/Allrun")
            # Use printf to avoid heredoc issues
            escaped = pbs_content.replace("\\", "\\\\").replace("'", "'\\''")
            ssh.run(f"printf '%s' '{escaped}' > {remote_dir}/run.pbs")

            scripts[case_id] = f"{remote_dir}/run.pbs"
            print(f"  {case_id}: PBS script written")

    if dry_run or not scripts:
        return results

    # Batch submit (single SSH session for all qsub calls)
    print(f"\n  Submitting {len(scripts)} jobs in batch...")
    job_ids = pbs.submit_batch(scripts)

    for case_id, job_id in job_ids.items():
        if job_id:
            db.update_state(case_id, CaseState.SUBMITTED, job_id=job_id)
            print(f"  {case_id}: submitted -> {job_id}")
            results[case_id] = job_id
        else:
            print(f"  {case_id}: SUBMIT FAILED")

    return results


def show_status():
    """Show current pilot cases in the DB + remote job status."""
    db = _get_db()
    try:
        config = _get_config()
        ssh = SSHExecutor.from_config(config)
        r = ssh.run("qstat -u maitreje 2>/dev/null || true")
        if r.stdout.strip():
            print("=== Active PBS jobs ===")
            print(r.stdout)
    except Exception:
        pass

    for group_name in GROUPS:
        cases = db.list_cases(group_name=group_name)
        if not cases:
            print(f"\n{group_name}: no cases in DB")
            continue
        print(f"\n{group_name} ({len(cases)} cases):")
        for c in cases:
            params = c.get("parameters", {})
            print(f"  {c['case_id']}: state={c['state']}, "
                  f"job={c.get('job_id', '-')}, "
                  f"dir={params.get('direction_deg', '?')}° "
                  f"stab={params.get('stability', '?')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pilot HPC submission")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="Register + upload + submit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.register = args.upload = args.submit = True

    if not any([args.register, args.upload, args.submit, args.status]):
        parser.print_help()
        return

    if args.status:
        show_status()
        return

    if args.register:
        print("=== Registering pilot cases ===")
        register_cases(dry_run=args.dry_run)

    if args.upload:
        print("\n=== Uploading to Aqua ===")
        upload_cases(dry_run=args.dry_run)

    if args.submit:
        print("\n=== Submitting PBS jobs ===")
        submit_cases(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
