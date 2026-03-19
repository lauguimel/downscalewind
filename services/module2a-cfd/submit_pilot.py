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
        "local_dir": PILOT_DIR,
        "walltime": "4:00:00",
        "mem_gb": 8,
        "ncpus": 12,
    },
    "pilot_bbsf": {
        "manifest": "campaign_manifest_pilot_bbsf.json",
        "local_dir": PILOT_DIR,
        "walltime": "5:00:00",
        "mem_gb": 8,
        "ncpus": 12,
    },
    # ── Phase 0.2 — Resolution sweep (500m / 250m / 100m, BBSF neutral) ──
    "phase0_resolution": {
        "manifest": None,   # auto-generated from configs/phase0_resolution.yaml
        "local_dir": PROJECT_ROOT / "data" / "cases" / "phase0_resolution",
        "walltime": "2:00:00",   # 100m run ≈ 40 min on 12 cores — keep margin
        "mem_gb": 16,
        "ncpus": 12,
    },
}


def _load_phase0_resolution_manifest() -> dict:
    """Build the phase0_resolution manifest from configs/phase0_resolution.yaml."""
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "phase0_resolution.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    manifest = {}
    for case_id, case_cfg in cfg["cases"].items():
        manifest[f"case_{case_id}"] = {
            "solver": case_cfg["solver"],
            "direction_deg": cfg["study"]["inflow"]["direction_deg"],
            "speed_ms": cfg["study"]["inflow"]["speed_ms"],
            "stability": case_cfg.get("stability", "neutral"),
            "resolution_m": case_cfg["resolution_m"],
            "thermal": case_cfg.get("thermal", True),
        }
    return manifest

# ── PBS template ────────────────────────────────────────────────────────
PBS_TEMPLATE = """\
#!/bin/bash -l
#PBS -N {job_name}
#PBS -l select=1:ncpus={ncpus}:mem={mem_gb}GB
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -V

cd {remote_dir}
chmod +x Allrun Allcontinue Allclean 2>/dev/null

# Run inside OpenFOAM 9 container
if [ -f "Allrun" ]; then
    {wrapper} -- "cd {remote_dir} && ./Allrun {ncpus}" > log.Allrun 2>&1
    exit $?
fi
"""


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_config() -> HPCConfig:
    return HPCConfig.load(str(HPC_PROFILE))


def _get_db() -> CaseDatabase:
    return CaseDatabase(str(DB_PATH))


# ── Commands ────────────────────────────────────────────────────────────

def _get_manifest(group_name: str, group_cfg: dict) -> dict:
    """Return the {case_id: params} manifest for a group."""
    if group_name == "phase0_resolution":
        return _load_phase0_resolution_manifest()
    manifest_path = group_cfg["local_dir"] / group_cfg["manifest"]
    return json.loads(manifest_path.read_text())


def register_cases(dry_run: bool = False, group_filter: str | None = None) -> list[str]:
    """Register pilot cases in the hpc-sim database."""
    db = _get_db()
    config = _get_config()
    registered = []

    for group_name, group_cfg in GROUPS.items():
        if group_filter and group_name != group_filter:
            continue
        manifest = _get_manifest(group_name, group_cfg)
        local_base = group_cfg["local_dir"]

        for case_id, params in manifest.items():
            local_dir = str(local_base / case_id)
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


def upload_cases(dry_run: bool = False, group_filter: str | None = None) -> list[str]:
    """Upload pilot cases to Aqua via hpc-sim TransferManager."""
    db = _get_db()
    config = _get_config()
    tm = TransferManager(config, db=db)
    uploaded = []

    for group_name in GROUPS:
        if group_filter and group_name != group_filter:
            continue
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


def submit_cases(dry_run: bool = False, group_filter: str | None = None) -> dict[str, str]:
    """Write PBS scripts remotely + submit via PBSManager.submit_batch()."""
    db = _get_db()
    config = _get_config()
    ssh = SSHExecutor.from_config(config)
    pbs = PBSManager(ssh)
    results = {}

    # Collect all scripts to submit in batch (single SSH session)
    scripts: dict[str, str] = {}  # case_id -> remote PBS script path

    for group_name, group_cfg in GROUPS.items():
        if group_filter and group_name != group_filter:
            continue
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
                remote_dir=remote_dir,
            )

            if dry_run:
                print(f"  [dry-run] Would submit {case_id} "
                      f"(walltime={group_cfg['walltime']}, ncpus={group_cfg['ncpus']})")
                continue

            # Write PBS script to remote via base64 (printf multiline fails over SSH)
            ssh.run(f"chmod +x {remote_dir}/Allrun")
            import base64
            b64 = base64.b64encode(pbs_content.encode()).decode()
            ssh.run(f"echo '{b64}' | base64 -d > {remote_dir}/run.pbs")

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
    parser.add_argument("--group", default=None,
                        choices=list(GROUPS.keys()),
                        help="Restrict to one group (default: all groups)")
    args = parser.parse_args()

    if args.all:
        args.register = args.upload = args.submit = True

    if not any([args.register, args.upload, args.submit, args.status]):
        parser.print_help()
        return

    if args.status:
        show_status()
        return

    g = args.group
    if args.register:
        print(f"=== Registering cases (group={g or 'all'}) ===")
        register_cases(dry_run=args.dry_run, group_filter=g)

    if args.upload:
        print(f"\n=== Uploading to Aqua (group={g or 'all'}) ===")
        upload_cases(dry_run=args.dry_run, group_filter=g)

    if args.submit:
        print(f"\n=== Submitting PBS jobs (group={g or 'all'}) ===")
        submit_cases(dry_run=args.dry_run, group_filter=g)


if __name__ == "__main__":
    main()
