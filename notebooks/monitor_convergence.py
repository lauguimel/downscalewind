import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convergence Study — HPC Dashboard

    **Site**: Perdigao, Portugal (IOP 2017-05-04 22:00 UTC)

    4 phases: mesh convergence (8 runs) | domain sensitivity (8) | physics comparison (10) | precursor/stability (4)

    **Workflow**: generate cases locally → upload to HPC → submit PBS jobs → monitor → download → analyze
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(0, '../services/module2a-cfd')
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import re
    import subprocess
    import time

    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    ROOT = Path('..').resolve()
    DATA = ROOT / 'data'
    CASES_DIR = DATA / 'convergence' / 'cases'
    MANIFEST_PATH = CASES_DIR / 'convergence_study_manifest.json'

    STUDY_CFG_PATH = ROOT / 'configs' / 'convergence_study.yaml'
    HPC_CFG_PATH = ROOT / 'configs' / 'hpc' / 'aqua.yaml'

    # hpc_foam path
    HPC_FOAM_PATH = ROOT.parent / 'IGNIS' / 'rheotool' / 'hpc_foam'
    if str(HPC_FOAM_PATH) not in sys.path:
        sys.path.insert(0, str(HPC_FOAM_PATH.parent))

    print(f"Root: {ROOT}")
    print(f"Cases: {CASES_DIR}")
    print(f"hpc_foam: {HPC_FOAM_PATH}")
    return (
        CASES_DIR,
        DATA,
        HPC_CFG_PATH,
        MANIFEST_PATH,
        ROOT,
        STUDY_CFG_PATH,
        json,
        np,
        pd,
        plt,
        re,
        subprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part A — HPC Deployment

    Generate, upload, submit, monitor, and download convergence study cases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A1. Study orchestrator

    Loads `convergence_study.yaml` + `hpc/aqua.yaml` and creates the `ConvergenceStudy` object.
    """)
    return


@app.cell
def _(HPC_CFG_PATH, STUDY_CFG_PATH, mo):
    from _archive.run_convergence_study import ConvergenceStudy

    _missing = []
    if not STUDY_CFG_PATH.exists():
        _missing.append(str(STUDY_CFG_PATH))
    if not HPC_CFG_PATH.exists():
        _missing.append(str(HPC_CFG_PATH))

    if _missing:
        study = None
        mo.output.replace(mo.md(
            f"**Missing config files:**\n\n" +
            "\n".join(f"- `{p}`" for p in _missing)
        ))
    else:
        study = ConvergenceStudy(STUDY_CFG_PATH, HPC_CFG_PATH)
        print(f"Study loaded: {len(study.manifest)} cases in manifest")
        print(f"HPC host: {study.hpc_cfg['connection']['hpc_host']}")
        print(f"OF version: {study.of_version}")
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A2. Phase selector + status
    """)
    return


@app.cell
def _(mo):
    PHASES = ["mesh_convergence", "domain_sensitivity", "physics_comparison", "precursor_stability"]

    phase_selector = mo.ui.dropdown(
        options=PHASES,
        value="mesh_convergence",
        label="Phase",
    )
    phase_selector
    return (phase_selector,)


@app.cell
def _(CASES_DIR, json, mo, pd, phase_selector, study):
    """Show current state of all cases for the selected phase."""
    _phase = phase_selector.value
    if study is None:
        mo.output.replace(mo.md("*Study not initialized*"))
    else:
        specs = study.generate_phase_specs(_phase)
        _rows = []
        for s in specs:
            _local = (study.cases_dir / s.case_id).exists()
            _in_manifest = s.case_id in study.manifest
            # Check if job exists
            _job_file = CASES_DIR / f"submitted_jobs_{_phase}.json"
            _job_id = ""
            if _job_file.exists():
                with open(_job_file) as _f:
                    _jdata = json.load(_f)
                for _j in _jdata.get("jobs", []):
                    if _j["case_name"] == s.case_id:
                        _job_id = _j["job_id"]
            _rows.append({
                "case_id": s.case_id,
                "resolution": f"{s.resolution_m}m",
                "domain": f"{s.domain_km}km",
                "direction": f"{s.direction_deg}°",
                "solver": s.solver_name,
                "local": "yes" if _local else "-",
                "manifest": "yes" if _in_manifest else "-",
                "job_id": _job_id or "-",
            })
        _df = pd.DataFrame(_rows)
        mo.output.replace(mo.vstack([
            mo.md(f"**Phase `{_phase}`** — {len(specs)} cases"),
            mo.ui.table(_df, selection=None),
        ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A3. Actions

    **Pipeline**: Clean → Generate → Upload → Submit

    | Button | What it does |
    |--------|-------------|
    | **Clean phase** | qdel jobs + delete local dirs + remove from manifest + clean remote |
    | **Clean ALL** | Same but for all phases (nuclear reset) |
    | **Generate** | Create case dirs locally (prepare_inflow + generate_mesh) |
    | **Upload** | Rsync local case dirs to HPC |
    | **Submit** | Auto-cancels old jobs for the phase, then qsub new ones |
    """)
    return


@app.cell
def _(mo):
    clean_phase_btn = mo.ui.run_button(label="Clean phase", kind="warn")
    clean_all_btn = mo.ui.run_button(label="Clean ALL", kind="danger")
    generate_btn = mo.ui.run_button(label="Generate")
    upload_btn = mo.ui.run_button(label="Upload to HPC")
    submit_btn = mo.ui.run_button(label="Submit to PBS", kind="danger")
    cancel_btn = mo.ui.run_button(label="Cancel jobs")
    mo.hstack([clean_phase_btn, clean_all_btn, generate_btn, upload_btn, submit_btn, cancel_btn], gap=0.5)
    return (
        cancel_btn,
        clean_all_btn,
        clean_phase_btn,
        generate_btn,
        submit_btn,
        upload_btn,
    )


@app.cell
def _(
    cancel_btn,
    clean_all_btn,
    clean_phase_btn,
    generate_btn,
    mo,
    phase_selector,
    study,
    submit_btn,
    upload_btn,
):
    """Handle all action buttons."""
    _phase = phase_selector.value
    if study is None:
        mo.output.replace(mo.md("*Study not initialized*"))
    elif clean_all_btn.value:
        try:
            _r = study.clean_all(remote=True)
            print(f"CLEANED ALL: {_r['cases']} cases deleted, "
                  f"{_r['cancelled']} jobs cancelled, "
                  f"remote={'yes' if _r['remote_cleaned'] else 'no'}")
        except Exception as e:
            print(f"Clean error: {e}")
    elif clean_phase_btn.value:
        try:
            _r = study.clean_phase(_phase, remote=True)
            print(f"CLEANED phase '{_phase}': {_r['local_deleted']} dirs deleted, "
                  f"{_r['cancelled']} jobs cancelled, "
                  f"remote={'yes' if _r['remote_cleaned'] else 'no'}")
        except Exception as e:
            print(f"Clean error: {e}")
    elif generate_btn.value:
        try:
            print(f"Generating phase '{_phase}'...")
            _dirs = study.prepare_phase(_phase)
            print(f"Generated {len(_dirs)} cases:")
            for _d in _dirs:
                print(f"  {_d.name}")
        except Exception as e:
            print(f"Generate error: {e}")
    elif upload_btn.value:
        _cases = [cid for cid, info in study.manifest.items()
                  if info.get("phase") == _phase]
        if not _cases:
            print(f"No cases for phase '{_phase}' — generate first")
        else:
            try:
                print(f"Uploading {len(_cases)} cases...")
                _results = study.upload_phase(_phase)
                _ok = sum(1 for v in _results.values() if v)
                print(f"Upload: {_ok}/{len(_results)} OK")
            except Exception as e:
                print(f"Upload error: {e}")
    elif submit_btn.value:
        try:
            print(f"Submitting phase '{_phase}' (auto-cancels old jobs)...")
            _jobs = study.submit_phase(_phase)
            print(f"Submitted {len(_jobs)} jobs:")
            for _j in _jobs:
                print(f"  {_j.name}: {_j.job_id}")
        except Exception as e:
            print(f"Submit error: {e}")
    elif cancel_btn.value:
        try:
            _n = study.cancel_phase(_phase)
            print(f"Cancelled {_n} jobs for phase '{_phase}'")
        except Exception as e:
            print(f"Cancel error: {e}")
    else:
        mo.output.replace(mo.md("*Select an action above.*"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A5. Monitor jobs (live SSH)

    Check the status of all submitted jobs via SSH to the HPC.
    Uses `qstat` to get real-time job state.
    """)
    return


@app.cell
def _(mo):
    monitor_btn = mo.ui.run_button(label="Refresh job status")
    monitor_btn
    return (monitor_btn,)


@app.cell
def _(CASES_DIR, json, mo, monitor_btn, pd, phase_selector, study):
    mo.stop(not monitor_btn.value, mo.md("*Click 'Refresh job status' above to query HPC via SSH.*"))

    _phase = phase_selector.value
    if study is None:
        print("Study not initialized")
    else:
        try:
            _status = study.monitor(_phase)

            # Flatten all job categories into a single list
            _jobs_info = []
            for _cat in ("running", "completed", "to_continue", "to_restart"):
                _cat_data = _status.get(_cat, {})
                for _j in _cat_data.get("jobs", []):
                    _j["state"] = _cat
                    _jobs_info.append(_j)

            if not _jobs_info:
                print("No submitted jobs found for this phase")
            else:
                _rows = []
                for _j in _jobs_info:
                    _cname = _j.get("case_name", "")
                    _meta = study.manifest.get(_cname, {})
                    # Build short description from manifest
                    _desc = ""
                    if _meta:
                        _res = _meta.get("resolution_m", "?")
                        _dom = _meta.get("domain_km", "?")
                        _dir = _meta.get("direction_deg", "?")
                        _sol = _meta.get("solver", "?")
                        _desc = f"{_res}m, {_dom}km, {_dir}°, {_sol}"
                        if _meta.get("thermal"):
                            _desc += ", thermal"
                        if _meta.get("canopy"):
                            _desc += ", canopy"
                        if _meta.get("precursor"):
                            _desc += ", precursor"
                    _rows.append({
                        "case": _cname,
                        "description": _desc,
                        "job_id": _j.get("job_id", ""),
                        "state": _j.get("state", "unknown"),
                        "last_time": _j.get("last_time", ""),
                        "end_time": _j.get("end_time", ""),
                        "warning": _j.get("warning", ""),
                    })
                _df_jobs = pd.DataFrame(_rows)
                mo.output.append(mo.ui.table(_df_jobs))

                _summary = _status.get("summary", {})
                print(f"\nSummary: {_status.get('total_jobs', 0)} jobs — "
                      f"{_summary.get('running', 0)} running, "
                      f"{_summary.get('completed', 0)} completed, "
                      f"{_summary.get('to_continue', 0)} to continue, "
                      f"{_summary.get('to_restart', 0)} to restart")

        except Exception as e:
            print(f"Monitor error: {e}")
            _status_file = CASES_DIR / "job_status.json"
            if _status_file.exists():
                with open(_status_file) as _f:
                    _cached = json.load(_f)
                print(f"\n(Showing cached status from {_status_file.name})")
                _jobs = _cached.get("jobs", [])
                for _j in _jobs:
                    print(f"  {_j.get('case_name', '?')}: {_j.get('state', '?')}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A6. Remote log tail

    SSH into HPC and read the last N lines of a solver log for a specific case.
    """)
    return


@app.cell
def _(mo, study):
    _cases = sorted(study.manifest.keys()) if study and study.manifest else []
    log_case_selector = mo.ui.dropdown(
        options=_cases,
        value=_cases[0] if _cases else None,
        label="Case to inspect",
    )
    log_lines_slider = mo.ui.slider(
        start=20, stop=200, step=20, value=50,
        label="Number of lines",
    )
    mo.hstack([log_case_selector, log_lines_slider])
    return log_case_selector, log_lines_slider


@app.cell
def _(log_case_selector, log_lines_slider, mo, study, subprocess):
    _cid = log_case_selector.value
    if not _cid or study is None:
        print("Select a case")
    else:
        _conn = study.hpc_cfg["connection"]
        _remote_base = _conn["remote_base_dir"]
        _host = _conn["hpc_host"]
        _user = _conn["username"]
        _n_lines = log_lines_slider.value

        # SSH ControlMaster options (reuse connection)
        _cm = ("-o ControlMaster=auto "
               "-o ControlPath=/tmp/hpc_foam_cm_%r@%h:%p "
               "-o ControlPersist=120s")

        # Try common log names — also show log.Allrun
        _cmd = (
            f"ssh {_cm} {_user}@{_host} "
            f"'cd {_remote_base}/{_cid} 2>/dev/null && "
            f"for f in log.Allrun log.*Foam* log.*foam* log.run; do "
            f"  if [ -f \"$f\" ]; then echo \"=== $f ===\"; tail -n {_n_lines} \"$f\"; break; fi; "
            f"done || echo \"Case directory not found on HPC\"'"
        )
        try:
            _result = subprocess.run(
                _cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            _output = _result.stdout.strip() or _result.stderr.strip() or "No output"
            mo.output.replace(mo.md(f"```\n{_output}\n```"))
        except subprocess.TimeoutExpired:
            print(f"SSH timeout — check connection to {_host}")
        except Exception as e:
            print(f"Error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A7. Download results from HPC

    Download completed case results (last timestep + logs) back to local.
    """)
    return


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="Download results", kind="warn")
    download_btn
    return (download_btn,)


@app.cell
def _(download_btn, mo, phase_selector, study):
    mo.stop(not download_btn.value, mo.md("*Click 'Download results' above to fetch from HPC.*"))

    _phase = phase_selector.value
    if study is None:
        print("Study not initialized")
    else:
        try:
            print(f"Downloading phase '{_phase}' results...")
            _results = study.download_phase(_phase)
            _ok = sum(1 for v in _results.values() if v)
            print(f"\nDownload complete: {_ok}/{len(_results)} successful")
            for _name, _success in _results.items():
                _sym = "OK" if _success else "FAIL"
                print(f"  {_name}: {_sym}")
        except Exception as e:
            print(f"Download error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A8. Restart failed / incomplete jobs

    Re-submit jobs that failed or did not converge within the allocated walltime.
    """)
    return


@app.cell
def _(CASES_DIR, json, mo):
    # Show failed/incomplete jobs
    _status_file = CASES_DIR / "job_status.json"
    _restart_candidates = []
    if _status_file.exists():
        with open(_status_file) as _f:
            _status = json.load(_f)
        for _j in _status.get("jobs", []):
            _state = _j.get("state", "")
            if _state in ("failed", "timeout", "needs_restart") or _j.get("needs_restart"):
                _restart_candidates.append(_j.get("case_name", "?"))

    if _restart_candidates:
        print(f"Cases needing restart: {len(_restart_candidates)}")
        for _c in _restart_candidates:
            print(f"  {_c}")
    else:
        print("No failed/incomplete jobs found (run monitor first)")

    restart_btn = mo.ui.run_button(label="Restart failed jobs", kind="danger")
    restart_btn
    return (restart_btn,)


@app.cell
def _(CASES_DIR, mo, restart_btn, study):
    mo.stop(not restart_btn.value, mo.md("*Click 'Restart failed jobs' above to re-submit.*"))

    if study is None:
        print("Study not initialized")
    else:
        _status_file = CASES_DIR / "job_status.json"
        if not _status_file.exists():
            print("No job_status.json — run monitor first")
        else:
            try:
                from hpc_foam import restart_failed_jobs_from_status
                config = study._get_hpc_config()
                restart_failed_jobs_from_status(
                    config=config,
                    status_file=_status_file,
                    new_end_time=study.n_iter,
                    confirm=False,
                    verbose=True,
                )
                print("Restart commands sent")
            except Exception as e:
                print(f"Restart error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A9. HPC queue overview

    Live `qstat` output for your user on the HPC.
    """)
    return


@app.cell
def _(mo):
    qstat_btn = mo.ui.run_button(label="Refresh qstat")
    qstat_btn
    return (qstat_btn,)


@app.cell
def _(mo, qstat_btn, study, subprocess):
    mo.stop(not qstat_btn.value, mo.md("*Click 'Refresh qstat' above to query the PBS queue.*"))

    if study is None:
        print("Study not initialized")
    else:
        _conn = study.hpc_cfg["connection"]
        _cmd = f"ssh {_conn['username']}@{_conn['hpc_host']} 'qstat -u {_conn['username']}'"
        try:
            _result = subprocess.run(
                _cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            _output = _result.stdout.strip() or "No jobs in queue"
            mo.output.replace(mo.md(f"```\n{_output}\n```"))
        except subprocess.TimeoutExpired:
            print("SSH timeout")
        except Exception as e:
            print(f"Error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## A10. HPC disk usage

    Check how much space convergence study cases occupy on the HPC.
    """)
    return


@app.cell
def _(mo, study, subprocess):
    if study is None:
        print("Study not initialized")
    else:
        _conn = study.hpc_cfg["connection"]
        _remote_base = _conn["remote_base_dir"]
        _cmd = (
            f"ssh {_conn['username']}@{_conn['hpc_host']} "
            f"'du -sh {_remote_base}/rheotool_* 2>/dev/null | sort -h; "
            f"echo \"---\"; "
            f"du -sh {_remote_base} 2>/dev/null'"
        )
        try:
            _result = subprocess.run(
                _cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            _output = _result.stdout.strip() or "No remote cases found"
            mo.output.replace(mo.md(f"```\n{_output}\n```"))
        except subprocess.TimeoutExpired:
            print("SSH timeout")
        except Exception as e:
            print(f"Error: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part B — Monitoring & Analysis

    Post-processing of completed simulations: residuals, validation metrics, profiles.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B1. Study overview

    Load the manifest and show all cases with their status.
    """)
    return


@app.cell
def _(CASES_DIR, MANIFEST_PATH, json, mo, pd):
    # Load manifest
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as _f:
            manifest = json.load(_f)
    else:
        manifest = {}
        print("No manifest found — generate a phase in Part A first")

    # Load submitted job info if available
    _submitted_jobs = {}
    for _jf in sorted(CASES_DIR.glob("submitted_jobs_*.json")):
        with open(_jf) as _f:
            _jdata = json.load(_f)
        for _j in _jdata.get("jobs", []):
            _submitted_jobs[_j["case_name"]] = _j.get("job_id", "?")

    # Load job status if available
    _job_status_file = CASES_DIR / "job_status.json"
    _job_statuses = {}
    if _job_status_file.exists():
        with open(_job_status_file) as _f:
            _status_data = json.load(_f)
        for _j in _status_data.get("jobs", []):
            _job_statuses[_j.get("case_name", "")] = _j.get("state", "unknown")

    # Build summary table
    def _build_rows():
        rows = []
        for cid, info in sorted(manifest.items()):
            case_dir = CASES_DIR / cid
            if cid in _job_statuses:
                st = _job_statuses[cid]
            elif cid in _submitted_jobs:
                st = "submitted"
            elif case_dir.exists():
                log_files = list(case_dir.glob("log.*Foam*")) + list(case_dir.glob("log.*foam*"))
                time_dirs = [d for d in case_dir.iterdir()
                             if d.is_dir() and d.name.replace('.', '').isdigit()
                             and float(d.name) > 0]
                if time_dirs:
                    st = "completed"
                elif log_files:
                    st = "running"
                else:
                    st = "generated"
            else:
                st = "planned"
            rows.append({
                "case_id": cid,
                "phase": info.get("phase", ""),
                "res_m": info.get("resolution_m", ""),
                "domain_km": info.get("domain_km", ""),
                "dir_deg": info.get("direction_deg", ""),
                "solver": info.get("solver", "simpleFoam"),
                "thermal": info.get("thermal", False),
                "canopy": info.get("canopy", False),
                "status": st,
            })
        return rows

    _built = _build_rows()
    df_manifest = pd.DataFrame(_built) if _built else pd.DataFrame()

    if not df_manifest.empty:
        _phase_counts = df_manifest.groupby(["phase", "status"]).size().unstack(fill_value=0)
        print("Phase summary:")
        print(_phase_counts.to_string())
        print()
        mo.output.append(mo.ui.table(df_manifest))
    else:
        print("No cases in manifest")
    return (manifest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B2. Residual convergence

    Parse OpenFOAM solver logs and plot initial residuals vs iteration.
    """)
    return


@app.cell
def _(manifest, mo):
    _case_ids = sorted(manifest.keys()) if manifest else []
    case_selector = mo.ui.multiselect(
        options=_case_ids,
        value=_case_ids[:4] if len(_case_ids) >= 4 else _case_ids,
        label="Select cases to plot residuals",
    )
    case_selector
    return (case_selector,)


@app.cell
def _(CASES_DIR, case_selector, mo, np, plt, re):

    def _plot():
        def parse_of_log(log_path):
            text = log_path.read_text(errors="replace")
            time_pat = re.compile(r"^Time = (\d+)", re.MULTILINE)
            res_pat = re.compile(
                r"Solving for (\w+),\s*Initial residual = ([\d.eE+-]+)", re.MULTILINE
            )
            times = []
            residuals = {}
            current_time = None
            current_res = {}
            for line in text.splitlines():
                tm = time_pat.match(line)
                if tm:
                    if current_time is not None and current_res:
                        times.append(current_time)
                        for f, v in current_res.items():
                            residuals.setdefault(f, []).append(v)
                    current_time = int(tm.group(1))
                    current_res = {}
                rm = res_pat.search(line)
                if rm and current_time is not None:
                    field = rm.group(1)
                    val = float(rm.group(2))
                    current_res[field] = val
            if current_time is not None and current_res:
                times.append(current_time)
                for f, v in current_res.items():
                    residuals.setdefault(f, []).append(v)
            return times, residuals

        selected = case_selector.value if case_selector.value else []
        if not selected:
            print("Select at least one case above")
            return
        all_res = {}
        for cid in selected:
            case_dir = CASES_DIR / cid
            log_candidates = (
                list(case_dir.glob("log.*Foam*"))
                + list(case_dir.glob("log.*foam*"))
                + list(case_dir.glob("log.run"))
            )
            if not log_candidates:
                print(f"  {cid}: no log file found")
                continue
            log_path = max(log_candidates, key=lambda p: p.stat().st_mtime)
            times, residuals = parse_of_log(log_path)
            if times:
                all_res[cid] = (times, residuals)
                print(f"  {cid}: {len(times)} iterations, fields={list(residuals.keys())}")
            else:
                print(f"  {cid}: log empty or unparseable")
        if not all_res:
            return
        all_fields = sorted(set(f for _, r in all_res.values() for f in r.keys()))
        key_fields = [f for f in ["Ux", "Uy", "Uz", "p", "k", "epsilon", "T"]
                      if f in all_fields]
        if not key_fields:
            key_fields = all_fields[:6]
        n_fields = len(key_fields)
        ncols = min(n_fields, 3)
        nrows = (n_fields + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False, sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_res), 1)))
        for fi, field in enumerate(key_fields):
            ax = axes[fi // ncols][fi % ncols]
            for ci, (cid, (times, residuals)) in enumerate(all_res.items()):
                if field in residuals:
                    ax.semilogy(times[:len(residuals[field])],
                                residuals[field],
                                color=colors[ci], lw=0.8, label=cid)
            ax.set_ylabel(field)
            ax.set_title(field, fontsize=10)
            ax.grid(True, alpha=0.3)
            if fi // ncols == nrows - 1:
                ax.set_xlabel("Iteration")
        for fi in range(n_fields, nrows * ncols):
            axes[fi // ncols][fi % ncols].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(all_res), 5), fontsize=8)
        fig.suptitle("Initial residuals", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B3. Validation metrics (CFD vs Perdigao obs)

    Load `at_masts.csv` for completed cases, compare against `perdigao_obs.zarr`.
    """)
    return


@app.cell
def _(CASES_DIR, DATA, ROOT, manifest, mo, np, pd):

    OBS_ZARR = DATA / 'raw' / 'perdigao_obs.zarr'

    import yaml
    _study_cfg_path = ROOT / 'configs' / 'convergence_study.yaml'
    if _study_cfg_path.exists():
        with open(_study_cfg_path) as _f:
            _study_cfg = yaml.safe_load(_f)
        TIMESTAMP = _study_cfg['study']['timestamp']
    else:
        TIMESTAMP = "2017-05-04T22:00"

    obs_data = None
    if OBS_ZARR.exists():
        try:
            from compare_cfd_obs import load_obs_snapshot
            obs_data = load_obs_snapshot(OBS_ZARR, TIMESTAMP)
            print(f"Obs loaded: {len(obs_data)} towers at {TIMESTAMP}")
        except Exception as e:
            print(f"Cannot load obs: {e}")
    else:
        print(f"Obs zarr not found: {OBS_ZARR}")

    from compare_cfd_obs import load_cfd_masts, compare

    def _compute_metrics():
        rows = []
        for cid, info in sorted(manifest.items()):
            case_dir = CASES_DIR / cid
            csv_path = case_dir / "at_masts.csv"
            if not csv_path.exists():
                csv_path = DATA / "cfd-database" / "perdigao" / cid / "at_masts.csv"
            if not csv_path.exists():
                continue
            try:
                cfd_rows = load_cfd_masts(csv_path)
            except Exception:
                continue
            if obs_data is None:
                continue
            matched = compare(cfd_rows, obs_data)
            n = len(matched["cfd_speed"])
            if n == 0:
                continue
            cfd_s = matched["cfd_speed"]
            obs_s = matched["obs_speed"]
            bias = float(np.mean(cfd_s - obs_s))
            rmse = float(np.sqrt(np.mean((cfd_s - obs_s) ** 2)))
            threshold = np.maximum(2.0, 0.3 * obs_s)
            hit_rate = float(np.mean(np.abs(cfd_s - obs_s) < threshold)) * 100
            rows.append({
                "case_id": cid,
                "phase": info.get("phase", ""),
                "res_m": info.get("resolution_m", ""),
                "domain_km": info.get("domain_km", ""),
                "dir_deg": info.get("direction_deg", ""),
                "solver": info.get("solver", ""),
                "N": n,
                "bias": round(bias, 2),
                "RMSE": round(rmse, 2),
                "HR%": round(hit_rate, 1),
            })
        return rows

    _mrows = _compute_metrics()
    df_metrics = pd.DataFrame(_mrows) if _mrows else pd.DataFrame()

    if not df_metrics.empty:
        print(f"\nValidation metrics for {len(df_metrics)} completed cases:")
        mo.output.append(mo.ui.table(df_metrics))
    else:
        print("No completed cases with at_masts.csv found yet")
    return df_metrics, obs_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B4. Mesh convergence — RMSE vs resolution
    """)
    return


@app.cell
def _(df_metrics, mo, np, plt):

    def _plot():
        if df_metrics is None or df_metrics.empty:
            print("No metrics available — waiting for completed runs")
            return None
        df_mesh = df_metrics[df_metrics["phase"] == "mesh_convergence"].copy()
        if df_mesh.empty:
            print("No mesh_convergence cases with metrics yet")
            return None
        directions = sorted(df_mesh["dir_deg"].unique())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        markers = {231: "o", 40: "s"}
        dir_labels = {231: "SW (231)", 40: "NE (40)"}
        for d in directions:
            df_d = df_mesh[df_mesh["dir_deg"] == d].sort_values("res_m")
            res = df_d["res_m"].values.astype(float)
            rmse = df_d["RMSE"].values.astype(float)
            bias = df_d["bias"].values.astype(float)
            m = markers.get(d, "^")
            lbl = dir_labels.get(d, f"{d}")
            ax1.plot(res, rmse, f"-{m}", ms=8, label=f"RMSE {lbl}")
            ax1.plot(res, np.abs(bias), f"--{m}", ms=6, alpha=0.6, label=f"|bias| {lbl}")
            for i in range(len(res) - 1):
                if rmse[i] > 0:
                    ratio = res[i] / res[i + 1] if res[i + 1] > 0 else 2.0
                    eps = abs(rmse[i + 1] - rmse[i]) / rmse[i]
                    gci = 1.25 * eps / (ratio ** 2 - 1)
                    ax2.bar(
                        f"{int(res[i])}->{int(res[i+1])}\n{lbl}",
                        gci * 100,
                        color="steelblue" if d == 231 else "coral",
                        edgecolor="k", linewidth=0.5,
                    )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        import matplotlib.ticker as mticker
        ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.set_xlabel("Resolution [m]")
        ax1.set_ylabel("Error [m/s]")
        ax1.set_title("(a) Mesh convergence")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        ax2.set_ylabel("GCI [%]")
        ax2.set_title("(b) Grid Convergence Index")
        ax2.axhline(5, color="green", ls="--", lw=0.8, label="5% threshold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Phase 1 — Mesh convergence", fontsize=12)
        fig.tight_layout()
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B5. Vertical wind profiles — CFD vs observations
    """)
    return


@app.cell
def _(CASES_DIR, DATA, manifest, mo):
    def _find_csv_cases():
        out = []
        for cid in sorted(manifest.keys()):
            csv = CASES_DIR / cid / "at_masts.csv"
            csv2 = DATA / "cfd-database" / "perdigao" / cid / "at_masts.csv"
            if csv.exists() or csv2.exists():
                out.append(cid)
        return out

    _cases_with_csv = _find_csv_cases()

    profile_selector = mo.ui.multiselect(
        options=_cases_with_csv,
        value=_cases_with_csv[:3] if len(_cases_with_csv) >= 3 else _cases_with_csv,
        label="Cases to overlay on profiles",
    )
    profile_selector
    return (profile_selector,)


@app.cell
def _(CASES_DIR, DATA, manifest, mo, np, obs_data, plt, profile_selector):

    def _plot():
        KEY_TOWERS = ["tse04", "tse09", "tse13"]
        selected_profiles = profile_selector.value if profile_selector.value else []
        if not selected_profiles:
            print("Select cases above")
            return
        if obs_data is None:
            print("Observations not loaded")
            return
        from compare_cfd_obs import load_cfd_masts
        multi_cfd = {}
        for cid in selected_profiles:
            csv_path = CASES_DIR / cid / "at_masts.csv"
            if not csv_path.exists():
                csv_path = DATA / "cfd-database" / "perdigao" / cid / "at_masts.csv"
            if csv_path.exists():
                info = manifest.get(cid, {})
                label = f"{cid} ({info.get('resolution_m', '?')}m)"
                multi_cfd[label] = load_cfd_masts(csv_path)
        active_towers = [t for t in KEY_TOWERS if t in obs_data]
        n_towers = len(active_towers)
        if n_towers == 0:
            print("No matching towers between obs and key towers")
            return
        if not multi_cfd:
            print("No at_masts.csv found for selected cases")
            return
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(multi_cfd), 1)))
        fig, axes = plt.subplots(1, n_towers, figsize=(4.5 * n_towers, 6), sharey=True)
        if n_towers == 1:
            axes = [axes]
        for idx, tid in enumerate(active_towers):
            ax = axes[idx]
            obs_h = np.array(obs_data[tid]["heights"])
            obs_spd = np.array(obs_data[tid]["speed"])
            ax.plot(obs_spd, obs_h, "ko-", ms=5, lw=2, label="Obs", zorder=10)
            for ci, (lbl, cfd_rows) in enumerate(multi_cfd.items()):
                cfd_h, cfd_spd = [], []
                for row in cfd_rows:
                    if row["tower_id"] == tid:
                        cfd_h.append(row["height_m"])
                        cfd_spd.append(row["speed"])
                if cfd_h:
                    order = np.argsort(cfd_h)
                    ax.plot(np.array(cfd_spd)[order], np.array(cfd_h)[order],
                            marker="s", ms=4, ls="--", color=colors[ci],
                            label=lbl, zorder=5)
            ax.set_xlabel("Wind speed [m/s]")
            if idx == 0:
                ax.set_ylabel("Height AGL [m]")
            ax.set_title(tid, fontweight="bold")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(multi_cfd) + 1, 5), fontsize=8)
        fig.suptitle("Vertical wind profiles — CFD vs Observations", fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B6. Domain sensitivity — RMSE vs domain size
    """)
    return


@app.cell
def _(df_metrics, mo, np, plt):

    def _plot():
        if df_metrics is None or df_metrics.empty:
            print("No metrics available")
            return
        df_dom = df_metrics[df_metrics["phase"] == "domain_sensitivity"].copy()
        if df_dom.empty:
            print("No domain_sensitivity cases with metrics yet")
            return
        directions = sorted(df_dom["dir_deg"].unique())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        markers = {231: "o", 40: "s"}
        dir_labels = {231: "SW (231)", 40: "NE (40)"}
        for d in directions:
            df_d = df_dom[df_dom["dir_deg"] == d].sort_values("domain_km")
            dom = df_d["domain_km"].values.astype(float)
            rmse = df_d["RMSE"].values.astype(float)
            bias = df_d["bias"].values.astype(float)
            m = markers.get(d, "^")
            lbl = dir_labels.get(d, f"{d}")
            ax.plot(dom, rmse, f"-{m}", ms=8, label=f"RMSE {lbl}")
            ax.plot(dom, np.abs(bias), f"--{m}", ms=6, alpha=0.6, label=f"|bias| {lbl}")
        ax.set_xlabel("Domain size [km]")
        ax.set_ylabel("Error [m/s]")
        ax.set_title("Phase 2 — Domain sensitivity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B7. Physics comparison — config A through E

    | Config | Solver | Coriolis | Canopy | Thermal | Precursor |
    |--------|--------|----------|--------|---------|-----------|
    | A | simpleFoam | - | - | - | - |
    | B | simpleFoam | yes | - | - | - |
    | C | simpleFoam | yes | yes | - | - |
    | D | BBSF | yes | yes | yes | - |
    | E | BBSF | yes | yes | yes | yes |
    """)
    return


@app.cell
def _(df_metrics, manifest, mo, np, plt):

    def _plot():
        if df_metrics is None or df_metrics.empty:
            print("No metrics available")
            return
        df_phys = df_metrics[df_metrics["phase"] == "physics_comparison"].copy()
        if df_phys.empty:
            print("No physics_comparison cases with metrics yet")
            return
        config_labels = {}
        for _, row in df_phys.iterrows():
            cid = row["case_id"]
            info = manifest.get(cid, {})
            notes = info.get("notes", "")
            if notes.startswith("config_"):
                config_labels[cid] = notes.replace("config_", "")
            else:
                config_labels[cid] = cid
        df_phys["config"] = df_phys["case_id"].map(config_labels)
        directions = sorted(df_phys["dir_deg"].unique())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        configs = sorted(df_phys["config"].unique())
        x = np.arange(len(configs))
        width = 0.35
        for di, d in enumerate(directions):
            df_d = df_phys[df_phys["dir_deg"] == d].set_index("config")
            rmse_vals = [df_d.loc[c, "RMSE"] if c in df_d.index else 0 for c in configs]
            hr_vals = [df_d.loc[c, "HR%"] if c in df_d.index else 0 for c in configs]
            offset = (di - 0.5) * width
            lbl = f"{int(d)}"
            ax1.bar(x + offset, rmse_vals, width, label=lbl, edgecolor="k", lw=0.5)
            ax2.bar(x + offset, hr_vals, width, label=lbl, edgecolor="k", lw=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.set_ylabel("RMSE [m/s]")
        ax1.set_title("(a) RMSE by physics config")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.set_ylabel("Hit Rate [%]")
        ax2.set_title("(b) Hit rate by physics config")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Phase 3 — Physics complexity comparison", fontsize=12)
        fig.tight_layout()
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B8. Scatter plot — CFD vs observed wind speed
    """)
    return


@app.cell
def _(CASES_DIR, DATA, manifest, mo):
    def _find_scatter_cases():
        out = []
        for cid in sorted(manifest.keys()):
            csv = CASES_DIR / cid / "at_masts.csv"
            csv2 = DATA / "cfd-database" / "perdigao" / cid / "at_masts.csv"
            if csv.exists() or csv2.exists():
                out.append(cid)
        return out

    _scatter_opts = _find_scatter_cases()

    scatter_selector = mo.ui.dropdown(
        options=_scatter_opts,
        value=_scatter_opts[0] if _scatter_opts else None,
        label="Case for scatter plot",
    )
    scatter_selector
    return (scatter_selector,)


@app.cell
def _(CASES_DIR, DATA, mo, np, obs_data, plt, scatter_selector):

    def _plot():
        cid = scatter_selector.value
        if not cid:
            print("Select a case above")
            return
        if obs_data is None:
            print("Observations not loaded")
            return
        from compare_cfd_obs import load_cfd_masts, compare
        csv_path = CASES_DIR / cid / "at_masts.csv"
        if not csv_path.exists():
            csv_path = DATA / "cfd-database" / "perdigao" / cid / "at_masts.csv"
        if not csv_path.exists():
            print(f"No at_masts.csv for {cid}")
            return
        cfd_rows = load_cfd_masts(csv_path)
        matched = compare(cfd_rows, obs_data)
        cfd_s = matched["cfd_speed"]
        obs_s = matched["obs_speed"]
        if len(cfd_s) == 0:
            print("No matched points")
            return
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(obs_s, cfd_s, s=30, alpha=0.7, edgecolors="k", linewidths=0.5)
        vmax = max(cfd_s.max(), obs_s.max()) * 1.15
        ax.plot([0, vmax], [0, vmax], "k--", lw=0.8, label="1:1")
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_xlabel("Observed [m/s]")
        ax.set_ylabel("CFD [m/s]")
        ax.set_aspect("equal")
        bias = float(np.mean(cfd_s - obs_s))
        rmse = float(np.sqrt(np.mean((cfd_s - obs_s) ** 2)))
        ax.text(0.05, 0.92,
                f"bias = {bias:+.2f} m/s\nRMSE = {rmse:.2f} m/s\nN = {len(cfd_s)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8))
        ax.set_title(f"CFD vs Obs — {cid}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    _fig = _plot()
    mo.output.append(_fig) if _fig is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B9. Summary — best config per phase
    """)
    return


@app.cell
def _(df_metrics, mo, pd):

    def _show_summary():
        if df_metrics is None or df_metrics.empty:
            print("No metrics available")
            return
        best_rows = []
        for phase in df_metrics["phase"].unique():
            df_p = df_metrics[df_metrics["phase"] == phase].copy()
            if df_p.empty:
                continue
            best = df_p.loc[df_p["RMSE"].idxmin()]
            best_rows.append({
                "phase": phase,
                "best_case": best["case_id"],
                "res_m": best["res_m"],
                "domain_km": best["domain_km"],
                "RMSE": best["RMSE"],
                "bias": best["bias"],
                "HR%": best["HR%"],
            })
        if best_rows:
            df_best = pd.DataFrame(best_rows)
            print("Best configuration per phase:")
            mo.output.append(mo.ui.table(df_best))

    _show_summary()
    return


if __name__ == "__main__":
    app.run()
