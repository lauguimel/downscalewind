import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 2A — Validation Matrix (Full Factorial)

    **Site**: Perdigão, Portugal (canonical case: 217°, 7.85 m/s, neutral)
    **Domain**: 25×25 km centred on obs masts
    **Solver**: OF2412 ESI → kraken-sim deployment

    ## Goal
    Evaluate the impact of 5 factors on CFD accuracy vs Perdigão observations:
    1. **Solver**: simpleFoam vs buoyantSimpleFoam
    2. **Coriolis**: on vs off
    3. **Canopy drag**: on vs off
    4. **Resolution**: 500 m, 250 m, 100 m
    5. **fvSchemes**: robust (upwind k/ε) vs accurate (linearUpwind k/ε)

    Full factorial: 2 × 2 × 2 × 3 × 2 = **48 cases**
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).parent.parent.resolve() if "__file__" in dir() else Path("..").resolve()
    DATA = ROOT / "data"
    CASES_DIR = DATA / "campaign" / "validation"
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    _cfd_path = str(ROOT / "services" / "module2a-cfd")
    if _cfd_path not in sys.path:
        sys.path.insert(0, _cfd_path)

    import yaml
    with open(ROOT / "configs" / "sites" / "perdigao.yaml") as _fh:
        SITE_CFG = yaml.safe_load(_fh)

    SRTM_TIF = ROOT / "data" / "raw" / "srtm_perdigao_30m.tif"

    print(f"Site: {SITE_CFG['site']['name']}")
    print(f"Cases dir: {CASES_DIR}")
    print(f"SRTM: exists={SRTM_TIF.exists()}")
    return CASES_DIR, DATA, ROOT, SITE_CFG, SRTM_TIF


@app.cell
def _(mo):
    mo.md("## Validation matrix (full factorial)")
    return


@app.cell
def _(mo):
    import itertools
    import pandas as pd

    # --- Fixed meteorological conditions (canonical Perdigão case) ---
    DIRECTION_DEG = 217.0
    SPEED_MS = 7.85
    STABILITY = "neutral"
    DOMAIN_KM = 25.0
    CONTEXT_CELLS = 1  # 25 km domain, no buffer needed

    # --- Factors to vary ---
    solvers = ["simpleFoam", "buoyantSimpleFoam"]
    coriolis_opts = [True, False]
    canopy_opts = [True, False]
    resolutions_m = [500, 250, 100]
    fvschemes_opts = ["robust", "accurate"]

    # --- Full factorial ---
    combos = list(itertools.product(solvers, coriolis_opts, canopy_opts, resolutions_m, fvschemes_opts))

    matrix = pd.DataFrame(combos, columns=["solver", "coriolis", "canopy", "resolution_m", "fvschemes"])
    matrix.insert(0, "case_id", [
        f"v_{row.solver[:2]}_{('cor' if row.coriolis else 'noc')}_{('can' if row.canopy else 'nca')}_{row.resolution_m}m_{row.fvschemes[:3]}"
        for row in matrix.itertuples()
    ])
    matrix["direction_deg"] = DIRECTION_DEG
    matrix["speed_ms"] = SPEED_MS
    matrix["stability"] = STABILITY
    matrix["domain_km"] = DOMAIN_KM

    mo.md(f"**{len(matrix)} cases** in full factorial design")

    return (
        CONTEXT_CELLS, DIRECTION_DEG, DOMAIN_KM, SPEED_MS, STABILITY,
        canopy_opts, coriolis_opts, fvschemes_opts, matrix, resolutions_m, solvers,
    )


@app.cell
def _(matrix, mo):
    mo.ui.table(matrix, selection=None)
    return


@app.cell
def _(mo):
    generate_btn = mo.ui.run_button(label="Generate all 48 cases + campaign.yaml")
    generate_btn
    return (generate_btn,)


@app.cell
def _(CASES_DIR, CONTEXT_CELLS, SITE_CFG, SRTM_TIF, generate_btn, matrix, mo):
    mo.stop(not generate_btn.value, mo.md("*Click the button above to generate all cases.*"))

    import json
    import logging
    from generate_mesh import generate_mesh
    from generate_campaign import build_parametric_inflow, CampaignCase, _patch_fvschemes_accurate, _patch_control_dict, _render_run_pbs

    logging.basicConfig(level=logging.WARNING)

    generated = []
    for row in matrix.itertuples():
        case_dir = CASES_DIR / row.case_id
        if case_dir.exists():
            generated.append(row.case_id)
            continue

        # Build inflow
        _inflow = build_parametric_inflow(
            speed_ms=row.speed_ms,
            direction_deg=row.direction_deg,
            stability=row.stability,
        )
        _inflow_json = CASES_DIR / "inflow_profiles" / f"inflow_{row.case_id}.json"
        _inflow_json.parent.mkdir(parents=True, exist_ok=True)
        with open(_inflow_json, "w") as _fh:
            json.dump(_inflow, _fh, indent=2)

        # Generate case
        _thermal = row.solver == "buoyantSimpleFoam"
        _boussinesq = row.solver == "buoyantSimpleFoam"

        generate_mesh(
            site_cfg=SITE_CFG,
            resolution_m=float(row.resolution_m),
            context_cells=CONTEXT_CELLS,
            output_dir=case_dir,
            srtm_tif=SRTM_TIF if SRTM_TIF.exists() else None,
            inflow_json=_inflow_json,
            domain_km=row.domain_km,
            solver_name=row.solver,
            thermal=_thermal,
            coriolis=row.coriolis,
            canopy_enabled=row.canopy,
            boussinesq=_boussinesq,
        )

        # Patch iterations
        _patch_control_dict(case_dir, 2000, 200)

        # Patch fvSchemes if accurate
        if row.fvschemes == "accurate":
            _patch_fvschemes_accurate(case_dir)

        # Render run.pbs
        _case = CampaignCase(
            case_id=row.case_id, solver=row.solver,
            direction_deg=row.direction_deg, speed_ms=row.speed_ms,
            stability=row.stability, thermal=_thermal,
            coriolis=row.coriolis, canopy=row.canopy,
            resolution_m=row.resolution_m, domain_km=row.domain_km,
            context_cells=CONTEXT_CELLS, n_refine_levels=2,
            boussinesq=_boussinesq, fvschemes_variant=row.fvschemes,
        )
        _render_run_pbs(case_dir, _case)

        generated.append(row.case_id)

    mo.md(f"**{len(generated)} cases generated** in `{CASES_DIR}`")
    return (generated,)


@app.cell
def _(CASES_DIR, SITE_CFG, generated, matrix, mo):
    # --- Write campaign.yaml manifest ---
    import yaml as _yaml

    _cases_list = []
    for row in matrix.itertuples():
        if row.case_id in generated:
            _cases_list.append({
                "id": row.case_id,
                "dir": row.case_id,
                "script": "run.pbs",
                "ncpus": 12,
                "tags": {
                    "solver": row.solver,
                    "direction_deg": float(row.direction_deg),
                    "speed_ms": float(row.speed_ms),
                    "stability": row.stability,
                    "resolution_m": int(row.resolution_m),
                    "coriolis": bool(row.coriolis),
                    "canopy": bool(row.canopy),
                    "fvschemes": row.fvschemes,
                },
            })

    # Remote dir: geohash 8 chars (≈±20 m precision)
    from generate_campaign import _geohash_encode
    _ghash = _geohash_encode(SITE_CFG["site"]["coordinates"]["latitude"],
                              SITE_CFG["site"]["coordinates"]["longitude"], precision=8)
    _remote_dir = f"/home/maitreje/campaigns/{_ghash}"

    _manifest = {
        "name": "perdigao_validation_factorial",
        "hpc": {
            "host": "aqua.qut.edu.au",
            "username": "maitreje",
            "remote_base_dir": _remote_dir,
        },
        "parser": "openfoam",
        "inject_monitoring": True,
        "monitoring_fields": ["U", "p", "k", "epsilon"],
        "cases": _cases_list,
    }

    _manifest_path = CASES_DIR / "campaign.yaml"
    with open(_manifest_path, "w") as _fh:
        _yaml.dump(_manifest, _fh, default_flow_style=False, sort_keys=False)

    mo.md(f"**campaign.yaml** written: `{_manifest_path}` ({len(_cases_list)} cases)")
    return


@app.cell
def _(CASES_DIR, mo):
    _manifest = CASES_DIR / "campaign.yaml"
    if _manifest.exists():
        _content = _manifest.read_text()
        _lines = _content.split("\n")
        _preview = "\n".join(_lines[:60])
        if len(_lines) > 60:
            _preview += f"\n\n... ({len(_lines) - 60} more lines)"
        mo.md(f"### campaign.yaml preview\n```yaml\n{_preview}\n```")
    else:
        mo.md("*No campaign.yaml yet.*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Next steps

    ```bash
    # 1. Smoke test (1 case, local Docker OF2412)
    kraken-sim smoke data/campaign/validation/campaign.yaml

    # 2. Deploy all 48 cases on HPC
    kraken-sim load data/campaign/validation/campaign.yaml

    # 3. Monitor convergence
    kraken-sim status data/campaign/validation/campaign.yaml

    # 4. Download results → Parquet
    kraken-sim pull data/campaign/validation/campaign.yaml
    ```

    After results are downloaded, use `compare_cfd_obs.py` to evaluate
    each case against Perdigão observations and build the sensitivity table.
    """)
    return


if __name__ == "__main__":
    app.run()
