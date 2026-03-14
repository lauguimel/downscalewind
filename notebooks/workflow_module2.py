import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 2A — Mesh Convergence Study (sf vs bbsf)

    **Site**: Perdigão, Portugal — 2 observed IOP conditions
    **Domain**: 5×5 km centred on obs masts
    **Solver**: OF2412 ESI → kraken-sim deployment

    ## Goal
    Grid convergence study comparing **simpleFoam** vs **buoyantSimpleFoam**
    on 4 resolutions (250, 100, 50, 30 m) against Perdigão observations.

    Fixed physics: Coriolis on, canopy drag on, fvSchemes robust (upwind k/ε).

    2 solvers × 4 resolutions × 2 meteo = **16 cases**
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).parent.parent.resolve() if "__file__" in dir() else Path("..").resolve()
    DATA = ROOT / "data"
    CASES_DIR = DATA / "campaign" / "convergence_sf_bbsf"
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
    mo.md("## Convergence matrix (sf vs bbsf × 4 resolutions × 2 meteo)")
    return


@app.cell
def _(mo):
    import itertools
    import pandas as pd

    DOMAIN_KM = 5.0
    CONTEXT_CELLS = 1

    # --- Two observed meteo conditions from Perdigão IOP ---
    meteo_cases = [
        {"label": "sw", "direction_deg": 228.7, "speed_ms": 10.46, "stability": "neutral",
         "timestamp": "2017-05-11T06:00", "description": "SW dominant, canonical"},
        {"label": "east", "direction_deg": 91.7, "speed_ms": 7.88, "stability": "neutral",
         "timestamp": "2017-05-23T18:00", "description": "Easterly, perpendicular to ridges"},
    ]

    # --- Fixed physics ---
    CORIOLIS = True
    CANOPY = True
    FVSCHEMES = "robust"

    # --- Factors to vary: solver × resolution ---
    solvers = ["simpleFoam", "buoyantSimpleFoam"]
    resolutions_m = [250, 100, 50, 30]

    # --- Convergence matrix: meteo × solver × resolution ---
    rows = []
    for meteo in meteo_cases:
        for solver, res in itertools.product(solvers, resolutions_m):
            _tag = "sf" if solver == "simpleFoam" else "bbsf"
            case_id = f"{meteo['label']}_{_tag}_{res}m"
            rows.append({
                "case_id": case_id,
                "meteo": meteo["label"],
                "timestamp": meteo["timestamp"],
                "direction_deg": meteo["direction_deg"],
                "speed_ms": meteo["speed_ms"],
                "stability": meteo["stability"],
                "solver": solver,
                "coriolis": CORIOLIS,
                "canopy": CANOPY,
                "resolution_m": res,
                "fvschemes": FVSCHEMES,
                "domain_km": DOMAIN_KM,
            })

    matrix = pd.DataFrame(rows)

    mo.md(f"**{len(matrix)} cases** — 2 meteo × 2 solvers × {len(resolutions_m)} resolutions")
    return CONTEXT_CELLS, DOMAIN_KM, matrix


@app.cell
def _(matrix, mo):
    mo.ui.table(matrix, selection=None)
    return


@app.cell
def _(mo):
    generate_btn = mo.ui.run_button(label="Generate all 16 cases + campaign.yaml")
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
    for _row in matrix.itertuples():
        case_dir = CASES_DIR / _row.case_id
        if case_dir.exists():
            generated.append(_row.case_id)
            continue

        # Build inflow
        _inflow = build_parametric_inflow(
            speed_ms=_row.speed_ms,
            direction_deg=_row.direction_deg,
            stability=_row.stability,
        )
        _inflow_json = CASES_DIR / "inflow_profiles" / f"inflow_{_row.case_id}.json"
        _inflow_json.parent.mkdir(parents=True, exist_ok=True)
        with open(_inflow_json, "w") as _fh:
            json.dump(_inflow, _fh, indent=2)

        # Generate case
        _thermal = _row.solver == "buoyantSimpleFoam"
        _boussinesq = _row.solver == "buoyantSimpleFoam"

        generate_mesh(
            site_cfg=SITE_CFG,
            resolution_m=float(_row.resolution_m),
            context_cells=CONTEXT_CELLS,
            output_dir=case_dir,
            srtm_tif=SRTM_TIF if SRTM_TIF.exists() else None,
            inflow_json=_inflow_json,
            domain_km=_row.domain_km,
            solver_name=_row.solver,
            thermal=_thermal,
            coriolis=_row.coriolis,
            canopy_enabled=_row.canopy,
            boussinesq=_boussinesq,
        )

        # Patch iterations
        _patch_control_dict(case_dir, 2000, 200)

        # Patch fvSchemes if accurate
        if _row.fvschemes == "accurate":
            _patch_fvschemes_accurate(case_dir)

        # Render run.pbs
        _case = CampaignCase(
            case_id=_row.case_id, solver=_row.solver,
            direction_deg=_row.direction_deg, speed_ms=_row.speed_ms,
            stability=_row.stability, thermal=_thermal,
            coriolis=_row.coriolis, canopy=_row.canopy,
            resolution_m=_row.resolution_m, domain_km=_row.domain_km,
            context_cells=CONTEXT_CELLS, n_refine_levels=2,
            boussinesq=_boussinesq, fvschemes_variant=_row.fvschemes,
        )
        _render_run_pbs(case_dir, _case)

        generated.append(_row.case_id)

    mo.md(f"**{len(generated)} cases generated** in `{CASES_DIR}`")
    return (generated,)


@app.cell
def _(CASES_DIR, SITE_CFG, generated, matrix, mo):
    # --- Write campaign.yaml via CampaignBuilder ---
    from generate_campaign import _geohash_encode
    from kraken_sim.schema import CampaignBuilder

    _ghash = _geohash_encode(SITE_CFG["site"]["coordinates"]["latitude"],
                              SITE_CFG["site"]["coordinates"]["longitude"], precision=8)
    _remote_dir = f"/home/maitreje/campaigns/{_ghash}"

    _builder = CampaignBuilder(
        name="perdigao_convergence_sf_bbsf",
        hpc_host="aqua.qut.edu.au",
        hpc_username="maitreje",
        hpc_remote_dir=_remote_dir,
        monitoring_fields=["U", "p", "k", "epsilon"],
        auto_group=["solver", "resolution_m"],
    )

    for _row in matrix.itertuples():
        if _row.case_id in generated:
            _builder.add_case(
                case_id=_row.case_id,
                case_dir=_row.case_id,
                ncpus=12,
                tags={
                    "solver": _row.solver,
                    "direction_deg": float(_row.direction_deg),
                    "speed_ms": float(_row.speed_ms),
                    "stability": _row.stability,
                    "resolution_m": int(_row.resolution_m),
                    "coriolis": bool(_row.coriolis),
                    "canopy": bool(_row.canopy),
                    "fvschemes": _row.fvschemes,
                },
            )

    _manifest_path = CASES_DIR / "campaign.yaml"
    _builder.write_yaml(_manifest_path)

    mo.md(f"**campaign.yaml** written: `{_manifest_path}` ({len(generated)} cases)")
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
    kraken-sim smoke data/campaign/convergence_sf_bbsf/campaign.yaml

    # 2. Deploy all 16 cases on HPC
    kraken-sim load data/campaign/convergence_sf_bbsf/campaign.yaml

    # 3. Monitor convergence
    kraken-sim status data/campaign/convergence_sf_bbsf/campaign.yaml

    # 4. Download results → Parquet
    kraken-sim pull data/campaign/convergence_sf_bbsf/campaign.yaml
    ```

    After results are downloaded, use `compare_cfd_obs.py` to evaluate
    each case against Perdigão observations and build the sensitivity table.
    """)
    return


if __name__ == "__main__":
    app.run()
