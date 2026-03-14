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
    # Module 2A — CFD Case Generator

    **Site**: Perdigão, Portugal
    **Solver**: simpleFoam + k-ε (OF2412 ESI)
    **Deployment**: kraken-sim (manifest-based)

    This notebook generates OpenFOAM case directories and the `campaign.yaml`
    manifest for kraken-sim deployment. It does NOT run CFD — that's handled
    by kraken-sim on HPC.

    ## Workflow
    1. Configure case parameters (direction, speed, stability, mesh)
    2. Generate single validation case OR batch campaign
    3. Use `kraken-sim smoke campaign.yaml` to test locally
    4. Use `kraken-sim load campaign.yaml` to deploy on HPC
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, '..')
    from pathlib import Path
    import yaml

    ROOT = Path('..').resolve()
    DATA = ROOT / 'data'
    CASES_DIR = DATA / 'campaign' / 'cases'
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROOT / 'configs' / 'sites' / 'perdigao.yaml') as _f:
        SITE_CFG = yaml.safe_load(_f)

    print(f"Site: {SITE_CFG['site']['name']}")
    print(f"Root: {ROOT}")
    print(f"Cases: {CASES_DIR}")
    return DATA, CASES_DIR, ROOT, SITE_CFG, yaml


@app.cell
def _(mo):
    # --- Case parameters form ---
    direction = mo.ui.slider(0, 350, step=10, value=220, label="Wind direction [°]")
    speed = mo.ui.slider(3, 20, step=1, value=10, label="Wind speed [m/s]")
    stability = mo.ui.dropdown(["neutral", "stable", "unstable"], value="neutral", label="Stability")
    resolution = mo.ui.dropdown([1000, 500, 250, 100], value=100, label="Resolution [m]")
    domain_km = mo.ui.slider(10, 50, step=5, value=25, label="Domain [km]")
    context_cells = mo.ui.dropdown([1, 3, 5], value=3, label="Context cells")
    ncpus = mo.ui.slider(1, 24, step=1, value=12, label="CPUs")
    canopy = mo.ui.switch(value=False, label="Canopy drag")
    coriolis = mo.ui.switch(value=True, label="Coriolis force")

    form = mo.vstack([
        mo.md("### Case parameters"),
        mo.hstack([direction, speed, stability]),
        mo.hstack([resolution, domain_km, context_cells]),
        mo.hstack([ncpus, canopy, coriolis]),
    ])
    form
    return canopy, context_cells, coriolis, direction, domain_km, ncpus, resolution, speed, stability


@app.cell
def _(CASES_DIR, SITE_CFG, canopy, context_cells, coriolis, direction, domain_km, mo, ncpus, resolution, speed, stability):
    # --- Generate single validation case ---
    generate_btn = mo.ui.run_button(label="Generate single case")
    generate_btn

    if generate_btn.value:
        sys_path_hack = __import__('sys')
        sys_path_hack.path.insert(0, str(__import__('pathlib').Path('..').resolve() / 'services' / 'module2a-cfd'))

        from generate_mesh import generate_mesh
        from generate_campaign import build_parametric_inflow
        import json

        case_id = f"val_{direction.value:03d}deg_{speed.value:02d}ms_{stability.value}"
        case_dir = CASES_DIR / case_id

        # Build inflow
        inflow = build_parametric_inflow(
            speed_ms=speed.value,
            direction_deg=direction.value,
            stability=stability.value,
        )

        # Write inflow JSON
        inflow_json = CASES_DIR / "inflow_profiles" / f"inflow_{case_id}.json"
        inflow_json.parent.mkdir(parents=True, exist_ok=True)
        with open(inflow_json, "w") as f:
            json.dump(inflow, f, indent=2)

        # Detect SRTM
        srtm = __import__('pathlib').Path('..').resolve() / 'data' / 'raw' / 'srtm_perdigao_30m.tif'

        geom = generate_mesh(
            site_cfg=SITE_CFG,
            resolution_m=resolution.value,
            context_cells=context_cells.value,
            output_dir=case_dir,
            srtm_tif=srtm if srtm.exists() else None,
            inflow_json=inflow_json,
            domain_km=domain_km.value,
            solver_name="simpleFoam",
        )

        mo.output.replace(mo.md(f"""
        **Case generated:** `{case_dir}`
        - Resolution: {resolution.value} m
        - Domain: {domain_km.value}×{domain_km.value} km, context={context_cells.value}
        - Cells (blockMesh): {geom['n_x']}×{geom['n_y']}×{geom['n_z']} = {geom['n_x']*geom['n_y']*geom['n_z']:,}
        - Wind: {direction.value}° @ {speed.value} m/s ({stability.value})
        """))
    return


@app.cell
def _(CASES_DIR, ROOT, mo, yaml):
    # --- Batch campaign generation ---
    campaign_btn = mo.ui.run_button(label="Generate campaign from YAML")
    campaign_btn

    if campaign_btn.value:
        sys_path_hack = __import__('sys')
        sys_path_hack.path.insert(0, str(ROOT / 'services' / 'module2a-cfd'))

        from generate_campaign import generate_campaign

        config_path = ROOT / 'configs' / 'training' / 'cfd_grid.yaml'
        site_path = ROOT / 'configs' / 'sites' / 'perdigao.yaml'
        srtm = ROOT / 'data' / 'raw' / 'srtm_perdigao_30m.tif'

        if config_path.exists():
            cases = generate_campaign(
                config_path=config_path,
                output_base=CASES_DIR,
                site_config_path=site_path,
                prefix="prd",
                srtm_tif=srtm if srtm.exists() else None,
            )
            mo.output.replace(mo.md(f"**Campaign generated:** {len(cases)} cases in `{CASES_DIR}`"))
        else:
            mo.output.replace(mo.md(f"Config not found: `{config_path}`"))
    return


@app.cell
def _(CASES_DIR, mo, yaml):
    # --- Display campaign.yaml ---
    manifest = CASES_DIR / "campaign.yaml"
    if manifest.exists():
        with open(manifest) as f:
            content = f.read()
        # Show first 80 lines
        lines = content.split('\n')
        preview = '\n'.join(lines[:80])
        if len(lines) > 80:
            preview += f"\n\n... ({len(lines) - 80} more lines)"
        mo.output.replace(mo.md(f"### campaign.yaml\n```yaml\n{preview}\n```"))
    else:
        mo.output.replace(mo.md("*No campaign.yaml yet — generate a campaign first.*"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Next steps

    ```bash
    # 1. Smoke test (local Docker, 1 case)
    kraken-sim smoke data/campaign/cases/campaign.yaml

    # 2. Deploy on HPC
    kraken-sim load data/campaign/cases/campaign.yaml

    # 3. Monitor
    kraken-sim status data/campaign/cases/campaign.yaml

    # 4. Download results
    kraken-sim pull data/campaign/cases/campaign.yaml
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
