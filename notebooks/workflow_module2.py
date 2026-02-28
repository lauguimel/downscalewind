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
    # Module 2A — CFD Pipeline: Validation-First

    **Site**: Perdigão, Portugal (IOP 2017)

    This notebook orchestrates the full Module 2A pipeline:
    1. Baseline ERA5 error (before any downscaling)
    2. Canonical case selection (SW, neutral, ~10 m/s)
    3. Terrain visualization at 4 resolutions
    4. Mesh generation + checkMesh
    5. Inlet profile reconstruction (ERA5 → 3-layer ABL)
    6. OpenFOAM run (buoyantSimpleFoam, Docker ESI v2412)
    7. CFD results visualization (cross-section, plan view)
    8. CFD vs observations comparison
    9. Mesh convergence study

    **Container**: `opencfd/openfoam-default:v2412` (ESI branch)
    **Reference**: Neunaber et al. (WES 2023) — OpenFOAM Perdigão RANS
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, '..')  # project root
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import yaml
    ROOT = Path('..').resolve()
    DATA = ROOT / 'data'
    # Project paths
    FIGURES = ROOT / 'figures'
    FIGURES.mkdir(exist_ok=True)
    with open(ROOT / 'configs' / 'sites' / 'perdigao.yaml') as _f:
        SITE_CFG = yaml.safe_load(_f)
    SITE_LAT = SITE_CFG['site']['coordinates']['latitude']
    # Site config
    SITE_LON = SITE_CFG['site']['coordinates']['longitude']
    print(f"Site: {SITE_CFG['site']['name']} ({SITE_LAT:.3f}°N, {SITE_LON:.3f}°E)")
    print(f'Root: {ROOT}')
    return DATA, FIGURES, ROOT, SITE_CFG, SITE_LAT, SITE_LON, np, plt, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 0 — Baseline ERA5 error (no downscaling)

    Quantifies how well ERA5 alone predicts wind at the Perdigão masts.
    This is the **zero reference**: any downscaling method must beat this.
    """)
    return


@app.cell
def _(DATA, ROOT):
    # Run baseline ERA5 comparison
    # (requires era5_perdigao.zarr and perdigao_obs.zarr)
    ERA5_ZARR = DATA / 'raw' / 'era5_perdigao.zarr'
    OBS_ZARR = DATA / 'raw' / 'perdigao_obs.zarr'
    if ERA5_ZARR.exists() and OBS_ZARR.exists():
        import subprocess
        _result = subprocess.run(['python', str(ROOT / 'services' / 'validation' / 'baseline_era5.py'), '--era5', str(ERA5_ZARR), '--obs', str(OBS_ZARR), '--site', 'perdigao', '--output-dir', str(DATA / 'processed')], capture_output=True, text=True)
        print(_result.stdout[-3000:])
        if _result.returncode != 0:
            print('STDERR:', _result.stderr[-1000:])
    else:
        print(f'ERA5 zarr: {ERA5_ZARR} exists={ERA5_ZARR.exists()}')
        print(f'Obs  zarr: {OBS_ZARR} exists={OBS_ZARR.exists()}')
        print('Run ingestion scripts first (ingest_era5.py, ingest_perdigao_obs.py)')
    return ERA5_ZARR, subprocess


@app.cell
def _(DATA, mo):
    # Display baseline summary
    import pandas as pd
    summary_csv = DATA / 'processed' / 'baseline_era5_summary.csv'
    if summary_csv.exists():
        _df = pd.read_csv(summary_csv)
        mo.output.replace(mo.ui.table(_df.round(3)))
        print(f"\nOverall RMSE(|u|) = {_df['rmse_speed'].mean():.2f} m/s")
    else:
        print(f'Summary not found: {summary_csv}')
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 1 — Canonical case selection

    Select timestamps from ERA5 IOP (May–June 2017) that satisfy:
    - Direction SW ≈ 220° (dominant Perdigão direction)
    - Speed 7–14 m/s at 850 hPa
    - Stability neutral (|Ri_b| < 0.10)
    """)
    return


@app.cell
def _(DATA, ERA5_ZARR, FIGURES, ROOT, mo, subprocess):
    CASES_YAML = DATA / 'processed' / 'validation_cases.yaml'
    if ERA5_ZARR.exists():
        _result = subprocess.run(['python', str(ROOT / 'services' / 'module2a-cfd' / 'select_validation_cases.py'), '--era5', str(ERA5_ZARR), '--site', 'perdigao', '--output', str(CASES_YAML), '--plot'], capture_output=True, text=True)
        print(_result.stdout[-2000:])
        if _result.returncode != 0:
            print('STDERR:', _result.stderr[-500:])
    else:
        print('ERA5 zarr not found — run ingest_era5.py first')
    wind_rose = FIGURES / 'wind_rose_era5_iop.png'
    if wind_rose.exists():
        mo.output.append(mo.image(src=str(wind_rose)))
    return (CASES_YAML,)


@app.cell
def _(CASES_YAML, yaml):
    # Load and display canonical case
    if CASES_YAML.exists():
        with open(CASES_YAML) as _f:
            _cases = yaml.safe_load(_f)
        canonical = _cases.get('canonical', {})
        print('Canonical case:')
        for _k, _v in canonical.items():
            print(f'  {_k}: {_v}')
        CANONICAL_ID = canonical.get('timestamp', '2017-05-15T12:00')
    else:
        CANONICAL_ID = '2017-05-15T12:00'
        print(f'Using fallback canonical ID: {CANONICAL_ID}')
    print(f'\nCANONICAL_ID = {CANONICAL_ID}')  # fallback
    return (CANONICAL_ID,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 2 — SRTM Terrain

    Visualize the terrain at 4 resolutions (10 km → 500 m).
    """)
    return


@app.cell
def _(DATA, FIGURES, mo):
    SRTM_TIF = DATA / 'raw' / 'srtm_perdigao_30m.tif'
    from services.validation.plot_terrain_refinement import plot_terrain_refinement
    terrain_fig = FIGURES / 'terrain_refinement_4panel.png'
    plot_terrain_refinement(srtm_tif=SRTM_TIF if SRTM_TIF.exists() else None, output_path=terrain_fig)
    mo.output.replace(mo.image(src=str(terrain_fig)))
    return (SRTM_TIF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 3 — Mesh generation + checkMesh

    Generate the OpenFOAM case directory and run checkMesh.

    Start with the **pipeline test** (context_cells=1, 25×25 km, blockMesh only)
    to validate the full pipeline in <1 minute before expensive runs.
    """)
    return


@app.cell
def _(DATA, SITE_CFG, SRTM_TIF):
    from services.module2a_cfd.generate_mesh import generate_mesh
    CASE_DIR_TEST = DATA / 'cases' / 'pipeline_test_1km_1x1'
    # --- Pipeline test: context_cells=1, resolution=1km ---
    geom = generate_mesh(site_cfg=SITE_CFG, resolution_m=1000.0, context_cells=1, output_dir=CASE_DIR_TEST, srtm_tif=SRTM_TIF if SRTM_TIF.exists() else None)
    print('Domain geometry:')
    for _k, _v in geom.items():
        print(f'  {_k}: {_v}')
    print(f"\nTotal cells: {geom['n_x'] * geom['n_y'] * geom['n_z']:,}")  # pipeline test: no buffer
    return CASE_DIR_TEST, generate_mesh


@app.cell
def _(CASE_DIR_TEST):
    # Run blockMesh + checkMesh via Docker
    from services.module2a_cfd.openfoam_runner import OpenFOAMRunner
    _runner = OpenFOAMRunner(CASE_DIR_TEST, n_cores=1)
    print('Running blockMesh...')
    _runner.block_mesh()
    print('Running checkMesh...')
    _quality = _runner.check_mesh()
    print(f'\nMesh quality:')
    print(f'  cells         = {_quality.n_cells:,}')
    print(f'  maxNonOrtho   = {_quality.max_non_ortho:.1f}°')
    print(f'  maxSkewness   = {_quality.max_skewness:.2f}')
    print(f'  maxAspectRatio= {_quality.max_aspect_ratio:.1f}')
    print(f'  ok            = {_quality.ok}')
    return (OpenFOAMRunner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 4 — Inlet profile reconstruction

    Reconstruct a 3-layer ABL profile from ERA5:
    - 0–100 m: log law + Monin-Obukhov
    - 100 m–2 km: cubic spline
    - >2 km: ERA5 direct
    """)
    return


@app.cell
def _(CANONICAL_ID, DATA, ERA5_ZARR, SITE_LAT, SITE_LON):
    INFLOW_JSON = DATA / 'processed' / 'inflow' / f"{CANONICAL_ID.replace(':', '_')}.json"

    if ERA5_ZARR.exists():
        from services.module2a_cfd.prepare_inflow import prepare_inflow
    
        profile = prepare_inflow(
            era5_zarr=ERA5_ZARR,
            timestamp=CANONICAL_ID,
            site_lat=SITE_LAT,
            site_lon=SITE_LON,
            z0_tif=DATA / 'raw' / 'z0_perdigao.tif' if (DATA/'raw'/'z0_perdigao.tif').exists() else None,
            output_json=INFLOW_JSON,
        )
    
        print(f"u_hub  = {profile['u_hub']:.2f} m/s")
        print(f"u_star = {profile['u_star']:.3f} m/s")
        print(f"z0_eff = {profile['z0_eff']:.4f} m")
        print(f"Ri_b   = {profile['Ri_b']:.3f}")
        print(f"T_ref  = {profile['T_ref']:.1f} K")
        print(f"flowDir = ({profile['flowDir_x']:.3f}, {profile['flowDir_y']:.3f})")
    else:
        print('ERA5 zarr not found')
        profile = None
    return INFLOW_JSON, profile


@app.cell
def _(CANONICAL_ID, np, plt, profile):
    # Plot inlet profile
    if profile is not None:
        from services.validation.plot_style import apply_style
        apply_style()
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    
        z = np.array(profile['z_levels'])
        u = np.array(profile['u_profile'])
        T = np.array(profile['T_profile'])
    
        ax1.plot(u, z, 'b-', lw=2, label='3-layer ABL')
        ax1.axhline(100,  color='gray', lw=0.8, ls='--', label='Layer boundary')
        ax1.axhline(2000, color='gray', lw=0.8, ls='--')
        ax1.set_xlabel('Wind speed [m/s]')
        ax1.set_ylabel('Height [m]')
        ax1.set_ylim(0, min(5000, z.max()))
        ax1.set_title('Inlet wind profile')
        ax1.legend(fontsize=8)
    
        ax2.plot(T, z, 'r-', lw=2)
        ax2.set_xlabel('Temperature [K]')
        ax2.set_ylabel('Height [m]')
        ax2.set_ylim(0, min(5000, z.max()))
        ax2.set_title('Inlet temperature profile')
    
        fig.suptitle(f'ERA5 → ABL inlet profile — {CANONICAL_ID}')
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 5 — OpenFOAM run (pipeline test)

    Run the full CFD pipeline on the pipeline test case (context_cells=1).
    This should complete in < 2 minutes for 1 km resolution.

    **Requires Docker**: `docker pull opencfd/openfoam-default:v2412`
    """)
    return


@app.cell
def _(CANONICAL_ID, DATA, INFLOW_JSON, SITE_CFG, SRTM_TIF, generate_mesh):
    # Regenerate mesh with inflow profile
    CASE_DIR_TEST_INFLOW = DATA / 'cases' / f'perdigao_{CANONICAL_ID.replace(":","_")}_1km_1x1'

    generate_mesh(
        site_cfg=SITE_CFG,
        resolution_m=1000.0,
        context_cells=1,
        output_dir=CASE_DIR_TEST_INFLOW,
        srtm_tif=SRTM_TIF if SRTM_TIF.exists() else None,
        inflow_json=INFLOW_JSON if INFLOW_JSON.exists() else None,
    )

    print(f'Case ready: {CASE_DIR_TEST_INFLOW}')
    return (CASE_DIR_TEST_INFLOW,)


@app.cell
def _(CASE_DIR_TEST_INFLOW, OpenFOAMRunner):
    # Full run: blockMesh + buoyantSimpleFoam (serial, pipeline test)
    import time
    _runner = OpenFOAMRunner(CASE_DIR_TEST_INFLOW, n_cores=1)
    t0 = time.perf_counter()
    _quality = _runner.run_case(solver='buoyantSimpleFoam', skip_snappy=True)
    dt = time.perf_counter() - t0
    print(f'Run completed in {dt:.0f} s ({dt / 60:.1f} min)')
    print(f'Mesh: {_quality.n_cells:,} cells, maxNonOrtho={_quality.max_non_ortho:.1f}°')  # pipeline test: blockMesh only
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 6 — Export CFD results
    """)
    return


@app.cell
def _(CANONICAL_ID, CASE_DIR_TEST_INFLOW, DATA, ROOT, SITE_CFG, mo, pd):
    from services.module2a_cfd.export_cfd import export_cfd
    CASE_ID_FULL = f"{CANONICAL_ID.replace(':', '_')}_1km_1x1"
    CDB_DIR = DATA / 'cfd-database' / 'perdigao'
    export_cfd(case_dir=CASE_DIR_TEST_INFLOW, towers_yaml=ROOT / 'configs' / 'sites' / 'perdigao_towers.yaml', site_cfg=SITE_CFG, case_id=CASE_ID_FULL, output_dir=CDB_DIR, metadata={'resolution_m': 1000, 'context_cells': 1})
    at_masts_csv = CDB_DIR / CASE_ID_FULL / 'at_masts.csv'
    if at_masts_csv.exists():
        _df = pd.read_csv(at_masts_csv)
        mo.output.replace(mo.ui.table(_df.head(15).round(2)))
    return CASE_ID_FULL, CDB_DIR, at_masts_csv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 7 — Convergence study

    Run all resolutions × context variants.
    **Warning**: this can take several hours for fine resolutions.

    Adjust `resolutions_m` and `context_cells` to control scope.
    """)
    return


@app.cell
def _(CANONICAL_ID, DATA, ERA5_ZARR, ROOT, subprocess):
    CONVERGENCE_DIR = DATA / 'processed' / 'convergence_study'
    _result = subprocess.run(['python', str(ROOT / 'services' / 'module2a-cfd' / 'run_convergence_study.py'), '--case-id', CANONICAL_ID, '--resolutions-m', '10000', '5000', '1000', '--context-cells', '1', '3', '--vertical-variants', '20', '30', '--era5', str(ERA5_ZARR), '--output', str(CONVERGENCE_DIR), '--n-cores', '4'], capture_output=True, text=True)
    print(_result.stdout[-3000:])
    if _result.returncode != 0:
        print('STDERR:', _result.stderr[-500:])
    return (CONVERGENCE_DIR,)


@app.cell
def _(CONVERGENCE_DIR, mo, pd):
    convergence_csv = CONVERGENCE_DIR / 'convergence_results.csv'
    if convergence_csv.exists():
        _df = pd.read_csv(convergence_csv)
        mo.output.replace(mo.ui.table(_df[['case_label', 'resolution_m', 'context_cells', 'n_z', 'n_cells', 'max_non_ortho', 'cpu_time_s', 'ok']].round(2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 8 — CFD vs observations
    """)
    return


@app.cell
def _(CASE_ID_FULL, FIGURES, at_masts_csv, mo):
    # Load at-mast results and compare to observations (if available)
    from services.validation.compare_cfd_vs_obs import load_at_masts, plot_vertical_profiles
    cfd_rows = []
    if at_masts_csv.exists():
        cfd_rows = load_at_masts(at_masts_csv)
        print(f'CFD at-mast: {len(cfd_rows)} rows')
    obs_rows = []
    if cfd_rows:
        plot_vertical_profiles(cfd_rows=cfd_rows, obs_rows=obs_rows, era5_rows=None, output_path=FIGURES / f'profiles_vs_obs_{CASE_ID_FULL}.png', case_id=CASE_ID_FULL)
        _img_path = FIGURES / f'profiles_vs_obs_{CASE_ID_FULL}.png'
        if _img_path.exists():
            mo.output.replace(mo.image(src=str(_img_path)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 9 — Physics QC
    """)
    return


@app.cell
def _(CASE_DIR_TEST_INFLOW, CASE_ID_FULL, CDB_DIR):
    from services.module2a_cfd.check_coherence import check_coherence, update_qc_report

    zarr_path = CDB_DIR / CASE_ID_FULL / 'fields.zarr'
    qc_json   = CDB_DIR / 'qc_report.json'

    if zarr_path.exists():
        report = check_coherence(
            case_id=CASE_ID_FULL,
            zarr_path=zarr_path,
            case_dir=CASE_DIR_TEST_INFLOW,
        )
        update_qc_report(report, qc_json)
    
        overall = 'PASS' if report['overall_ok'] else 'FAIL'
        print(f'QC {overall}: {CASE_ID_FULL}')
        for check_name, check_result in report['checks'].items():
            flag = 'OK' if check_result.get('ok', True) else 'FAIL'
            val  = check_result.get('value', '—')
            print(f'  {check_name:<25} {flag}  value={val}')
    else:
        print(f'Zarr not found: {zarr_path}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Section 10 — Batch PoC (240 runs)

    Launch the full parametric batch locally.
    **Requires** the convergence study to have identified the optimal resolution.
    """)
    return


@app.cell
def _(ROOT, yaml):
    from services.module2a_cfd.run_cfd_batch import generate_case_params
    GRID_YAML = ROOT / 'configs' / 'training' / 'cfd_grid.yaml'
    with open(GRID_YAML) as _f:
        grid_cfg = yaml.safe_load(_f)
    _cases = generate_case_params(grid_cfg, perturbations=False)
    print(f'PoC batch: {len(_cases)} cases (no perturbations)')
    cases_hpc = generate_case_params(grid_cfg, perturbations=True)
    print(f'HPC batch: {len(cases_hpc)} cases (with perturbations)')
    print('\nFirst 3 cases:')
    for c in _cases[:3]:
        print(f"  {c['case_id']}: dir={c['direction_deg']}°, speed={c['speed_ms']:.1f} m/s, stab={c['stability']}")
    return


@app.cell
def _():
    # To actually run the batch (uncomment and run):
    # !python {ROOT}/services/module2a-cfd/run_cfd_batch.py \
    #     --mode local \
    #     --era5 {ERA5_ZARR} \
    #     --resolution-m 1000 \
    #     --context-cells 3 \
    #     --output {CDB_DIR} \
    #     --n-processes 2 \
    #     --n-cores 4

    print('Batch command ready — uncomment to run')
    return


if __name__ == "__main__":
    app.run()
