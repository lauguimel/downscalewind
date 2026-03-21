# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wind downscaling pipeline: ERA5 25km/6h → 1km/1h for complex terrain (Perdigão, Portugal).
4 modules: temporal interpolation → vertical profile → CFD batch → GNN surrogate.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate downscalewind
# Shared package (editable install)
pip install -e ./shared
```

Python 3.11, PyTorch 2.3.1 (MPS on Apple Silicon), zarr 3.x, mlflow 2.14.

## Key Commands

```bash
# Module 1 — training
cd services/module1-temporal
python train.py --zarr-6h ../../data/raw/era5_perdigao.zarr \
  --zarr-1h ../../data/raw/era5_hourly_perdigao.zarr \
  --output ../../data/models/module1 --device auto

# Module 1 — evaluation
python evaluate.py --model ../../data/models/module1/best_model.pt --split test

# Module 1 — inference (6h → 1h Zarr)
python infer.py --zarr-6h ... --zarr-out ... --model ... --start 2017-05-01 --end 2017-06-15

# Module 1 — debug (synthetic data, no CDS needed)
marimo edit debug_notebook.py

# Data ingestion
cd services/data-ingestion
python ingest_era5.py --site perdigao --start 2016-01 --end 2017-12 --output ../../data/raw/era5_perdigao.zarr
python ingest_era5_hourly.py --site perdigao --start 2016-01 --end 2017-06 --output ../../data/raw/era5_hourly_perdigao.zarr

# Module 2A — case generation
cd services/module2a-cfd
python generate_campaign.py configs/training/cfd_grid.yaml --output ../../data/campaign/cases --prefix prd

# Module 2A — deployment (kraken-sim)
kraken-sim smoke data/campaign/cases/campaign.yaml
kraken-sim load data/campaign/cases/campaign.yaml

# Tests (no test suite yet — debug_notebook.py serves as integration test)
cd services/module1-temporal && marimo run debug_notebook.py
```

## Architecture

### Module dependency flow
```
shared/          → imported by ALL services (logging, Zarr I/O)
data-ingestion/  → produces data/raw/*.zarr
module1-temporal → reads 6h Zarr, produces 1h Zarr
module2a-cfd/    → reads ERA5 + SRTM, produces CFD database
module2b-surrogate/ → reads CFD database, trains GNN (TODO)
validation/      → reads all outputs, produces metrics + figures
```

### Cross-module rules
- Services communicate **only via Zarr stores** in `data/` — no cross-service imports
- All services may import from `shared/` only
- CLI entry points use `click`

### Zarr schema (shared/data_io.py)
```
{source}_{site}.zarr/
  pressure/{u,v,z,t,q}  → float32 [time, level, lat, lon]
  surface/{u10,v10,t2m}  → float32 [time, lat, lon]
  coords/{time,level,lat,lon}
```
- `coords/time`: int64 (datetime64[ns] stored as .astype(np.int64))
- Chunks: time=120, level/lat/lon=-1
- Compression: Blosc LZ4

### Module 1 — Temporal (implemented)
- **Model**: `AdvectionResidualInterpolator` in `services/module1-temporal/src/model.py`
- Per-level 2D advection via `F.grid_sample` + CNN 3D residual (Conv3d)
- Bridge: `S(τ) = S_adv(τ) + τ(1-τ)·correction` (zero-init → starts as pure advection)
- ~76K params, <1 MB
- **Variables**: V=5 `[u=0, v=1, z=2, t=3, q=4]`, L=10 pressure levels
- **ERA5 grid convention**: lat[0]=North, v>0=northward

### Module 2A — CFD (in progress)
- OpenFOAM `simpleFoam` (k-ε, OF ESI) via Apptainer on HPC
- Mesher: cfMesh `cartesianMesh` (octree, 2:1 transitions) — replaced blockMesh+snappyHexMesh
- Docker: `microfluidica/openfoam:latest` (OF v2512 ESI + cfMesh pre-compiled)
- 240 runs: 16 directions × 5 speeds × 3 stabilities
- Source terms: Coriolis + plant canopy drag (fvOptions)
- Land cover: ESA WorldCover 2021 (10 m) + ETH Canopy Height 2020 (10 m)
- Pipeline: SRTM→STL→build_domain_fms→cartesianMesh→solver→kraken-sim→Parquet→Zarr

## Data Splits (never change)
- train: 2016-01-01 → 2016-10-31
- val: 2016-11-01 → 2016-12-31
- test: 2017-05-01 → 2017-06-15 (IOP Perdigão — never touch during training)

## Conventions
- Communication language: French; code/comments/commits: English
- Commit style: conventional commits, English
- Logging: JSON structured (shared.logging_config.get_logger)
- Variables: short names (u, v, z, t, q), pressure in hPa, lat/lon in degrees
- Hyperparameters: YAML in configs/training/
- Site config: configs/sites/perdigao.yaml
- MLflow tracking: local in data/mlruns/
- Notebooks: Marimo (.py files), not Jupyter. Use `marimo edit <file>.py`

## Directory Layout — Module 2A CFD

```
services/module2a-cfd/
├── *.py                  # Core pipeline scripts (run_sf_poc, generate_mesh, prepare_inflow, etc.)
├── analysis/             # Post-processing, figures, validation vs obs
│   analyze_convergence.py, make_convergence_figures.py, compare_cfd_obs.py,
│   plot_convergence.py, viz_3d_terrain.py, viz_3d_notebook.py
├── _archive/             # Deferred scripts (HPC orchestration, precursor, old batch runner)
├── templates/openfoam/   # Jinja2 templates for OF case generation
└── tests/                # Unit tests (octagon STL, meshDict, BC templates)
```

```
data/validation/
├── figures/              # ALL publication figures, organized by study
│   ├── convergence/      # Mesh convergence study (5 canonical figures)
│   ├── sf_bbsf/          # SF vs BBSF + Venkatraman reproductions
│   ├── physics_study/    # Physics progressive profiles
│   └── poc_sf/           # PoC SF convergence + CFD vs ERA5
├── convergence_study/    # Resolution sweep results (500/250/100m)
├── physics_study/        # 5-case physics study (Zarr + CSVs)
├── phase0_stability/     # Thermal stratification results
├── phase0_resolution/    # Resolution sweep results
└── phase1_cylinder/      # Cylindrical domain results
```

## Workflow Rules — Studies & Campaigns

### When starting a new study or campaign:
1. **Config first**: create a YAML in `configs/` (e.g., `configs/my_study.yaml`)
2. **Core scripts only at root**: only scripts that are part of the reusable pipeline go in `services/module2a-cfd/`
3. **Analysis scripts → `analysis/`**: any post-processing, figure generation, or comparison script goes in `services/module2a-cfd/analysis/`
4. **One-off experiments → notebook**: use Marimo in `notebooks/` for exploratory work, not standalone .py scripts
5. **Results → `data/validation/<study_name>/`**: each study gets its own subfolder
6. **Figures → `data/validation/figures/<study_name>/`**: all PNGs/PDFs go here, never at the root of `data/validation/`

### When a study is superseded:
- Move its scripts to `_archive/` (not delete — may be reused in v1.5+)
- Keep its results in `data/validation/` (reproducibility)
- Update references in CLAUDE.md and README if needed

### What NOT to do:
- Never put figures directly in `data/validation/` (use `figures/<study>/`)
- Never create standalone .py analysis scripts at the module root — use `analysis/`
- Never leave OpenFOAM case directories in `data/` root — use `data/cases/<study>/`
- Never duplicate pipeline logic in one-off scripts — import from core scripts

## Key Files
- `shared/data_io.py` — Zarr creation/read helpers, CF-conventions
- `shared/logging_config.py` — JSON logger factory
- `services/module1-temporal/src/model.py` — AdvectionResidualInterpolator
- `services/module1-temporal/src/normalization.py` — NormStats (Welford), VARIABLE_ORDER, LOSS_WEIGHTS
- `services/module1-temporal/src/dataset.py` — ERA5TemporalDataset
- `services/module1-temporal/train.py` — AdamW + CosineWarmRestart + MLflow
- `configs/sites/perdigao.yaml` — all site metadata (coordinates, ERA5 grid, CFD domain, data sources)
- `configs/training/cfd_grid.yaml` — parametric grid (directions × speeds × stabilities)
- `services/module2a-cfd/generate_campaign.py` — campaign case generation + kraken-sim manifest
- `DECISIONS.md` — 13 architectural decisions with status (STABLE/A_REVOIR/OUVERTE)
- `LIMITATIONS.md` — scientific assumptions and validity conditions
