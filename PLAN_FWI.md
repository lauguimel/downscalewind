# Plan: NatComms — Terrain-resolved fire weather downscaling
Generated: 2026-04-01
Test command: conda run -n downscalewind python -c "from shared.fwi import *; print('OK')"

## Phase 1 — FWI validation at Perdigao towers (obs vs ERA5 vs CFD)
**Files to create/modify:**
- `services/module2a-cfd/analysis/validate_fwi_at_towers.py` (fix ERA5 RH scatter bug + add CFD comparison when available)
- `shared/fwi.py` (already done)
- `services/data-ingestion/ingest_perdigao_obs.py` (already done — T+RH ingested)

**What to implement:**
Fix the ERA5 RH calculation bug in validate_fwi_at_towers.py (scatter shows aberrant values). Add a --cfd option to interpolate CFD fields at tower positions and compute FWI_cfd vs FWI_obs. Requires campaign Zarr from UGA.

**Success test:** Run validate_fwi_at_towers.py with --obs and --era5, verify 16+ towers produce valid FWI time series. ERA5 RH scatter should show values in [0, 100]%.
**Status:** partial (obs vs ERA5 done, RH scatter bug remains, CFD comparison pending UGA)

## Phase 2 — EFFIS FWI benchmark download
**Files to create/modify:**
- `services/data-ingestion/ingest_effis_fwi.py` (new — download CEMS fire historical FWI from CDS API)

**What to implement:**
Download daily FWI grids from Copernicus CDS dataset `cems-fire-historical` for the Perdigao IOP period (May-June 2017) and Pedrogao Grande fire date (2017-06-17). Extract FWI at Perdigao lat/lon and at Pedrogao Grande. Save as local netCDF or Zarr. Requires `cdsapi` package and CDS credentials (~/.cdsapirc).

**Success test:** EFFIS FWI file exists with daily values for May-June 2017, values in range [0, 50].
**Status:** pending

## Phase 3 — Pedrogao Grande case study
**Files to create/modify:**
- `services/module2a-cfd/analysis/case_study_pedrogao.py` (new)
- `configs/sites/pedrogao_grande.yaml` (new site config)

**What to implement:**
Generate a CFD domain centered on Pedrogao Grande (39.93 N, 8.23 W) using existing pipeline. Run surrogate FNO or direct CFD for 2017-06-17 ERA5 conditions. Compute FWI map at 60m and compare with EFFIS FWI (8km uniform). Overlay fire perimeter if available. Show that terrain channeling in Ribeira de Alge valley creates extreme FWI hotspots.

**Success test:** FWI map figure generated showing spatial heterogeneity, FWI_max > FWI_EFFIS in valley corridors.
**Status:** pending

## Phase 4 — ICOS station validation (3 independent sites)
**Files to create/modify:**
- `services/data-ingestion/ingest_icos.py` (new — download ICOS atmospheric data via icoscp)
- `services/module2a-cfd/analysis/validate_fwi_icos.py` (new)

**What to implement:**
Download T, RH, wind data from 3 ICOS stations in fire-prone terrain (Puechabon FR-Pue, OHP FR-OHP, El Arenosillo ES-Arn) for summer 2017. Generate CFD/surrogate domains for each. Compare FWI_predicted vs FWI_observed vs FWI_ERA5. Compute skill metrics (RMSE, bias, correlation).

**Success test:** RMSE and skill computed for at least 2 stations, skill > 0.3 vs ERA5 baseline.
**Status:** pending

## Phase 5 — FNO retrain on full dataset + benchmark EFFIS
**Files to create/modify:**
- `services/module2b-surrogate/run_training_uga.sh` (update for full 6500-case dataset)
- `services/module2a-cfd/analysis/benchmark_effis.py` (new — EFFIS vs downscaled FWI skill comparison)

**What to implement:**
Retrain FNO on full 6500-case dataset (vs 2504). Run on UGA A6000. Then benchmark: compute RMSE(FWI_downscaled - FWI_obs) vs RMSE(FWI_EFFIS - FWI_obs) at all validation stations. Detection rate of extreme FWI days (FWI > 30).

**Success test:** RMSE U < 0.25 m/s on test set. RMSE(ours) < RMSE(EFFIS) on validation stations.
**Status:** pending
