# DownscaleWind FWI Publication + API Workplan

Date: 2026-05-06

## Current Decision State

The paper and PoC should now be organized around the dedicated near-ground FWI
model:

- FWI model: `surrogate_v2_vit_base_resid_s4_geo_agl100_k24/best.pt`
- output grid: 24 fixed AGL levels from 0 to 100 m
- test full-domain RMSE: u 0.908 m/s, v 0.881 m/s, w 0.235 m/s,
  T 1.66 K, q 7.30e-4 kg/kg
- test 0-50 m RMSE: u 0.843 m/s, v 0.813 m/s, w 0.205 m/s,
  T 1.61 K, q 7.17e-4 kg/kg
- parapente / full-volume model remains separate:
  `surrogate_v2_vit_base_resid_s4_geo_agl/best.pt`

The central-crop fine-tune should not be promoted as a primary model: it did not
beat the baseline on the 2 km crop.

## Publication Claim

Do not frame the paper as a frontal FuXi-CFD benchmark. FuXi-CFD is the wind-only
neighboring state of the art.

Primary claim:

> A CFD-informed, terrain-aware, multi-variable downscaling operator produces
> near-ground wind, temperature, and humidity fields suitable for online FWI
> mapping in complex terrain.

Defensible contrast with FuXi-CFD:

- FuXi-CFD inputs: terrain, z0, coarse u/v wind.
- FuXi-CFD outputs: u, v, w, k.
- DownscaleWind inputs: terrain, z0, lat, ERA5 pressure/surface u/v/T/q,
  inflow metadata, and past rain for FWI.
- DownscaleWind outputs: u, v, w, T, q, then FWI fields.
- FuXi-CFD is more accurate on wind-only; DownscaleWind is FWI-ready.

## Strategic Update - Highest Impact Validation Path

The strongest paper path is no longer only "surrogate versus CFD teacher". It should
add two observational/application pillars:

1. near-ground station validation for wind, and where possible T/RH;
2. Europe-wide past-precipitation downscaling / bias correction for `rain24_mm`.

This gives a clearer story:

- the CFD teacher proves that the model learns terrain-induced micro-meteorology;
- stations prove that the near-ground wind signal is useful outside the teacher;
- precipitation correction closes the missing FWI input;
- fire-weather case studies show why a 30 m FWI map differs from a frozen ERA5
  25 km pixel.

Important publication rule: station selection must be based on objective criteria
known before scoring, not on whether the model performs well. It is acceptable to
select showcase figures for clarity, but the validation table must include every
station that passes the predeclared QC/terrain/source filters.

## Priority Work Packages

### WP1 - Observed Near-Ground Wind Validation

Goal: show that the surrogate improves near-ground wind at real professional
stations, not only against OpenFOAM.

Candidate sources:

- WMO/SYNOP stations as the primary validation source;
- METAR airport stations where terrain context is useful enough or as low-relief
  controls;
- national professional weather stations, including Meteo-France / European
  open-data sources where licensing allows;
- Infoclimat or equivalent aggregator data only if provenance and license are
  acceptable;
- ICOS/tower sites only as optional cross-checks when data are already local and
  aligned.

Station preselection filters:

- wind observations available at or near 10 m;
- enough timestamps overlapping ERA5/campaign periods;
- station metadata includes lat/lon, altitude, instrument height if possible;
- within Europe and inside terrain/land-cover coverage;
- terrain-relief class assigned before evaluation;
- exclude stations with obvious siting problems, offshore/airport-only cases, or
  missing metadata unless explicitly used as a low-complexity baseline.

FWI timing convention:

- primary validation uses local-noon standard-time weather, nearest station report
  within +/- 1 hour;
- precipitation is the preceding 24 h amount ending at that nominal time when
  available;
- a fixed 12 UTC sensitivity product is allowed for SYNOP-only sources but must
  be flagged separately.

Outputs:

- `station_inventory.parquet`: all candidate stations and metadata;
- `station_validation_selection.yaml`: frozen inclusion/exclusion rules;
- `wind_station_timeseries.parquet`: observed, ERA5 baseline, surrogate;
- `wind_station_metrics.csv`: RMSE/MAE/bias/correlation by station and relief class.

FWI validation variants:

- `OBS_FWI`: station-derived reference;
- `ERA5_FWI`: raw ERA5;
- `DS_WIND_FWI`: downscaled wind only;
- `DS_METEO_FWI`: downscaled wind + T/RH;
- `DS_METEO_RAINV3_FWI`: final wind + T/RH + corrected rain.

Acceptance:

- report metrics for the complete selected station set;
- show 3-5 illustrative time series only after the objective selection is frozen;
- include a terrain-relief stratification so the paper can show where the method
  matters most.
- showcase cases selected after scoring from high-FWI, high-ERA5-mismatch days
  where the final product improves against station-derived FWI.

### WP2 - Europe-Wide Rain24 Downscaling / Bias Correction

Goal: make past precipitation a credible FWI input instead of a weak fallback.

Starting point:

- reuse the precipitation module already started in `services/module3-precip`;
- extend the station set from a few hundred stations to broad Europe coverage;
- include climate/biome/season classes and terrain features;
- keep the existing XGBoost path as a fast baseline, but add a CNN
  patch-to-point branch that can ingest gridded IMERG/ERA5-Land/terrain context.

Model target:

- `rain24_mm` or daily precipitation, depending on source alignment;
- calibrated estimate at station/grid point from satellite/reanalysis + terrain +
  season/climate class;
- uncertainty or source flags if feasible.

Candidate features:

- raw IMERG or best available satellite precipitation;
- ERA5/ERA5-Land precipitation fallback;
- elevation, slope, aspect, TPI/orographic exposure;
- month/season, latitude/longitude, climate class;
- optional land-cover class.

Model candidates:

- tabular occurrence + amount model with XGBoost/LightGBM and stratified QM;
- CNN patch-to-point model using spatial patches around each station/date;
- final calibrated ensemble if CNN improves spatial CV but keeps regional bias.

Outputs:

- `precip_station_inventory.parquet`;
- `precip_training_europe.parquet`;
- `precip_patch_dataset.zarr` or equivalent shard set;
- `precip_qm_model.*` or calibrated model artifact;
- `precip_cnn_patch_model.pt` if the CNN branch wins validation;
- `rain24_by_case.parquet` for every FWI case;
- validation metrics by region, season, elevation band, and wet/dry regime.

Acceptance:

- spatial split by station or region, not random rows;
- metrics reported from Norway/northern Europe through Mediterranean/southern
  Europe;
- fixed hierarchy for FWI: corrected rain first, then documented fallback.

### WP3 - Fire-Weather Station Time Series

Goal: show at actual sites that ERA5 and the downscaled product diverge in
physically meaningful ways during fire-weather periods.

Tasks:

- identify stations near historical fire-prone or fire-event areas with some
  terrain relief;
- align observations, ERA5 baseline, surrogate wind/T/RH, and corrected rain;
- compute FWI time series for each variant;
- quantify high-FWI event timing, peaks, and categorical danger thresholds.

Outputs:

- `fwi_station_timeseries.parquet`;
- `fwi_station_metrics.csv`;
- figure-ready time series for selected fire-weather episodes.

Acceptance:

- use all stations passing the frozen filters for metrics;
- showcase time series may be selected for interpretability, but not used as the
  only evidence;
- explicitly separate "observed-station validation" from "fire-event case study".

### WP4 - 30 m FWI Map Demonstration

Goal: produce the visual/application proof: ERA5 is spatially flat at 25 km, while
the surrogate creates terrain-driven gradients at 30-33 m.

Tasks:

- choose 3-6 map patches after WP1/WP3 selection;
- compute ERA5 FWI as a coarse or lifted single-pixel baseline;
- compute downscaled FWI at the surrogate grid;
- produce maps and heterogeneity metrics.

Outputs:

- ERA5 FWI map;
- downscaled FWI map;
- difference map;
- gradient/heterogeneity indices:
  domain standard deviation, high-risk area fraction, ridge-valley contrast,
  max-minus-median FWI.

Acceptance:

- at least one case must show a clear terrain-driven gradient and explain whether
  wind, T/RH, or rain drives it;
- maps must be reproducible from derived tables, not hand-built arrays.

## Work Groups

### Group A - Model, Metrics, and Ablations

Goal: freeze the model evidence needed for the paper.

Tasks:

| ID | Task | Inputs | Output | Acceptance |
| --- | --- | --- | --- | --- |
| A1 | Model card for AGL100/K24 | checkpoint, history, eval summaries | `docs/model_card_agl100_k24.md` | includes checkpoint path, epoch, training recipe, levels, test metrics |
| A2 | Commit AGL100 code path | current local changes | git commit | repo clean; old 40-level model path still supported |
| A3 | Group metrics by site class | `per_case.csv`, split manifest/site metadata | CSV + table by D_fire/E_mountain/F_wind/C_morpho | RMSE for u/v/w/T/q, 0-50 m and 0-100 m |
| A4 | ERA5-lifted baseline table | eval summaries | CSV/table | model skill vs baseline for all variables and AGL bands |
| A5 | FuXi positioning table | Nature paper + supplement | manuscript table | clearly marks "not same task"; includes inputs/outputs/resolution/levels/runtime |
| A6 | Optional 10 epoch continuation | AGL100/K24 best checkpoint | new checkpoint + eval | only keep if test 0-50 m improves without obvious instability |

Dependencies:

- A2 should happen before further API/paper code work.
- A3/A4 depend on current evaluation outputs already available on Aqua.

### Group B - Fixed-AGL Field Export and Showcase Selection

Goal: produce clean derived data for figures and FWI maps.

Tasks:

| ID | Task | Inputs | Output | Acceptance |
| --- | --- | --- | --- | --- |
| B1 | Export selected case predictions | AGL100 model, test cases | compact Zarr/NetCDF per selected case | contains terrain, z0, AGL levels, ERA5 baseline, surrogate, CFD teacher |
| B2 | Select candidate showcase cases | per-case skill, terrain relief, site group | ranked CSV | at least 5 candidates per D_fire, E_mountain, F_wind |
| B3 | Lock 3 main showcase cases | B2 + visual inspection | manifest YAML | one D_fire, one simple wind/ridge, one complex mountain |
| B4 | Export maps at standard levels | B1 | PNG-ready arrays at 2/10/50/100 m | wind speed, u/v, w, T, q/RH, error fields |
| B5 | Build heterogeneity metrics | B1/B4 | CSV | domain std, high-risk area fraction, ridge-valley contrast, max-minus-median |

Acceptance gate:

- Showcase cases must be strong but not cherry-picked only for low error.
- Include one difficult/high-relief case.

### Group C - Precipitation and FWI Inputs

Goal: define the non-surrogate inputs required for FWI reproducibly.

Tasks:

| ID | Task | Inputs | Output | Acceptance |
| --- | --- | --- | --- | --- |
| C1 | Choose rain source hierarchy | IMERG_QM, ERA5/ERA5-Land, gauges if present | written convention | one canonical `rain24_mm` source plus fallback flags |
| C2 | Build rain24 extraction script | case timestamp/site, rain source | `rain24_by_case.parquet` | each case has rain24, source, missing flag |
| C3 | Convert q/T to RH | surrogate q/T, pressure approximation | tested function | RH clipped/validated; method documented |
| C4 | Build FWI input table | surrogate, ERA5 baseline, rain24 | `fwi_inputs.parquet` | columns: case_id, site_id, time, x, y, T2, RH2, wind10, rain24, source |
| C5 | Define FWI variants | C4 | variant config YAML | ERA5 baseline, wind-only downscale, wind+T/q downscale, full surrogate+rain |

Acceptance gate:

- No FWI result goes into the paper until C1 is fixed.
- Every FWI number must be reproducible from `fwi_inputs.parquet`.

### Group D - FWI Computation and Validation

Goal: prove that downscaling changes FWI in a useful and physically interpretable way.

Tasks:

| ID | Task | Inputs | Output | Acceptance |
| --- | --- | --- | --- | --- |
| D1 | Implement canonical FWI calculator | C4/C5 | `compute_fwi_downscaled.py` | matches a known reference implementation on test vectors |
| D2 | Grid FWI maps for showcase cases | B1/C4/D1 | Zarr/PNG arrays | ERA5 FWI, downscaled FWI, difference, heterogeneity metrics |
| D3 | Site/time validation table | OMM/SYNOP/METAR/FWI obs, ERA5 baseline, surrogate | `fwi_validation.parquet` | aligned by timestamp/site; missing data audited |
| D4 | FWI metrics | D3 | CSV/table | RMSE/MAE/corr; threshold skill for fire danger classes |
| D5 | Event-day analysis | D3/D4 | figure/table | high-FWI and dry/windy subset metrics |

Acceptance gate:

- At least one validation figure must compare ERA5 baseline vs downscaled FWI
  against an observational or accepted reference product.
- If validation data are too sparse, label the result as "case-study PoC", not
  broad operational validation.

### Group E - Paper Figures

Goal: generate all article figures from derived tables, not ad hoc notebooks.

Tasks:

| ID | Figure | Script | Required data | Acceptance |
| --- | --- | --- | --- | --- |
| E1 | Fig. 1 operator concept | `make_fig01_concept.py` | selected terrain + workflow assets | readable in one panel; includes API/FWI endpoint |
| E2 | Fig. 2 dataset/native grid | `make_fig02_dataset_grid.py` | site manifest, z/AGL example | map, split table, terrain-following grid schematic |
| E3 | Fig. 3 surrogate metrics | `make_fig03_surrogate_metrics.py` | A3/A4 | global, 0-50 m, profiles, baseline skill |
| E4 | Fig. 4 spatial fidelity | `make_fig04_spatial_cases.py` | B1/B4 | ERA5, surrogate, CFD, error for 3 cases |
| E5 | Fig. 5 FWI validation | `make_fig05_fwi_validation.py` | D3/D4/D5 | time/skill plots and ablations |
| E6 | Fig. 6 relief FWI heterogeneity | `make_fig06_fwi_maps.py` | D2/B5 | downscaled vs ERA5 FWI maps and heterogeneity index |

Acceptance gate:

- All figure outputs live under `paper/figures/`.
- Every figure script can run from a documented derived-data directory.

### Group F - API and Demo

Goal: turn the PoC into a small, credible online product surface.

Tasks:

| ID | Task | Inputs | Output | Acceptance |
| --- | --- | --- | --- | --- |
| F1 | Define API contract | model I/O, FWI fields | OpenAPI schema/spec | request/response fixed before implementation |
| F2 | Replace old 9k FNO engine path | AGL100 ViT checkpoint | `FWIEngine` or equivalent | preloads model; preserves viewer-only mode |
| F3 | Build preprocessing path | terrain, z0, ERA5 u/v/T/q, surface fields | model-ready tensors | matches training normalization and AGL levels |
| F4 | Add rain provider | C1/C2 source hierarchy | rain24 value + flags | deterministic fallback behavior |
| F5 | Add `/v1/fwi` endpoint | F2/F3/F4/D1 | JSON + optional binary grid response | returns wind10, T2, RH2, rain24, FWI, metadata |
| F6 | Add demo case mode | B3/D2 | static demo endpoint | works without live ERA5/rain downloads |
| F7 | Latency benchmark | local GPU/Aqua + CPU target | benchmark table | reports model-only and end-to-end latency |
| F8 | Deployment plan | OCI/free API constraints | runbook | cache paths, env vars, rate limit, model artifact location |

Suggested API response fields:

- `metadata`: model id, timestamp, domain, grid spacing, AGL levels, sources
- `fields`: wind10, T2, RH2, rain24, FFMC/DMC/DC/ISI/BUI/FWI if available
- `summary`: min/median/max, high-risk fraction, heterogeneity index
- `assets`: optional URLs for compressed grid or rendered PNG

Acceptance gate:

- The API must support a precomputed demo first. Live data can come second.
- The old parapente/full-volume model must not be overwritten.

### Group G - Manuscript Writing

Goal: write sections in the same order as data maturity.

Tasks:

| ID | Section | Dependencies | Output | Acceptance |
| --- | --- | --- | --- | --- |
| G1 | Methods: dataset/grid/model | A1/A5 | manuscript draft | can be written now |
| G2 | Methods: FWI/rain | C1/D1 | manuscript draft | includes exact formulas and source hierarchy |
| G3 | Results: surrogate skill | A3/A4/E3 | manuscript draft | includes FuXi positioning without overclaiming |
| G4 | Results: spatial fidelity | B3/E4 | manuscript draft | explains physical features in selected cases |
| G5 | Results: FWI validation | D3/D4/E5 | manuscript draft | only claims what validation supports |
| G6 | Results: API feasibility | F5/F7 | manuscript draft | includes throughput/latency and endpoint behavior |
| G7 | Discussion | all | manuscript draft | limitations: rain, weak wind/stability, teacher assumptions |

Acceptance gate:

- Do not write strong FWI validation claims before D3/D4 are complete.
- Do write Methods and surrogate Results immediately; those are ready.

### Group H - Reproducibility and Repository Hygiene

Goal: make the whole chain auditable.

Tasks:

| ID | Task | Output | Acceptance |
| --- | --- | --- | --- |
| H1 | Commit AGL100 implementation | git commit | includes dataset, train/eval, PBS script |
| H2 | Sync final scripts to Aqua | remote code state | `py_compile` and `bash -n` pass |
| H3 | Derived-data manifest | `paper/data/manifest.yaml` | all tables/figures point to versioned files |
| H4 | Model artifact manifest | `models/manifest.yaml` or doc | lists parapente and FWI checkpoints separately |
| H5 | Repro runbook | `docs/fwi_repro_runbook.md` | fresh user can regenerate metrics/figures from outputs |

## Near-Term Sequence

### Day 0-1

1. Commit AGL100/K24 support.
2. Write model card.
3. Generate group metrics and model-vs-ERA5 summary tables.
4. Freeze rain source decision.

### Day 2-4

1. Select showcase cases.
2. Export fixed-AGL fields for selected cases.
3. Build FWI input table.
4. Implement canonical FWI computation and produce first maps.

### Day 5-7

1. Generate Figures 2-4.
2. Draft Methods and surrogate Results.
3. Implement demo-mode `/v1/fwi`.
4. Produce latency numbers.

### Week 2

1. Finish FWI validation or downgrade to PoC case-study language.
2. Generate Figures 5-6.
3. Draft full manuscript.
4. Prepare API deployment runbook and demo.

## Decisions To Validate Together

1. Final paper model: promote AGL100/K24 as FWI model.
2. Rain convention: IMERG_QM vs ERA5/ERA5-Land fallback.
3. FWI level convention: wind at 10 m, T/RH at 2 m or nearest model level.
4. Three showcase cases.
5. Validation source hierarchy and whether the Results claim is validation or PoC.
6. API scope: precomputed demo only first, or live ERA5/rain fetch in v1.
