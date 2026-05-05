# DownscaleWind v2 FWI Paper Plan

Date: 2026-05-05

## Objective

Build the short-term PoC paper around online Fire Weather Index (FWI) maps driven by
fine-scale surrogate downscaling:

- wind: downscaled `u, v, w`, with FWI wind speed sampled near 10 m AGL;
- temperature: downscaled `T`, sampled near screen level or nearest valid low-AGL layer;
- humidity: downscaled `q`, converted to RH using pressure/temperature;
- precipitation: past rainfall from IMERG_QM or the best available gauge-corrected product;
- benchmark: ERA5/ERA5-land style coarse forcing lifted/interpolated to the same grid;
- teacher: OpenFOAM CFD/micro-meteorology fields where available.

The editorial claim should not be a frontal FuXi-CFD benchmark. FuXi-CFD is the
wind-only neighbouring state of the art. Our contribution is the multi-variable
micro-meteorological operator needed by FWI: `u, v, w, T, q` plus past rain.

## Current Model State

Dataset v2 is frozen on Aqua:

- 9,252 `grid.zarr` cases, 630 sites;
- watertight splits: 6,567 train / 1,300 val / 1,385 test;
- groups include `D_fire`, `E_mountain`, `F_wind_onshore`, `C_morpho`;
- native OpenFOAM mesh-grid convention: 180 x 180 x 40, about 33 m horizontal,
  terrain-following vertical coordinate.

Latest residual/geo training and physical test evaluation:

| model | u RMSE | v RMSE | w RMSE | T RMSE | q RMSE | throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FNO S4 | 1.206 m/s | 1.164 m/s | 0.420 m/s | 2.25 K | 1.008e-3 | 2.34 cases/s |
| FNO S4+AGL | 1.221 m/s | 1.186 m/s | 0.439 m/s | 2.27 K | 1.016e-3 | 2.38 cases/s |
| ViT base S4 | 0.953 m/s | 0.924 m/s | 0.378 m/s | 1.82 K | 0.861e-3 | 2.65 cases/s |
| ViT base S4+AGL | 0.949 m/s | 0.932 m/s | 0.391 m/s | 1.86 K | 0.871e-3 | 2.55 cases/s |
| ViT large S4 | 0.958 m/s | 0.926 m/s | 0.378 m/s | 1.81 K | 0.862e-3 | 2.66 cases/s |
| ViT large S4+AGL | 0.974 m/s | 0.936 m/s | 0.392 m/s | 1.84 K | 0.874e-3 | 2.47 cases/s |

Model selection:

- global surrogate paper model: `surrogate_v2_vit_base_resid_s4_geo`;
- near-ground/FWI model: `surrogate_v2_vit_base_resid_s4_geo_agl`;
- keep FNO as architecture baseline, not final model.

0-50 m AGL performance for the FWI model:

| variable | RMSE |
| --- | ---: |
| u | 0.900 m/s |
| v | 0.874 m/s |
| w | 0.243 m/s |
| T | 1.76 K |
| q | 0.733e-3 kg/kg |

## Paper Structure Inspired by FuXi-CFD

### Figure 1 - Concept and Operator

Purpose: explain the scientific object in one panel.

Content:

- ERA5 pressure/surface fields + terrain/z0/lat + past rain;
- OpenFOAM teacher over native terrain-following cells;
- surrogate residual operator;
- FWI map output and API/app use.

Generation needed:

- one clean workflow schematic;
- one real example terrain tile with coarse ERA5 arrows and fine FWI map.

### Figure 2 - Dataset and Native Grid

Purpose: make the grid/input convention defensible versus FuXi-CFD.

Content:

- map of 630 sites by group;
- split counts and watertight split statement;
- native grid: 180 x 180 x 40, terrain-following `coords/z`;
- input tensor inventory: terrain, z0, lat, ERA5 3 x 3 pressure fields, surface fields, inflow/meta.

Generation needed:

- site map by `D_fire`, `E_mountain`, `F_wind_onshore`, `C_morpho`;
- 3D/vertical schematic of terrain-following z and AGL;
- small table of dataset size, groups, splits.

### Figure 3 - Surrogate Accuracy Against CFD Teacher

Purpose: core model-validation figure.

Content:

- parity/scatter or density plots for `u, v, w, T, q`;
- vertical/AGL RMSE profiles;
- comparison against ERA5-lifted baseline;
- architecture comparison: FNO, ViT, ViT+AGL.

Generation needed:

- aggregate predictions from `summary.json` and `per_case.csv`;
- height-resolved and AGL-band metrics;
- group-level metrics by `D_fire`, `E_mountain`, `F_wind_onshore`;
- table of inference speed.

Acceptance gate:

- report both global and 0-50 m AGL metrics;
- explicitly separate wind-only comparison from multi-variable contribution.

### Figure 4 - Spatial Pattern Fidelity

Purpose: visual proof that the surrogate captures terrain-induced heterogeneity.

Content:

- 3 showcase cases:
  - one `D_fire` Mediterranean site;
  - one `F_wind_onshore` simple ridge/wind case;
  - one `E_mountain` complex terrain case;
- maps at low AGL layer for ERA5-lifted, surrogate, CFD teacher, and error;
- variables: wind speed, T, q/RH, optionally w.

Generation needed:

- select representative test cases using terrain relief + per-case skill;
- export surrogate fields for selected cases;
- sample/interpolate to fixed AGL levels: 2 m, 10 m, 50 m, 100 m.

Acceptance gate:

- selected cases should be visually strong but not cherry-picked only for best error;
- include one difficult/high-relief example.

### Figure 5 - FWI Pipeline and Validation

Purpose: show that the downscaled fields matter for fire-weather quantities.

Content:

- FWI components or final FWI time series at validation sites;
- comparison: observations/gauge-derived reference vs ERA5 baseline vs downscaled;
- ablation: wind-only downscale, wind+T/q downscale, rain source variants.

Generation needed:

- define FWI sampling:
  - wind speed from `sqrt(u^2 + v^2)` near 10 m AGL;
  - T near 2 m AGL or nearest low-AGL valid layer;
  - RH from `q`, T, and pressure;
  - rain: previous 24 h accumulation from IMERG_QM, with ERA5 rain fallback;
- station/tower validation set:
  - ICOS/FWI sites already ingested where possible;
  - fire-oriented Mediterranean cases where rain and meteo observations exist;
- compute metrics:
  - RMSE/MAE/correlation for FWI;
  - categorical skill for fire-danger thresholds;
  - event-day/high-FWI subset metrics.

Acceptance gate:

- FWI calculation must be reproducible from a single derived table per site/time;
- precipitation source choice must be fixed before writing Results.

### Figure 6 - Relief-Induced FWI Heterogeneity

Purpose: the application figure for the PoC/API story.

Content:

- 3 fire-weather showcase maps:
  - ERA5 coarse FWI;
  - downscaled FWI;
  - difference or heterogeneity index;
  - optional teacher-derived FWI where CFD fields are available.

Generation needed:

- run surrogate for selected D_fire cases;
- combine with past rain;
- compute FWI at every horizontal cell;
- quantify heterogeneity:
  - domain standard deviation;
  - high-risk area fraction;
  - ridge/valley contrast;
  - max-minus-median FWI.

Acceptance gate:

- at least one case must show a clear relief-driven difference with physically
  interpretable wind/T/RH drivers.

## Supplementary Material

Supplementary figures/tables:

- FuXi-CFD positioning table:
  - FuXi inputs: terrain + z0 + coarse `u,v`;
  - our inputs: terrain + z0 + lat + ERA5 pressure/surface `u,v,T,q` + inflow/meta;
  - FuXi outputs: `u,v,w,k`;
  - our outputs: `u,v,w,T,q`;
  - same family of CFD-informed surrogates, different operator and application.
- ablation table:
  - residual vs absolute target;
  - geo/AGL injection;
  - AGL-weighted loss;
  - FNO vs ViT;
  - large vs base.
- runtime:
  - cases/s per GPU;
  - memory footprint;
  - estimated API latency with model preloaded.
- limitations:
  - no explicit precipitation downscaling by the surrogate yet;
  - q-to-RH conversion sensitivity to pressure choice;
  - OpenFOAM teacher assumptions;
  - weak-wind/stable-boundary-layer regimes.

## Data Generation Tasks

1. Freeze model artifacts:
   - choose final FWI model checkpoint;
   - write a model card with checkpoint path, epoch, val metric, test metrics.

2. Build paper metrics tables:
   - global metrics by variable;
   - AGL-band metrics;
   - group metrics by site class;
   - per-case skill distribution versus ERA5-lifted baseline.

3. Export fixed-AGL fields:
   - script: `export_fixed_agl_fields.py`;
   - outputs: compact Zarr/NetCDF for selected cases and optional full test subset;
   - levels: 2 m, 10 m, 50 m, 100 m.

4. Select showcase cases:
   - script: `select_showcase_cases.py`;
   - constraints: one `D_fire`, one `F_wind_onshore`, one `E_mountain`;
   - criteria: terrain relief, surrogate skill, visually interpretable flow.

5. Build FWI input tables:
   - script: `build_fwi_inputs.py`;
   - columns: site_id, case_id, time, T, RH, wind10, rain24, source flags;
   - variants: ERA5 baseline, surrogate, optional CFD teacher.

6. Compute FWI:
   - script: `compute_fwi_downscaled.py`;
   - use shared `shared/fwi.py`;
   - output site/time CSV and grid maps.

7. Validate FWI:
   - script: `evaluate_fwi_validation.py`;
   - metrics by site, season, high-FWI days, rain-source variant;
   - threshold/categorical metrics for fire-danger classes.

8. Generate figures:
   - script: `make_paper_figures.py`;
   - output `paper/figures/fig01_*.png/pdf` through `fig06_*.png/pdf`;
   - all figures reproducible from derived tables.

## Writing Plan

1. Introduction:
   - problem: FWI needs local wind/T/RH/rain, but ERA5 smooths terrain;
   - gap: existing ML downscaling and FuXi-CFD focus mostly on wind-only fields;
   - contribution: multi-variable CFD-informed surrogate for FWI-ready micro-meteorology.

2. Results:
   - dataset/native-grid convention;
   - surrogate skill against OpenFOAM teacher;
   - near-ground/AGL performance;
   - FWI validation and relief-induced heterogeneity;
   - runtime/API feasibility.

3. Methods:
   - CFD campaign and mesh/grid convention;
   - input features and residual target;
   - ViT/FNO architectures and losses;
   - physical-unit evaluation;
   - FWI calculation and precipitation source.

4. Discussion:
   - relation to FuXi-CFD;
   - why multi-variable matters;
   - operational limits and next data needs.

## Joint Validation Gates

Before writing each Results section, validate together:

1. final model choice: global model vs FWI near-ground model;
2. precipitation source and rain accumulation convention;
3. selected showcase sites and cases;
4. FWI validation sites and accepted observation sources;
5. final figure list and any claims involving FuXi-CFD.
