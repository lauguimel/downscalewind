# FWI Detailed Execution Plan - Stations, Precipitation, Inference, API

Date: 2026-05-06

This document turns the global publication/API plan into an execution protocol.

## Narrative First

The paper should not be written as "we built another CFD surrogate". The cleaner
Nature-style narrative is:

> Fire danger is computed from near-surface meteorology, but operational
> reanalyses are horizontally smooth and miss terrain-induced gradients. We learn
> a CFD-informed, multi-variable near-ground operator and combine it with corrected
> past precipitation to produce FWI-ready fields. The result is not just a lower
> RMSE surrogate; it changes station time series and creates physically
> interpretable 30 m fire-weather gradients inside one ERA5 pixel.

The core storyline:

1. ERA5 is spatially frozen at fire-management scale.
2. Terrain creates near-ground wind/T/RH gradients.
3. The AGL100/K24 surrogate resolves those gradients for wind/T/q.
4. Corrected rain24 closes the missing FWI input.
5. Stations validate the time series.
6. Maps show why the product matters operationally.
7. A lightweight API demonstrates deployability.

Before producing final figures, lock these choices:

- model: AGL100/K24 for FWI, old 40-level model preserved for parapente;
- FWI sampling: wind at 10 m AGL, T/RH at 2 m AGL;
- rain convention: corrected rain24 first, ERA5/ERA5-Land fallback with flags;
- station selection rules before scoring;
- map showcase rules before selecting final figures.

## Station Validation Plan

### Station Source Tiers

Use three tiers. The main validation should be OMM/SYNOP/METAR and professional
weather stations, not ICOS. ICOS/tower data are valuable but operationally
expensive to recover and harmonize, and many sites do not directly provide the
full FWI input set in a standardized station format.

#### Tier 0 - OMM/SYNOP/METAR Professional FWI Stations

Use OMM/SYNOP/METAR first because they are easier to retrieve from public
servers, broadly standardized, and usually contain the variables needed for FWI:
near-surface wind, temperature, humidity/dew point, pressure and precipitation
where available.

Primary sources:

- Météo-France SYNOP OMM archives for France;
- NOAA ISD for Europe-wide SYNOP/METAR harmonized access;
- national open professional station networks when the license is compatible;
- METAR airport stations only when the terrain context is scientifically useful
  or when they act as clean low-relief controls.

Target OMM/SYNOP/METAR families:

| Region | Station examples / families | Why |
| --- | --- | --- |
| South France / Corsica | Montpellier, Nîmes, Istres, Marignane, Perpignan, Montélimar, Ajaccio, Bastia | Mistral/Tramontane/fire weather |
| Relief France | Mont Aigoual, Millau, Aurillac, Clermont-Ferrand, Le Puy/Loudes, Saint-Auban, Grenoble/St-Geoirs | terrain-induced wind signal |
| Iberia | Portugal/Spain SYNOP/METAR near Serra da Estrela, Sierra Nevada, Catalonia, Valencia | Mediterranean fire + orography |
| Italy / islands | Sardinia, Sicily, Apennines, Ligurian coast, Po valley controls | wind/fire gradients |
| Greece / Balkans | coastal/island/relief SYNOP-METAR | Mediterranean extension |
| Alps / Pyrenees | valley + high-altitude professional stations | complex terrain stress test |
| Norway / Scandinavia | professional stations with precipitation | precipitation model generalization |
| Atlantic / flat controls | Bordeaux, La Rochelle, Rennes, Trappes, western Europe low-relief stations | baseline/control class |

Tier 0 outputs:

- `paper/data/stations/tier0_omm_inventory.parquet`
- `paper/data/stations/tier0_omm_timeseries.parquet`
- `paper/data/stations/tier0_wind_metrics.csv`
- `paper/data/stations/tier0_fwi_metrics.csv`

Decision:

- Tier 0 is the scientific and operational anchor.
- Validation focuses on stations with the full FWI variable chain.
- Station selection is frozen before scoring to avoid cherry-picking.

#### Tier 1 - Météo-France Open Professional Stations

Use France as the first broad professional network because licensing and metadata
are clean: data.gouv / Météo-France open-license hourly climate base and SYNOP
archives.

Priority station families:

1. Mediterranean fire / Mistral / Tramontane:
   - Montpellier
   - Mont Aigoual
   - Millau
   - Avignon
   - Nîmes / Nîmes-Garons
   - Istres
   - Perpignan
   - Marignane / Marseille-Provence if available in hourly base
   - Montélimar if available

2. Massif Central / relief controls:
   - Aurillac
   - Clermont-Ferrand
   - Le Puy / Loudes if available
   - Puy de Dôme / nearby mountain stations if available

3. Alpine / pre-Alpine:
   - Saint-Auban if available
   - Grenoble / Saint-Geoirs if available
   - Embrun / Briançon / Bourg-Saint-Maurice if available

4. Atlantic / lower-relief controls:
   - Bordeaux
   - La Rochelle
   - Rennes
   - Trappes / Paris-region control

Implementation rule:

- do not hardcode only the known 15 SYNOP IDs from `module3-precip`;
- build a station metadata inventory from the hourly climate-base resources;
- then apply objective filters.

Tier 1 inclusion filters:

- hourly wind speed and direction available;
- T and humidity/dew point available for FWI if possible;
- at least 60 valid fire-season days per year for selected years;
- station altitude metadata available;
- station is land and not inside dense urban or airport-only exclusion class unless
  used as a "flat/control" category;
- terrain relief class computed from DEM within 1, 3, and 6 km.

Target size:

- first pass: 40-80 French stations;
- final paper table: all stations passing filters;
- showcase: 3-5 time series selected after metrics are computed.

#### Tier 2 - Additional Europe-Wide Professional Networks

Use NOAA NCEI ISD plus national open networks as the broad Europe source for
hourly/synoptic wind/T/dew point and precipitation-related fields.

Target regions:

| Region | Purpose | Candidate station types |
| --- | --- | --- |
| Iberia | fire-weather + terrain | Portugal/Spain SYNOP/METAR and mountain-adjacent stations |
| South France / Corsica | Mistral/Tramontane/fire | SYNOP + national stations |
| Italy / Sardinia / Sicily | Mediterranean wind/fire | SYNOP/METAR |
| Greece / Balkans | fire-weather extension | SYNOP/METAR |
| Alps / Pyrenees | complex terrain | valley + mountain stations |
| Norway / Scandinavia | precipitation generalization | not necessarily FWI showcase; important for rain model |
| Atlantic / flat controls | baseline control | low-relief stations |

Tier 2 target size:

- wind validation: 100-250 stations after filters;
- precipitation training: thousands of daily stations using GHCN-D/ECA&D/MF daily.

Tier 2 use in paper:

- use as broad external validation if quality is good;
- otherwise use as supplementary robustness and keep main text on Tier 0/Tier 1.

#### Optional - ICOS / Tower Cross-Checks

ICOS and tower data should not block the paper path. Use them only when local,
already aligned observations are available or when a specific tower gives a
unique vertical-profile argument.

Optional tower set:

| Site | Role | Use if cheap |
| --- | --- | --- |
| OPE | multiple-height tower | wind profile validation |
| HPB | atmosphere tower | mountain/complex control |
| IPR | atmosphere tower | lowland/complex transition |
| JFJ | high mountain site | difficult/high-relief case |
| PUY | mountain site | elevated terrain validation |
| SAC | atmosphere tower | multi-height validation |
| TRN | atmosphere tower | tall-tower validation |

Optional fire/ecosystem cross-checks from existing campaign config:

| Site | Role |
| --- | --- |
| FR-Pue | Mediterranean oak / fire-risk reference |
| FR-OHP | Haute-Provence / pre-Alpine relief |
| ES-LJu | Sierra Nevada |
| ES-Arn | Mediterranean fire-prone ecosystem |
| ES-Cnd | Spanish fire-prone ecosystem |
| PT-Mi1 | Portugal dry/fire context |
| IT-Cp2 | Italy/Mediterranean |
| IT-Noe | Sardinia / Mediterranean wind-fire context |

### Station Selection Rules

Freeze this before scoring:

1. domain: Europe and Mediterranean fire-relevant belt, nominal bbox
   `lon [-11, 31], lat [35, 72]`;
2. years: 2020-2024 where ERA5 and observations overlap; for first pass use
   2022 fire season plus one recent high-fire-weather year per region;
3. season:
   - fire paper main: May-October;
   - rain model: all year;
4. observation completeness:
   - wind: at least 300 hourly observations in fire season, or at least 60 daily
     noon/afternoon observations if reduced to daily FWI timing;
   - precipitation: at least 300 daily observations/year for training years;
5. wind height:
   - accept known 10 m stations directly;
   - if height unknown but source is SYNOP/METAR, assume 10 m and flag;
   - towers are evaluated at matching known heights;
6. terrain class:
   - flat: relief_3km < 50 m;
   - moderate: 50-200 m;
   - complex: >200 m or slope_1km above threshold;
7. fire relevance:
   - climate class Mediterranean/dry-summer OR historical fire-prone region OR
     high observed fire-season FWI days;
8. exclusion:
   - offshore, ship, buoy, duplicate co-located stations;
   - stations with impossible wind/T/RH values after QC;
   - stations inside land-cover classes incompatible with the surrounding DEM
     unless used as controls.

### Station Validation Outputs

Scripts to add:

- `services/validation/build_station_inventory.py`
- `services/validation/freeze_station_selection.py`
- `services/validation/run_station_surrogate_inference.py`
- `services/validation/evaluate_station_wind.py`
- `services/validation/evaluate_station_fwi.py`

Derived tables:

- `paper/data/stations/station_inventory_all.parquet`
- `paper/data/stations/station_selection_frozen.yaml`
- `paper/data/stations/wind_station_timeseries.parquet`
- `paper/data/stations/wind_station_metrics.csv`
- `paper/data/stations/fwi_station_timeseries.parquet`
- `paper/data/stations/fwi_station_metrics.csv`

Metrics:

- wind speed RMSE/MAE/bias/correlation;
- wind direction MAE only when speed > 2 m/s;
- T/RH RMSE where observations exist;
- FWI MAE/RMSE/correlation;
- threshold skill: FWI > 12, 21, 38 or local danger classes;
- event peak timing error and peak magnitude error.

### Frozen FWI Validation Protocol - Where, When, How

The validation must be split into two products:

1. predeclared validation tables using every station/date that passes QC;
2. showcase figures selected after scoring from the cases where ERA5 and
   observations disagree most strongly.

This prevents cherry-picking while still allowing the paper to show visually
compelling terrain-driven examples.

#### What To Copy From FuXi-CFD

Do not try to reproduce FuXi-CFD as a direct benchmark. Copy the figure logic:

- teacher validation first: parity plots and height-resolved errors against CFD
  truth;
- case-level maps selected from the full test distribution by terrain relief and
  error, not by hand;
- real-world validation conditioned by wind-speed regime and wind direction;
- strong baselines rather than weak raw interpolation baselines.

For our paper, the FuXi-style verification block should use:

- all held-out CFD/test cases for metrics;
- selected representative maps from low, moderate, high and extreme relief;
- heights: 10, 50 and 100 m AGL for wind comparability, plus 2 m T/RH for FWI;
- variables: wind speed, u/v, w, T, q/RH;
- baselines: ERA5 raw/lifted, FNO3D, old ViT/full-volume, AGL100/K24 final model.

Directly reusing FuXi validation sites is optional only. Their main real-world
examples are tall towers such as OPE, Torfhaus and Ispra, but our operational FWI
paper should prioritize OMM/SYNOP/METAR stations with full FWI inputs.

#### Time Convention

Primary FWI convention:

- compute daily FWI from local-noon standard-time weather;
- use station T/RH/wind nearest to local noon, tolerance +/- 1 hour;
- use precipitation over the preceding 24 hours ending at the same nominal time
  where available;
- if only daily or 12 UTC precipitation is available, keep it with a
  `rain_time_convention` flag.

Implementation for Europe:

- France/Germany/Italy/Spain/Norway standard-time zones usually map local noon to
  11 UTC for CET locations, 12 UTC for WET locations, and 10 UTC for EET/Greece;
- for pure SYNOP-only records, also compute a 12 UTC sensitivity variant because
  12 UTC is often the cleanest internationally available timestamp;
- all final tables must include `met_time_utc`, `local_noon_offset`,
  `time_convention`, and `rain_time_convention`.

The primary paper claim should use the local-noon convention. The 12 UTC variant
is a robustness/supplement check.

#### Validation Periods

First frozen run:

- years: 2022-2024;
- fire season: May-October;
- daily FWI timestamps only;
- precipitation model training/validation can use all-year data, but FWI figures
  focus on fire season.

Expansion if data access is clean:

- add 2020-2021 for robustness;
- add 2025 only if station and precipitation sources are complete and stable.

#### Station Region Groups

Build the station inventory first, then freeze exact station IDs. Candidate
regions to include before scoring:

| Group | Purpose | Candidate station families |
| --- | --- | --- |
| FR_MED | fire/Mistral/Tramontane | Nimes, Istres, Marignane, Montpellier, Perpignan, Montelimar, Ajaccio, Bastia |
| FR_RELIEF | terrain wind response | Mont Aigoual, Millau, Saint-Auban, Grenoble/St-Geoirs, Aurillac, Clermont-Ferrand, Le Puy/Loudes |
| IBERIA | Mediterranean fire + relief | Catalonia, Valencia, Andalusia, Sierra Nevada, Portugal interior/coastal contrast |
| GREECE | high fire danger cases | Attica, Evia, Peloponnese, Crete, Rhodes, Thessaly |
| ITALY | Mediterranean islands/Apennines | Sardinia, Sicily, Calabria, Liguria, Apennine valleys, Po valley controls |
| ALPS_PYR | complex terrain stress test | Alpine/Pyrenean valleys and high stations |
| GERMANY | regime/relief contrast | Harz, Black Forest, Bavarian Alps, Rhine/lowland controls |
| BRIT_UK | coastal wind regime and controls | Brittany, Cornwall, Wales, Scottish/English uplands, lowland controls |
| NORDIC | rain/relief generalization | Norway coastal/valley stations, Sweden/Finland controls |
| ATLANTIC_FLAT | low-relief baseline | western France, Netherlands/Belgium, UK lowlands |

Inclusion target:

- main validation: every station passing QC in these groups;
- practical first run: 150-300 stations;
- final showcase: 6-10 station time series, selected after metrics.

#### Model Variants To Score

For each station/date, compute:

1. `OBS_FWI`: FWI from station T/RH/wind/rain, the reference.
2. `ERA5_FWI`: raw ERA5 T/RH/wind/rain at station.
3. `ERA5LAND_RAIN_FWI`: ERA5 meteorology with ERA5-Land or best baseline rain.
4. `DS_WIND_FWI`: downscaled wind only, ERA5 T/RH/rain.
5. `DS_METEO_FWI`: downscaled wind + T/RH, baseline rain.
6. `DS_METEO_RAINV3_FWI`: downscaled wind + T/RH + precip V3, final product.

This ablation is essential: it shows whether the gain comes from wind, T/RH,
rain, or their combination.

#### Mismatch-Driven Showcase Selection

After all validation metrics are computed, select showcase cases with objective
rules:

- high observed fire danger: `OBS_FWI` above regional 90th percentile or above a
  fixed high-danger threshold;
- large ERA5 error: absolute `ERA5_FWI - OBS_FWI` above regional 90th percentile;
- model improvement: `DS_METEO_RAINV3_FWI` closer to `OBS_FWI` than `ERA5_FWI`;
- terrain signal: relief_3km or slope/TPI above the predeclared complex-terrain
  threshold, unless the case is a low-relief control;
- complete data: no missing station variables, no rain-time ambiguity for main
  figures;
- meteorological interest: regime transition, strong wind, dry/wet transition,
  or post-rain drying period.

Showcase categories:

1. France/Mediterranean strong wind and high FWI.
2. Iberia or Greece high fire-risk episode.
3. Italy/Sardinia/Sicily coastal-relief case.
4. Norway or Alpine precipitation/terrain case.
5. Brittany/UK coastal wind or regime-transition control.
6. Low-relief control where downscaling should not invent false structure.

#### Required Outputs

Tables:

- `paper/data/stations/station_inventory_all.parquet`
- `paper/data/stations/station_selection_frozen.yaml`
- `paper/data/fwi/fwi_daily_station_validation.parquet`
- `paper/data/fwi/fwi_daily_metrics_by_station.csv`
- `paper/data/fwi/fwi_daily_metrics_by_group.csv`
- `paper/data/fwi/fwi_showcase_candidates.csv`

Maps and time series:

- `paper/outputs/fwi_maps/{case_id}/terrain.png`
- `paper/outputs/fwi_maps/{case_id}/era5_fwi.png`
- `paper/outputs/fwi_maps/{case_id}/downscaled_fwi.png`
- `paper/outputs/fwi_maps/{case_id}/delta_fwi.png`
- `paper/outputs/fwi_timeseries/{station_id}_{event_id}.png`

Metrics to report:

- FWI MAE/RMSE/bias/correlation by station group;
- threshold skill for high-FWI classes;
- event peak timing and peak magnitude error;
- improvement ratio `(MAE_ERA5 - MAE_downscaled) / MAE_ERA5`;
- heterogeneity metrics inside an ERA5 pixel: range, IQR, 90th-10th percentile,
  high-risk area fraction.

## Precipitation V3 Plan

### Current State

Existing module:

- station labels: GHCN-D and Météo-France SYNOP in code;
- IMERG download through GEE;
- terrain feature extraction through GEE;
- V1 XGBoost trained with IMERG + terrain + location + month;
- current local V1 metrics: spatial CV RMSE about 3.64 mm/day, MAE about
  1.66 mm/day, near-zero mean bias;
- stratified QM exists: season x elevation band x broad climate class;
- V2 feature list is already designed in code but not fully end-to-end:
  ERA5-Land rain, IMERG lags, 3/7-day accumulations, multi-radius TPI,
  distance to coast, wind-orography interaction, climatology anomaly.

### V3 Goal

Make `rain24_mm` publication-grade:

- accurate enough for FWI moisture codes;
- calibrated across dry/wet regimes;
- valid from Norway/northern Europe to Mediterranean;
- reproducible, with spatial and regional validation.

### Data Sources

Training labels:

1. GHCN-D daily PRCP, broad Europe and Mediterranean.
2. ECA&D downloadable daily precipitation where license permits.
3. Météo-France daily climate base and complementary daily stations.
4. Météo-France SYNOP `rr24` as professional cross-check.
5. ICOS rain only as an optional independent fire-site cross-check when already
   locally available and aligned.

Predictors:

1. IMERG V07 Final for research/offline figures.
2. IMERG Late/Early for API low-latency mode if needed.
3. ERA5-Land total precipitation as fallback and feature.
4. Terrain: elevation, slope, aspect, TPI at 1/5/10/25 km, local relief.
5. Distance to coast.
6. Climate class / biome / Koppen-like class.
7. Month/day-of-year cyclic features.
8. Optional ERA5 wind/orographic exposure: upslope_flow, windward index.

### V3 Model Recipe

Use two complementary branches. XGBoost/LightGBM remains the reproducible
tabular baseline and calibration model; add a CNN patch-to-point model because
precipitation has spatial structure that point features cannot fully capture.

#### V3A - Tabular Baseline

Use a two-stage model, not pure one-shot XGBoost:

1. occurrence model:
   - target: wet day, e.g. rain24 > 0.2 or 1 mm;
   - model: calibrated classifier, XGBoost or LightGBM;
   - output: wet probability.

2. amount model:
   - target: positive rain amount after transform `log1p(rain)`;
   - model: XGBoost/LightGBM quantile or squared-error regressor;
   - postprocess: probability-weighted expected rain and quantiles.

3. stratified quantile mapping:
   - apply before residual model or as final calibration;
   - strata: season x climate region x elevation band x wet regime;
   - require minimum samples per stratum and global fallback.

4. residual terrain correction:
   - train residual against station rain after baseline correction;
   - features include multi-scale terrain and exposure.

#### V3B - CNN Patch-To-Point Model

Train a spatial model that takes gridded patches around each station/date and
predicts station rain. This is the likely performance upgrade because it can use
the shape and displacement of rain cells, coast/relief gradients, and upstream
orographic context.

Input patch:

- recommended first version: 64 x 64 or 96 x 96 pixels;
- resolution: native IMERG/ERA5-Land grid first for speed, then optional
  terrain upsampling / separate high-resolution terrain branch;
- temporal context: D0, D-1, D-2 plus 3-day and 7-day accumulations;
- channels:
  - IMERG rain D0/D-1/D-2, IMERG 3d/7d;
  - ERA5-Land precipitation D0/D-1 and 3d;
  - optional ERA5 moisture/wind fields if cheap to extract;
  - DEM, slope, aspect sin/cos, TPI/local relief at multiple scales;
  - distance to coast / land-sea mask;
  - month/day-of-year sin/cos broadcast as channels or metadata.

Architecture:

- small ResNet/UNet/FPN encoder;
- station-aware readout: sample/attention around the station pixel, not only
  global average pooling;
- metadata MLP for station altitude, lat/lon, climate class and rough terrain
  descriptors;
- two heads:
  - occurrence probability;
  - positive amount distribution, e.g. log-rain mean plus quantiles.

Loss:

- binary cross-entropy or focal loss for wet/dry occurrence;
- Huber/MAE on `log1p(rain)` for wet amount;
- optional quantile pinball loss for uncertainty;
- heavier weights for fire-season Mediterranean and high-FWI-relevant dry/wet
  transition days, without hiding global metrics.

Outputs:

- point `rain24_mm` at station/API location;
- optional gridded corrected rain map for showcase tiles;
- uncertainty bands useful for FWI sensitivity.

#### V3C - Final Ensemble / Calibration

Compare these candidates:

1. raw IMERG;
2. ERA5-Land;
3. tabular V3A;
4. CNN V3B;
5. ensemble `CNN + tabular residual/QM`.

The publishable product should be the best spatial-CV model after calibration,
not necessarily the most complex model. If CNN wins on wet/dry skill and
Mediterranean fire-season bias but has residual regional bias, use tabular QM as
the final calibration layer.

Model artifacts:

- `precip_occurrence_model.json`
- `precip_amount_model.json`
- `precip_cnn_patch_model.pt`
- `precip_qm_v3.npz`
- `precip_feature_config.yaml`
- `precip_model_card.md`

### V3 Validation

Splits:

- station-grouped split;
- spatial block split, e.g. 2-5 degree blocks;
- leave-region-out:
  - Nordic,
  - Atlantic,
  - Continental,
  - Alpine,
  - Mediterranean west,
  - Mediterranean east.

Metrics:

- daily RMSE/MAE/bias;
- monthly accumulation bias;
- dry-day false alarm rate;
- wet-day precision/recall;
- heavy-rain recall for >10 mm and >20 mm;
- BUI/DC sensitivity metrics after passing rain through FWI codes;
- metrics by elevation band, season, climate region.

Acceptance to use in paper:

- improves or matches IMERG/ERA5-Land in at least most regions;
- removes strong drizzle/wet bias in Mediterranean fire-season cases;
- no hidden station leakage;
- all results reported by region, not just global mean.

### V3 Implementation Tasks

| ID | Task | Output |
| --- | --- | --- |
| P1 | Upgrade station loaders for ECA&D + MF daily/hourly metadata | `precip_station_inventory.parquet` |
| P2 | Add ERA5-Land rain extraction | `era5land_at_stations.parquet` |
| P3 | Add IMERG lags and rolling windows | `imerg_features.parquet` |
| P4 | Add multi-scale terrain/climate features | `terrain_climate_features.parquet` |
| P5 | Build unified V3 tabular dataset | `precip_training_v3.parquet` |
| P6 | Build CNN patch dataset from IMERG/ERA5-Land/terrain grids | `precip_patch_dataset.zarr` or shard parquet/Zarr |
| P7 | Train tabular occurrence + amount + QM/residual | tabular model artifacts |
| P8 | Train CNN patch-to-point occurrence + amount model | `precip_cnn_patch_model.pt` |
| P9 | Run spatial/region CV and compare raw/tabular/CNN/ensemble | `precip_v3_metrics.csv` |
| P10 | Generate `rain24_by_case` for FWI stations and maps | `rain24_by_case.parquet` |

## Surrogate Inference Production

### Heavy 2D Map Mode

Use AGL100/K24 ViT.

Inputs per site/time:

- terrain tile 6 x 6 km, 180 x 180;
- z0_eff / land-cover roughness;
- lat/lon;
- ERA5 pressure fields u/v/T/q, 3 x 3 x Np;
- ERA5 surface t2m/d2m/u10/v10;
- corrected rain24 from precip V3.

Outputs:

- u/v/w/T/q on 180 x 180 x 24 fixed AGL levels;
- derived wind10, T2, q2/RH2;
- FWI map;
- summary/heterogeneity metrics.

Production scripts:

- `services/module2b-surrogate/infer_agl100_tile.py`
- `services/module2b-surrogate/export_fwi_tile.py`
- `services/module2b-surrogate/batch_infer_station_timeseries.py`

### Station Point Mode

For validation, use the same tile inference but save only the center point and
nearby averaging kernels:

- center pixel;
- 3 x 3 pixel mean;
- 9 x 9 pixel mean for representativeness sensitivity.

This lets the paper report whether station-scale mismatch is due to unresolved
siting rather than model physics.

### Lightweight 1D API Model

Train a separate tiny model for the API point endpoint.

Purpose:

- instant point prediction;
- no 2D map generation required;
- works as a public demo even on CPU;
- preserves heavy 2D map mode for precomputed showcases.

Training data:

- from existing `grid.zarr` campaign;
- sample center points and random pixels from train sites;
- targets at AGL levels 2, 10, 50, 100 m or just FWI variables:
  wind10, T2, q2/RH2, optional w10.

Inputs:

- ERA5 3 x 3 pressure/surface features;
- lat/lon/elevation/z0;
- local terrain features at radii 0.25, 0.5, 1, 3 km:
  elevation, slope, aspect, relief, TPI, exposure to wind direction;
- land-cover roughness;
- corrected rain24 for FWI endpoint.

Candidate models:

- LightGBM/XGBoost for quickest baseline;
- small MLP with residual target;
- tiny transformer/MLP if we want neural consistency.

Outputs:

- `point_downscale_model.json` or `point_downscale_model.pt`;
- latency target: <50 ms CPU model-only;
- endpoint `/v1/fwi/point`.

API split:

- `/v1/fwi/point`: live lightweight 1D prediction.
- `/v1/fwi/map/demo`: precomputed 2D maps for selected cases.
- `/v1/fwi/map`: optional queued heavy inference, later.

## Figure and Output Production Order

### Phase 1 - Evidence Tables

1. AGL100 model card.
2. Station inventory and frozen selection file.
3. Wind station validation metrics.
4. Precip V3 validation metrics.
5. FWI station time series metrics.

### Phase 2 - Inference Products

1. station time series: OBS / ERA5 / surrogate / surrogate+rainV3;
2. 2D map patches for 3-6 cases;
3. point API model training set and benchmark.

### Phase 3 - Figures

Suggested main figures:

1. concept: ERA5 pixel to 30 m FWI-ready micro-meteorology;
2. data/model: CFD-trained AGL100/K24 multi-variable surrogate;
3. station wind validation: time series + metrics by relief;
4. precipitation correction: Europe map + regional CV + Mediterranean drizzle fix;
5. FWI station time series: ERA5 vs ours during fire-weather episodes;
6. 30 m maps: ERA5 flat pixel vs terrain-induced FWI gradients.

Supplement:

- FuXi-CFD positioning;
- full station tables;
- all precip CV folds;
- API latency and endpoint schema;
- 1D point model benchmark versus 2D ViT center-point output.

## Immediate Tasks

### This Week

1. Commit the AGL100/K24 implementation.
2. Build `station_inventory_all.parquet` from:
   - Météo-France SYNOP OMM and hourly climate base;
   - NOAA ISD station inventory and hourly observations for Europe;
   - national open professional networks where licensing is clean;
   - existing ICOS/tower data only as optional cross-checks.
3. Write `station_selection_frozen.yaml`.
4. Upgrade `module3-precip` to V3 dataset generation:
   - features actually match `FEATURE_COLUMNS_V2` plus occurrence/amount split;
   - add ECA&D/MF daily where possible.
5. Run first station inference for Tier 0 OMM/SYNOP/METAR + selected Tier 1
   France stations.
6. Draft narrative skeleton before final figure work.

### Next Week

1. Train precip V3 on Europe multi-year station data.
2. Generate `rain24_by_case.parquet`.
3. Generate station FWI time series.
4. Train the lightweight point model.
5. Produce first API demo outputs:
   - point endpoint live;
   - precomputed 2D map endpoint.

## External Source References

- NOAA NCEI Integrated Surface Database: global hourly/synoptic observations,
  wind/T/dew point/pressure/precip fields and European coverage.
- Météo-France data.gouv hourly climate base: open-license hourly station data,
  updated regularly, all available station parameters.
- Météo-France Archive Synop OMM: 3-hour SYNOP surface observations under
  Licence Ouverte.
- ECA&D daily data: European daily station observations, partly downloadable for
  non-commercial research/education.
- NASA GPM IMERG: 0.1 degree, half-hourly precipitation product, with Final Run
  recommended for research and Early/Late for low-latency applications.
