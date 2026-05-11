# FWI Station Validation Audit - 2026-05-11

## Frozen Protocol

- Station source: Meteo-France SYNOP/OMM public archives, 2022.
- Scored fire-weather window: 2022-06-01 to 2022-09-30, nearest observation to 12 UTC.
- Selected stations: n=20.
- Event inference manifest: m=20 timestamps per station, n x m=400 UVT cases.
- Daily validation table: 2354 station-days for time-series FWI/rain scoring.

The station list is frozen in
`data/validation/fwi_station_audit_2022_n20/selected_stations.csv`.
The top-case inference manifest is frozen in
`data/validation/fwi_station_audit_2022_n20/inference_manifest.csv`.

## Selected Stations

1. MARIGNANE
2. MONTELIMAR
3. MILLAU
4. PERPIGNAN
5. LYON-ST EXUPERY
6. EMBRUN
7. RENNES-ST JACQUES
8. NANCY-OCHEY
9. NANTES-BOUGUENAIS
10. DIJON-LONGVIC
11. TOURS
12. POITIERS-BIARD
13. TROYES-BARBEREY
14. BASTIA
15. LE PUY-LOUDES
16. BORDEAUX-MERIGNAC
17. STRASBOURG-ENTZHEIM
18. TOULOUSE-BLAGNAC
19. NICE
20. BREST-GUIPAVAS

The selection is not a paper taxonomy. It is a pragmatic validation set: high
observed summer FWI first, with enough geographic spread to avoid selecting only
one weather regime.

## Rain Correction Status

Station rain comparison was run on all 2354 station-days:

| Product | RMSE mm/day | MAE mm/day | Bias mm/day | Dry false wet rate |
|---|---:|---:|---:|---:|
| IMERG raw | 5.113 | 1.903 | +0.304 | 0.155 |
| ERA5-Land rain | 4.808 | 1.822 | +0.485 | 0.183 |
| DownscalRain direct CNN | 3.932 | 1.383 | -0.345 | 0.149 |
| IMERG fire-corrected | 4.989 | 1.692 | -0.067 | 0.066 |

For the specific dry-period halo subset, IMERG fire-corrected reduces the dry
false wet rate from 0.646 to 0.274. For high observed FWI days, the same
correction removes false wet days entirely in this station set.

Interpretation: the CNN direct prediction is the better generic rain estimator,
but the IMERG-first dry correction is the better FWI moisture veto. The final FWI
pipeline should keep both outputs and use the fire-corrected product as the
operational FWI rain input.

## FWI Baseline Status

`build_fwi_baseline_validation.py` now builds a long FWI table with these
products:

- `OBS_station_FWI`
- `ERA5_met_ObsRain_FWI`
- `ERA5_met_IMERG_FWI`
- `ERA5LandRain_FWI`
- `ERA5_met_DownscalRainDirect_FWI`
- `ERA5_met_IMERGFireCorrected_FWI`
- `StationMet_IMERGFireCorrected_FWI`
- `DownscaleWind_DownscalRain_FWI`, once station UVT inference is provided

Current diagnostic comparison uses a fire-season window initialized on the first
selected day per station. This avoids mixing full-history OBS FWI with products
that do not yet have Jan-May ERA5/UVT meteorological spin-up.

All station-days, RMSE against the window-consistent OBS reference:

| Product | RMSE FWI | MAE FWI | Bias FWI | Corr |
|---|---:|---:|---:|---:|
| ERA5-Land rain FWI | 4.224 | 2.600 | -2.063 | 0.720 |
| ERA5 met + DownscalRain direct | 3.670 | 2.263 | -1.714 | 0.789 |
| ERA5 met + IMERG fire-corrected | 3.142 | 1.967 | -1.066 | 0.828 |
| ERA5 met + observed rain | 2.954 | 1.743 | -1.481 | 0.887 |
| Station met + IMERG fire-corrected | 2.191 | 1.240 | +0.533 | 0.935 |

The rain correction gives a clear FWI improvement over the ERA5-Land rain
baseline in this diagnostic setting. The remaining gap is mainly meteorology,
especially near-surface wind/T/RH, which is exactly where the UVT surrogate must
enter.

## UVT Inference Status

Not completed yet for the 400 station event cases.

The trained checkpoint exists on Aqua:
`/home/maitreje/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24/best.pt`.

Checkpoint config:

- ViT base, residual model, slopes, z+AGL geo channels.
- AGL target levels: 0, 2, 3, 4, 5, 10, ..., 100 m.
- Epoch 27, validation MSE 0.1207 in normalized units.

The new runner `services/validation/run_station_surrogate_inference.py` can
perform the station extraction once each case has an input-ready `grid.zarr`.
It writes the UVT table expected by `build_fwi_baseline_validation.py`.

Missing data-generation step:

1. Build one v2 input-only `grid.zarr` per station/timestamp:
   terrain 180x180, z0, terrain-following z/AGL, ERA5 3x3 pressure/surface.
2. Run the ViT AGL100 checkpoint on those inputs.
3. Extract center-cell 10 m wind and 2 m T/RH.
4. Re-run `build_fwi_baseline_validation.py --downscaled-uvt`.

## Forecast And Reanalysis Baselines

- ERA5: local JJA 2022 surface/pressure Zarr exists and was used for surface
  T/RH/wind. ERA5 total precipitation is not present in the local JJA store.
- ERA5-Land: local 2022 daily rain is present and sampled. Full ERA5-Land
  hourly meteo/rain can be added from CDS; official ERA5-Land is ~9 km and
  hourly from 1950 onward.
- EFFIS/GWIS: useful product-level fire-danger baseline. EFFIS uses Canadian FWI
  and deterministic ECMWF/Meteo-France forecast products, but this compares FWI
  products rather than validating our local UVT variables.
- ICON-D2: useful only for Germany/neighbouring-country cases unless the paper
  includes those stations. DWD documents ICON-D2 as 2.2 km, +48 h, updated every
  3 h, with 65 vertical atmosphere levels.
- AROME: useful for recent French forecast demos and API comparison. For 2022
  historical station validation, archive availability and licensing still need
  a concrete download path before including it as a required baseline.

Official references checked:

- DWD ICON/ICON-D2 open forecast data:
  https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html
- EFFIS fire danger forecast:
  https://forest-fire.emergency.copernicus.eu/about-effis/technical-background/fire-danger-forecast
- ERA5-Land CDS:
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- Meteo-France AROME open-data meteograms:
  https://www.data.gouv.fr/datasets/meteogrammes-modeles-arome

## Immediate Next Step

Implement the station v2 input builder. Without it, any
`DownscaleWind_DownscalRain_FWI` score would be fake. The rest of the station
pipeline is ready: station selection, rain correction, ERA5 meteo baselines, FWI
tables, metrics, and the UVT model runner.
