# FWI station UVT inference check — 2026-05-12

## Frozen run

- Stations: 20 SYNOP/OMM stations.
- UVT inference cases: 400 = 20 stations x 20 fire-weather timestamps.
- Terrain input: COPDEM GLO-30, 6 km x 6 km, 180 x 180, written as input-only v2 `grid.zarr`.
- Surrogate: `surrogate_v2_vit_base_resid_s4_geo_agl100_k24/best.pt`.
- Output: `data/validation/fwi_station_audit_2022_n20/downscalewind_uvt_station_agl100_copdem.csv`.
- Runtime: 400/400 inferred on Aqua A100, walltime 5m52, mean per-case inference 0.106 s, median 0.086 s, p95 0.095 s.

## Fair FWI comparison on the 400 UVT cases

| product | n | RMSE | MAE | bias | corr |
|---|---:|---:|---:|---:|---:|
| OBS_station_FWI | 400 | 0.000 | 0.000 | 0.000 | 1.000 |
| StationMet_IMERGFireCorrected_FWI | 400 | 2.625 | 1.644 | 0.737 | 0.942 |
| ERA5_met_IMERGFireCorrected_FWI | 400 | 5.307 | 3.791 | -3.310 | 0.748 |
| ERA5_met_ObsRain_FWI | 400 | 5.458 | 3.985 | -3.843 | 0.781 |
| DownscaleWind_DownscalRain_FWI | 400 | 6.307 | 4.729 | -1.118 | 0.354 |
| ERA5_met_DownscalRainDirect_FWI | 400 | 6.322 | 4.631 | -4.351 | 0.676 |
| ERA5_met_IMERG_FWI | 400 | 7.108 | 5.339 | -5.241 | 0.630 |
| ERA5LandRain_FWI | 400 | 7.334 | 5.409 | -5.085 | 0.526 |

Interpretation: the rain correction is useful, but the current station UVT inference is not publishable as a station-validation result. It beats raw ERA5-Land FWI on the selected 400 cases, but it is worse than ERA5 meteorology plus the IMERG fire correction. The failure mode is wind: DownscaleWind further damps an already low ERA5 wind at airport/SYNOP stations.

## UVT event metrics against station observations

| product | T RMSE (C) | RH RMSE (%) | wind RMSE (m/s) | wind bias (m/s) |
|---|---:|---:|---:|---:|
| ERA5 | 2.474 | 7.659 | 2.370 | -1.852 |
| DownscaleWind | 2.621 | 7.905 | 3.176 | -2.798 |

The model improves the T/RH correlation slightly, but the wind magnitude bias is too negative for FWI. This is consistent with the surrogate learning a CFD teacher near complex terrain, not a calibrated station-observation product. The station inputs also use a constant `z0_eff=0.05`, so airport roughness/exposure is not represented.

## Immediate conclusion

Do not use the current DownscaleWind station-FWI result as the main validation claim. For the paper/API path, keep:

- rain correction as a strong station-backed result;
- UVT surrogate as a CFD-teacher/downscaling product;
- station FWI as a calibration/validation workstream requiring wind exposure/roughness handling or a post-processing calibration split.

## Surface-residual rerun — 2026-05-13

After fixing the residual baseline height mismatch, the AGL100 checkpoint was fine-tuned with `residual_baseline_mode=surface` and rerun on the same 400 station cases.

- Surrogate: `surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`.
- Output: `data/validation/fwi_station_audit_2022_n20/downscalewind_uvt_station_agl100_surface_copdem.csv`.
- Runtime: 400/400 inferred on Aqua CPU fallback, walltime 16m34, mean per-case inference 1.49 s.

Fair FWI comparison on the 400 UVT cases:

| product | n | RMSE | MAE | bias | corr |
|---|---:|---:|---:|---:|---:|
| OBS_station_FWI | 400 | 0.000 | 0.000 | 0.000 | 1.000 |
| StationMet_IMERGFireCorrected_FWI | 400 | 2.625 | 1.644 | 0.737 | 0.942 |
| ERA5_met_IMERGFireCorrected_FWI | 400 | 5.307 | 3.791 | -3.310 | 0.748 |
| ERA5_met_ObsRain_FWI | 400 | 5.458 | 3.985 | -3.843 | 0.781 |
| DownscaleWind_DownscalRain_FWI | 400 | 6.112 | 4.519 | -1.074 | 0.393 |
| ERA5_met_IMERG_FWI | 400 | 7.108 | 5.339 | -5.241 | 0.630 |
| ERA5LandRain_FWI | 400 | 7.334 | 5.409 | -5.085 | 0.526 |

UVT event metrics against station observations:

| product | T RMSE (C) | RH RMSE (%) | wind RMSE (m/s) | wind bias (m/s) |
|---|---:|---:|---:|---:|
| ERA5 | 2.474 | 7.659 | 2.370 | -1.852 |
| DownscaleWind old pressure-index residual | 2.621 | 7.905 | 3.176 | -2.798 |
| DownscaleWind surface residual | 2.424 | 7.844 | 3.113 | -2.753 |

Interpretation: the residual-baseline fix is real but it is not the dominant station-validation failure. It slightly improves T/RH and FWI relative to the invalid pressure-index residual run, but the wind is still too damped at SYNOP/airport stations, so ERA5 meteorology plus IMERG fire correction remains better for station FWI.
