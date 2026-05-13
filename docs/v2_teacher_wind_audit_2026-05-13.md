# v2 teacher wind audit — 2026-05-13

Question: is the low station wind caused by the surrogate training, or is the OpenFOAM teacher already low near the ground?

## Protocol

- Dataset: `/scratch/maitreje/dsw/training_v2`.
- Sample: 500 v2 `grid.zarr` cases, seed 13.
- Heights: 2, 10, 50, 100 m AGL.
- Spatial views: whole domain, central 2 x 2 km crop, centre column, upstream/downstream internal edges.
- References:
  - ERA5 centre-grid `u10/v10`;
  - reconstructed `inflow.json` log-law/ERA5 profile.

Script:

```bash
python services/module2a-cfd/analysis/audit_v2_teacher_wind.py \
  --data-dir /scratch/maitreje/dsw/training_v2 \
  --output data/validation/v2_teacher_wind_audit_sample500_edges.csv \
  --summary-output data/validation/v2_teacher_wind_audit_sample500_edges_summary.csv \
  --limit 500 --seed 13 --heights 2,10,50,100 --crop-km 2
```

## Main result

At 10 m AGL, for the full 500-case sample:

- median central-crop CFD / ERA5 u10 = 0.879;
- median central-crop CFD / inflow = 0.877.

For windy cases, the damping is stronger:

| subset | n | mean ERA5 u10 | mean CFD crop 10 m | median CFD/ERA5 | median CFD/inflow |
|---|---:|---:|---:|---:|---:|
| ERA5 u10 >= 3 m/s | 108 | 4.09 | 3.15 | 0.767 | 0.779 |
| ERA5 u10 >= 5 m/s | 20 | 5.72 | 4.17 | 0.696 | 0.764 |

This matches the station inference failure mode: on the 400 station cases, ERA5 wind mean was 4.14 m/s and DownscaleWind surface-residual mean was 3.24 m/s, i.e. about 0.78 x ERA5.

## Boundary vs internal field

On a strongly damped case (`ct_d_fire_0170_case_ts014`):

- ERA5 u10 = 5.04 m/s.
- inflow 10 m = 5.42 m/s.
- boundaryData lateral U near low AGL is not weak: section means are about 5.6-7.0 m/s.
- CFD centre 10 m = 2.53 m/s.

The low speed is therefore not explained by the stored station inference or by weak lateral boundary values. It appears inside the OpenFOAM solve.

Internal-edge diagnostic at 10 m for windy cases:

| subset | upstream edge / inflow median | central crop / inflow median | downstream edge / inflow median |
|---|---:|---:|---:|
| ERA5 u10 >= 3 m/s | 0.853 | 0.779 | 0.812 |
| ERA5 u10 >= 5 m/s | 0.848 | 0.764 | 0.845 |

The internal upstream edge is already below the inflow, and the central crop is lower again.

## Training vs teacher

A small CPU check on 5 test cases, central 2 km crop, with `surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt` showed positive skill against the surface residual baseline and no obvious negative u/v bias relative to the teacher:

| variable | RMSE | baseline RMSE | bias |
|---|---:|---:|---:|
| u | 1.220 | 1.613 | +0.134 |
| v | 0.921 | 1.125 | +0.420 |
| T | 1.477 | 2.986 | -0.606 |
| q | 4.91e-4 | 7.73e-4 | -2.18e-4 |

This is a tiny sample because CPU ViT full-field evaluation is slow, but it supports the audit conclusion: the dominant station wind bias is teacher/physics/domain related, not primarily a surrogate training bug.

## Station overlap

The selected 20 OMM/FWI stations are not collocated with v2 training sites. Nearest v2 site distances:

- closest: Perpignan -> `ct_d_fire_0045`, 13.9 km;
- Millau -> `ct_f_wind_onshore_0021`, 14.2 km;
- Montelimar -> `ct_c_morpho_0014`, 17.7 km;
- no selected OMM station has a v2 site within 10 km.

Direct teacher-vs-station validation therefore requires generating new OpenFOAM cases centred on station locations, or using a separate campaign that was explicitly built around measured towers.

## Interpretation

The current teacher behaves like a terrain/RANS downscaling product with substantial near-ground momentum loss over finite fetch. For station FWI, especially airport/SYNOP stations in windy regimes, it underestimates 10 m wind more than ERA5.

Likely causes to test next:

- horizontal ABL degradation over terrain/noSlip fetch despite lateral log-law BC;
- absence or insufficient strength of a geostrophic/pressure-gradient momentum driver;
- wall-function/effective roughness mismatch;
- vertical discretisation and AGL convention near the first cells;
- station exposure mismatch, because airports are not complex-terrain mast points.

The next corrective experiment should be an OpenFOAM physics/BC audit on a flat-terrain or station-centred case, not another blind surrogate fine-tune.
