# Hub-height out-of-sample benchmark — data provenance (started 2026-09-16)

Goal: first action of paper 1 — check that surrogate v3 + M_I8 gives useful wind at hub height
on sites never seen in training, against ERA5, NEWA mesoscale (3 km hourly), Global Wind Atlas
(250 m climatology) and, where scored, FuXi-CFD.

## Sites

| site | lat | lon | what | heights | period | status |
|---|---|---|---|---|---|---|
| Alaiz (ALEX17 MP5 mast, CENER/NEWA) | 42.695 | -1.558 | 118 m mast, complex ridge ~1000 m | WS/WD 40, 78, 90, 102, 118 m; T/RH 2-113 m | 2017-07 → 2019-07 | data on NEWA SFTP, credentials via jmsq@dtu.dk (CC BY 4.0) — NOT yet obtained |
| Penmanshiel (Cubico, Zenodo 8253010) | 55.905 | -2.305 | 14 Senvion MM82, hub **59 m**, rotor 82 m, elev ~200-212 m | nacelle WS + power 10-min | 2016 → 2022 (2018 fetched) | static CSV + 2018 SCADA zips downloading (CC BY 4.0) |
| Kelmarsh (Cubico, Zenodo 5841834) | 52.400 | -0.947 | 6 Senvion MM92, flat control | | 2016 → 2024 | not fetched yet |

## Reference products fetched

- ERA5 hourly (CDS, 10 pressure levels u/v/T/q/z + u10/v10/t2m/d2m):
  `data/raw/era5_alaiz_2017_2019.zarr` (bbox 41.5,-3.0,44.0,0.0) and
  `data/raw/era5_penmanshiel_2018.zarr` (bbox 54.5,-3.75,57.0,-0.75). 7-day CDS requests
  (31-day requests hit "cost limits exceeded").
- NEWA mesoscale time series API (`wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1`,
  1989-2022, heights 50/75/100/150/200/250/500 m, NetCDF): `data/raw/newa_ts/newa_alaiz_2017-07_2019-07.nc`,
  `data/raw/newa_ts/newa_penmanshiel_2018.nc`.
- Global Wind Atlas 3 mean speed (GEE community asset `sat-io/open-datasets/global_wind_atlas/wind-speed`,
  250 m): `data/raw/gwa_points_hub_sites.json`.
  Alaiz 10/50/100/150/200 m = 6.46 / 8.99 / 10.10 / 11.05 / 11.73 m/s ;
  Penmanshiel = 6.20 / 8.74 / 10.03 / 11.06 / 11.81 ; Kelmarsh = 4.84 / 7.19 / 8.53 / 9.81 / 10.67.

## First competitor number (2026-09-16, no model yet)

NEWA mesoscale (3 km) log-interpolated 50/75 m → hub 59 m vs Penmanshiel farm-mean nacelle wind,
8 736 hours of 2018 (≥10 turbines reporting): obs mean 7.04 m/s, NEWA 8.30 m/s →
**bias +1.26, MAE 2.15, RMSE 2.76 m/s, corr 0.805**. Bias is +1.5 to +1.7 below 8 m/s and −0.33
above 12 m/s. Caveat: nacelle anemometer sits behind the rotor (typically reads a few % low),
so part of the +1.26 is instrument, not model. GWA 50 m climatology at the site = 8.74 m/s.
Pairings: `data/inference/penmanshiel_turbines_v1.parquet` (builder
`services/validation/build_penmanshiel_pairings.py`, 122 029 hourly rows, 14 turbines).

## Scripts

- `services/validation/build_penmanshiel_pairings.py` → hourly turbine pairings (multiheight schema + power).
- `services/validation/run_hub_height_inference.py` + `configs/validation/hub_height_{penmanshiel,alaiz}.yaml`
  → RAW v3 vs M_I8 at each unit/hour, materialising inputs on the fly (derived from run_masts_M_K2.py).
  Cost control: `--station-filter penmanshiel_T01,penmanshiel_T08,penmanshiel_T15 --hour-stride 1`
  (~26k entries ≈ 3 GB cache) before the full 14-turbine run (~15 GB).
- `services/validation/score_hub_height.py` → metrics per product, per observed wind class, monthly
  capacity factor via the empirical site power curve. Run now with NEWA + GWA only; add `--predictions`
  once the model has run and `--era5-u100` once the 100 m store is downloaded.

## First model results (2026-09-17, Aqua jobs 25413254 + 25413485)

Penmanshiel, 3 turbines (T01/T08/T15), every 3rd hour, Jan-Sep 2018 (partial ERA5 store), 2 178 site-mean hours, hub 59 m.

| product | bias | MAE | corr | capacity factor |
|---|---|---|---|---|
| observed | | | | 0.289 |
| ours raw (v3) | -0.41 | 1.64 | 0.833 | 0.290 |
| ours calibrated (M_I8) | +1.08 | 1.92 | 0.820 | 0.393 |
| ERA5 10 m (driver) | -0.44 | 1.52 | 0.839 | 0.275 |
| NEWA 3 km | +1.02 | 2.03 | 0.795 | 0.407 |
| GWA climatology | +2.27 | | | |

Predicted mean profile (raw): 10 m 4.69 / 30 m 5.55 / 60 m 6.37 / 100 m 7.06 / 150 m 7.61 m/s → shear exponent
10-100 m = **0.175** (physically sound). ERA5 10 m = 6.34 m/s: the coastal cell is partly marine, so the surrogate
SLOWS the 10 m wind by 26 % and rebuilds the profile; "raw 59 m ≈ ERA5 10 m" is a coincidence of this site, not a flat profile.
Calibrated profile: 6.27 / 7.15 / 7.85 / 9.16 / 9.75 → M_I8 adds ≈ +1.5 m/s at every height (learned on stations where the
raw surrogate under-predicts). Here it over-corrects below 12 m/s (bias +1.2) and helps only above 12 m/s
(bias -2.73 → -0.13; obs>8 m/s: obs 11.32, raw 9.99, calibrated 11.92). Same signature as the Perdigão calm-regime over-correction.
Nacelle anemometer check: the empirical site power curve matches the MM92 manufacturer curve scaled by rotor area
(82/92.5)^2 within a few % at 6, 8 and 10 m/s → nacelle wind is reliable.
GPU note: batch 32 OOMs on a 40 GB A100 (two 180x180x32 forwards) → batch 8 + expandable_segments.

## ICOS out-of-sample towers (2026-09-17) — Karlsruhe KIT, JJA 2020, job 25413849

Towers never used in training, ICOS ATC meteo (CC BY 4.0), ingested with the renewed token: KIT (30/60/100/200 m),
KRE (10/50/125 m), TOH (10/76/110/147 m; 76 m reads higher than 110/147 m → exposure issue, score per height only),
OXK (wind at 163 m only). Pairings `data/inference/icos_oos_towers_v1.parquet` (29 056 rows), builder
`services/validation/build_icos_oos_pairings.py`, per-height scorer `services/validation/score_towers_hub_height.py`,
PBS `configs/hpc/hub_height_icos_tower.pbs` (qsub -v TOWER=…,ERA5=…).

KIT MAE (m/s), n=2424 h per height:

| height | obs mean | ours raw | ours calibrated | NEWA 3 km | ERA5 10 m |
|---|---|---|---|---|---|
| 30 m | 1.90 | 0.99 | 1.41 | n/a | 0.78 |
| 60 m | 3.25 | 1.20 | **0.97** | 1.46 | 0.99 |
| 100 m | 4.01 | 1.52 | **1.29** | 1.55 | 1.63 |
| 200 m | 5.17 | 2.03 | **1.50** | 1.68 | 2.73 |

Raw predicted profile 10→200 m: 1.86 / 2.20 / 2.48 / 2.78 / 3.07 / 3.34 (shear exponent 30-200 m ≈ 0.22) vs observed
1.90 → 5.17 (≈ 0.53, summer nocturnal stable shear over forest/valley): neutral RANS cannot produce that shear, so the raw
surrogate under-predicts aloft (bias −1.2 at 100 m, −1.8 at 200 m). M_I8 restores most of it (4.51 at 100 m) and beats NEWA
at 60/100/200 m. Opposite verdict to Penmanshiel (windy coastal hills, calibration over-corrects): the calibration carries
the regime of its training towers (inland, summer). GWA is an ANNUAL climatology scored against SUMMER obs here
(+1.5 to +2.1 m/s): not a fair comparison, report only with matching periods.

## Model side

Checkpoints live on Aqua only: `~/dsw/data/models/surrogate_v3_vit_base_agl200_k32/best.pt`,
`~/dsw/data/models/surrogate_v3_devine_M_I8_multiheight/best.pt`. Plan: scp both locally
(~150 MB) and run inference on the Mac (MPS) with a mast-pairings script derived from
`data/validation/crest_deficit/run_masts_M_K2.py` (multi-height, fixed hourly timestamps).

## Caveats to carry into the paper

- Nacelle anemometers are rotor-perturbed: validate Penmanshiel on power via the MM82 curve,
  or on nacelle WS only with the manufacturer transfer function.
- NEWA meso is 3 km WRF driven by ERA5 (1989-2018 run extended to 2022); GWA is a 10-year
  climatology → compare climatologies on matching periods only.
- Alaiz mast at 40-118 m sits on a ridge with >500 m relief: also outside the training population B
  (steep >20°) only if local slope exceeds it — check slope at the mast from the DEM.
