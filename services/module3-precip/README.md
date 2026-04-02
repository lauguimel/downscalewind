# Module 3 — Precipitation Bias Correction

Corrects satellite precipitation (GPM IMERG, 10 km) using ground station observations
and terrain features. Produces station-quality daily precipitation at any point in Europe.

## Pipeline

```
IMERG (10km, global)  +  Terrain (SRTM 30m)  +  Season/Location
                          │
                    XGBoost correction
                          │
                          ▼
              Corrected precipitation (~10km, debiased)
```

## Data Sources

| Source | Role | Coverage | Resolution |
|--------|------|----------|------------|
| GPM IMERG V07 | Input (satellite) | Global | 0.1° / 30 min |
| GHCN-D | Training labels | Global, ~2000 stn Europe | Daily |
| ECA&D | Training labels (dense) | Europe, ~3500 stn | Daily |
| MF Synop/Clim | Training labels (France) | France, ~3000 stn | Daily |
| SRTM | Terrain features | Global | 30 m |
| ICOS / COMEPHORE | Validation only | Europe / France | Point / 1 km |

## Structure

```
services/module3-precip/
├── README.md
├── configs/
│   └── training.yaml          # hyperparameters, station filters, bbox
├── src/
│   ├── stations.py            # download & parse GHCN-D, ECA&D, MF stations
│   ├── imerg.py               # download IMERG at station locations via GEE
│   ├── terrain.py             # extract SRTM features at station locations via GEE
│   ├── dataset.py             # merge stations + IMERG + terrain → training DataFrame
│   └── model.py               # XGBoost train, predict, spatial CV
├── train.py                   # CLI: download → merge → train → evaluate
├── predict.py                 # CLI: apply model at arbitrary lat/lon
└── tests/
    └── test_model.py
```

## Usage

```bash
# Train on 2022 data, 500 stations
python train.py --year 2022 --max-stations 500 --output ../../data/models/precip_correction/

# Predict corrected precip at a point
python predict.py --model ../../data/models/precip_correction/ \
    --lat 43.74 --lon 3.60 --start 2022-01-01 --end 2022-12-31
```

## Validation

Held-out validation on:
- **Spatial CV**: 5-fold grouped by station (no geographic leakage)
- **ICOS stations**: independent, not in training
- **COMEPHORE**: 1 km radar+gauge merged product (France only)
