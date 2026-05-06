# Module 3 — Precipitation Bias Correction

Corrects satellite precipitation (GPM IMERG, 10 km) using ground station observations
and terrain features. Produces station-quality daily precipitation at any point in Europe.

## Pipeline

```
IMERG / ERA5-Land / Terrain patches  +  station metadata
                          │
        XGBoost baseline  │  DownscalRain CNN patch-to-point
                          │
                          ▼
              Corrected precipitation for FWI rain24
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
│   ├── training.yaml          # XGBoost baseline config
│   └── downscalrain_cnn.yaml  # CNN patch-to-point config
├── src/
│   ├── stations.py            # download & parse GHCN-D, ECA&D, MF stations
│   ├── imerg.py               # download IMERG at station locations via GEE
│   ├── terrain.py             # extract SRTM features at station locations via GEE
│   ├── dataset.py             # merge stations + IMERG + terrain → training DataFrame
│   ├── model.py               # XGBoost train, predict, spatial CV
│   ├── patch_dataset.py       # patch dataset format and station-group splits
│   └── downscalrain.py        # CNN patch-to-point model/loss/metrics
├── train.py                   # CLI: download → merge → train → evaluate
├── build_patch_dataset.py     # CLI: station labels + gridded sources → patches
├── train_downscalrain.py      # CLI: train CNN patch-to-point model
├── predict_downscalrain.py    # CLI: apply trained CNN to patch dataset
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

# Build CNN patches after filling gridded source paths in the config
python build_patch_dataset.py --config configs/downscalrain_cnn.yaml

# Train DownscalRain CNN
python train_downscalrain.py --config configs/downscalrain_cnn.yaml

# Apply DownscalRain CNN to a patch dataset
python predict_downscalrain.py \
    --checkpoint ../../data/models/downscalrain_cnn_v1/best.pt \
    --dataset ../../data/processed/downscalrain/patches_v1 \
    --output ../../data/processed/downscalrain/predictions.parquet
```

## Validation

Held-out validation on:
- **Spatial CV**: 5-fold grouped by station (no geographic leakage)
- **OMM/SYNOP/METAR stations**: FWI-ready professional station validation
- **Regional splits**: Mediterranean, Alpine/Pyrenees, Atlantic, Nordic, etc.
- **COMEPHORE**: 1 km radar+gauge merged product for France when available

The XGBoost path is kept as a reproducible baseline and calibration layer. The
CNN path should only become the paper product if it wins on station-grouped and
region-held-out validation, especially wet/dry skill and Mediterranean fire-season
bias.
