# DownscalRain Tabular Validation - 2022 Cache

Date: 2026-05-06

## Run

Command:

```bash
python services/module3-precip/evaluate_downscalrain_tabular.py \
  --dataset data/raw/precip_correction_cache/dataset_2022.parquet \
  --output-dir data/validation/downscalrain \
  --folds 5 \
  --save-predictions \
  --model-dir data/models/downscalrain_tabular_v1
```

Data:

- 431 stations.
- 155,434 station-days after lag-feature filtering.
- Year: 2022.
- Validation: 5-fold station-grouped split, so no station appears in both train
  and test for a fold.

Model:

- Baseline: raw IMERG daily precipitation at station.
- DownscalRain tabular: occurrence classifier + wet-day log-amount regressor.
- Features: IMERG D0, D-1, D-2, 3-day/7-day IMERG accumulations, terrain,
  aspect, TPI, lat/lon, month and day-of-year cyclic encodings.
- Heavy-rain balanced amount loss: `heavy_rain_weight=4.0`,
  `rain_amount_weight=0.015`.

## Main Metrics

| Subset | Model | RMSE mm/day | MAE mm/day | Bias mm/day | Corr | Wet recall | Heavy recall >10mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | IMERG raw | 4.281 | 1.765 | 0.231 | 0.580 | 0.676 | 0.476 |
| All | DownscalRain tabular | 3.368 | 1.426 | 0.020 | 0.667 | 0.835 | 0.421 |
| Fire season | IMERG raw | 4.284 | 1.718 | 0.208 | 0.587 | 0.731 | 0.459 |
| Fire season | DownscalRain tabular | 3.601 | 1.493 | 0.065 | 0.656 | 0.848 | 0.418 |

## Readout

- All-year RMSE improves by 21.3%.
- All-year MAE improves by 19.2%.
- All-year mean bias is nearly removed: +0.231 to +0.020 mm/day.
- Fire-season RMSE improves by 16.0%.
- Fire-season MAE improves by 13.1%.
- Wet-day recall improves strongly: 0.676 to 0.835 all-year, 0.731 to 0.848
  fire season.
- Heavy-rain recall remains slightly below raw IMERG. This is the main remaining
  weakness of the tabular model and supports moving to the CNN patch model.

## Outputs

- `downscalrain_tabular_metrics.csv`
- `downscalrain_tabular_fold_metrics.csv`
- `downscalrain_tabular_summary.json`
- `downscalrain_tabular_predictions.parquet`
- `figures/downscalrain_global_metrics.png`
- `figures/downscalrain_elevation_metrics.png`
- `figures/downscalrain_monthly_rmse.png`
- `figures/downscalrain_scatter_hexbin.png`
- final model artifacts in `data/models/downscalrain_tabular_v1`
