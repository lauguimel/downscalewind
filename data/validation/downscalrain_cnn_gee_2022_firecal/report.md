# DownscalRain CNN validation

## all

| model | n | RMSE | MAE | bias | corr | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 156296 | 4.292 | 1.771 | 0.240 | 0.579 | 0.677 | 0.477 |
| raw_era5land_center | 156296 | 4.087 | 1.742 | 0.305 | 0.557 | 0.770 | 0.432 |
| downscalrain_cnn | 156296 | 3.453 | 1.276 | -0.631 | 0.659 | 0.540 | 0.391 |

## test

| model | n | RMSE | MAE | bias | corr | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 23566 | 4.288 | 1.760 | 0.292 | 0.537 | 0.658 | 0.444 |
| raw_era5land_center | 23566 | 3.992 | 1.697 | 0.385 | 0.544 | 0.768 | 0.436 |
| downscalrain_cnn | 23566 | 3.434 | 1.267 | -0.567 | 0.606 | 0.522 | 0.339 |

## test_fire_season

| model | n | RMSE | MAE | bias | corr | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 11904 | 4.376 | 1.741 | 0.294 | 0.531 | 0.720 | 0.430 |
| raw_era5land_center | 11904 | 4.365 | 1.859 | 0.401 | 0.496 | 0.750 | 0.372 |
| downscalrain_cnn | 11904 | 3.655 | 1.292 | -0.624 | 0.593 | 0.536 | 0.297 |
