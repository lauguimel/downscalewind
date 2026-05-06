# DownscalRain Fire-Period Precipitation Validation

Dry threshold: station rain <= 1 mm/day.
Wet threshold: model rain > 1 mm/day.

These subsets target FWI failure modes: dry-season false precipitation and gridded-product halos.

## test_jjas_mediterranean

Held-out stations, Mediterranean 35-45.5N, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 1085 | 925 | 5.411 | 1.905 | 0.169 | 0.251 | 0.137 | 0.904 | 0.463 | 0.263 |
| raw_era5land_center | 1085 | 925 | 5.653 | 1.959 | 0.348 | 0.248 | 0.165 | 0.837 | 0.625 | 0.184 |
| downscalrain_cnn | 1085 | 925 | 4.403 | 1.423 | -0.302 | 0.339 | 0.141 | 0.534 | 0.650 | 0.105 |

## test_jjas_mediterranean_dry

Station dry days inside the held-out Mediterranean fire window.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 925 | 925 | 3.227 | 0.900 | 0.850 | 0.227 | 0.137 | 0.904 | 0.000 | 0.000 |
| raw_era5land_center | 925 | 925 | 2.689 | 0.822 | 0.783 | 0.216 | 0.165 | 0.837 | 0.000 | 0.000 |
| downscalrain_cnn | 925 | 925 | 1.403 | 0.510 | 0.480 | 0.253 | 0.141 | 0.534 | 0.000 | 0.000 |

## test_jjas_mediterranean_imerg_halo

Station dry, IMERG wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 127 | 127 | 8.695 | 6.094 | 6.094 | 0.140 | 1.000 | 6.254 | 0.000 | 0.000 |
| raw_era5land_center | 127 | 127 | 6.401 | 3.427 | 3.386 | 0.108 | 0.583 | 3.546 | 0.000 | 0.000 |
| downscalrain_cnn | 127 | 127 | 3.206 | 2.000 | 1.966 | 0.116 | 0.575 | 2.126 | 0.000 | 0.000 |

## test_jjas_mediterranean_era5_halo

Station dry, ERA5-Land wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 153 | 153 | 7.267 | 3.757 | 3.682 | 0.148 | 0.484 | 3.831 | 0.000 | 0.000 |
| raw_era5land_center | 153 | 153 | 6.592 | 4.483 | 4.483 | 0.108 | 1.000 | 4.631 | 0.000 | 0.000 |
| downscalrain_cnn | 153 | 153 | 3.066 | 1.905 | 1.887 | 0.128 | 0.582 | 2.035 | 0.000 | 0.000 |

## test_puechabon_150km_jjas

Held-out stations within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 244 | 209 | 4.625 | 1.572 | 0.503 | 0.480 | 0.120 | 0.794 | 0.686 | 0.625 |
| raw_era5land_center | 244 | 209 | 4.604 | 1.772 | 0.473 | 0.323 | 0.196 | 0.875 | 0.800 | 0.125 |
| downscalrain_cnn | 244 | 209 | 3.952 | 1.385 | -0.054 | 0.418 | 0.215 | 0.708 | 0.743 | 0.250 |

## test_puechabon_150km_jjas_dry

Held-out dry station-days within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 209 | 209 | 2.906 | 0.767 | 0.734 | 0.544 | 0.120 | 0.794 | 0.000 | 0.000 |
| raw_era5land_center | 209 | 209 | 2.179 | 0.833 | 0.816 | 0.343 | 0.196 | 0.875 | 0.000 | 0.000 |
| downscalrain_cnn | 209 | 209 | 1.406 | 0.667 | 0.649 | 0.243 | 0.215 | 0.708 | 0.000 | 0.000 |

## puechabon_100km_jjas_all

Diagnostic only: nearest stations are train/val, not held-out.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 603 | 517 | 6.891 | 2.150 | -0.108 | 0.492 | 0.118 | 0.931 | 0.570 | 0.519 |
| raw_era5land_center | 603 | 517 | 8.294 | 2.789 | 0.822 | 0.351 | 0.176 | 1.401 | 0.721 | 0.556 |
| downscalrain_cnn | 603 | 517 | 6.998 | 1.832 | -0.362 | 0.449 | 0.191 | 0.728 | 0.849 | 0.444 |

## puechabon_100km_jjas_imerg_halo_all

Diagnostic only: Puéchabon-near IMERG false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 61 | 61 | 11.137 | 7.353 | 7.353 | -0.032 | 1.000 | 7.493 | 0.000 | 0.000 |
| raw_era5land_center | 61 | 61 | 10.449 | 6.971 | 6.913 | 0.041 | 0.721 | 7.052 | 0.000 | 0.000 |
| downscalrain_cnn | 61 | 61 | 4.849 | 3.099 | 3.063 | -0.033 | 0.705 | 3.202 | 0.000 | 0.000 |

## puechabon_100km_jjas_era5_halo_all

Diagnostic only: Puéchabon-near ERA5-Land false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 91 | 91 | 9.046 | 4.680 | 4.602 | -0.024 | 0.484 | 4.742 | 0.000 | 0.000 |
| raw_era5land_center | 91 | 91 | 10.258 | 7.507 | 7.507 | 0.121 | 1.000 | 7.646 | 0.000 | 0.000 |
| downscalrain_cnn | 91 | 91 | 4.317 | 2.770 | 2.755 | 0.030 | 0.703 | 2.895 | 0.000 | 0.000 |
