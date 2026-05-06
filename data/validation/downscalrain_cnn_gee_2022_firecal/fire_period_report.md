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
| downscalrain_cnn | 1085 | 925 | 4.470 | 1.284 | -0.630 | 0.314 | 0.058 | 0.300 | 0.431 | 0.105 |
| downscalrain_cnn_firecalibrated | 1085 | 925 | 4.495 | 1.264 | -0.705 | 0.308 | 0.054 | 0.232 | 0.431 | 0.105 |
| downscalrain_cnn_fireguard | 1085 | 925 | 4.694 | 1.249 | -1.042 | 0.202 | 0.005 | 0.062 | 0.075 | 0.105 |

## test_jjas_mediterranean_dry

Station dry days inside the held-out Mediterranean fire window.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 925 | 925 | 3.227 | 0.900 | 0.850 | 0.227 | 0.137 | 0.904 | 0.000 | 0.000 |
| raw_era5land_center | 925 | 925 | 2.689 | 0.822 | 0.783 | 0.216 | 0.165 | 0.837 | 0.000 | 0.000 |
| downscalrain_cnn | 925 | 925 | 1.209 | 0.297 | 0.246 | 0.242 | 0.058 | 0.300 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 925 | 925 | 1.204 | 0.252 | 0.178 | 0.226 | 0.054 | 0.232 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 925 | 925 | 0.857 | 0.113 | 0.008 | 0.082 | 0.005 | 0.062 | 0.000 | 0.000 |

## test_jjas_mediterranean_imerg_halo

Station dry, IMERG wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 127 | 127 | 8.695 | 6.094 | 6.094 | 0.140 | 1.000 | 6.254 | 0.000 | 0.000 |
| raw_era5land_center | 127 | 127 | 6.401 | 3.427 | 3.386 | 0.108 | 0.583 | 3.546 | 0.000 | 0.000 |
| downscalrain_cnn | 127 | 127 | 3.184 | 1.550 | 1.471 | 0.190 | 0.331 | 1.630 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 127 | 127 | 3.181 | 1.486 | 1.334 | 0.183 | 0.315 | 1.494 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 127 | 127 | 2.280 | 0.591 | 0.291 | 0.068 | 0.039 | 0.450 | 0.000 | 0.000 |

## test_jjas_mediterranean_era5_halo

Station dry, ERA5-Land wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 153 | 153 | 7.267 | 3.757 | 3.682 | 0.148 | 0.484 | 3.831 | 0.000 | 0.000 |
| raw_era5land_center | 153 | 153 | 6.592 | 4.483 | 4.483 | 0.108 | 1.000 | 4.631 | 0.000 | 0.000 |
| downscalrain_cnn | 153 | 153 | 2.827 | 1.264 | 1.185 | 0.151 | 0.255 | 1.334 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 153 | 153 | 2.824 | 1.184 | 1.043 | 0.147 | 0.248 | 1.191 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 153 | 153 | 2.081 | 0.507 | 0.226 | 0.068 | 0.033 | 0.374 | 0.000 | 0.000 |

## test_puechabon_150km_jjas

Held-out stations within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 244 | 209 | 4.625 | 1.572 | 0.503 | 0.480 | 0.120 | 0.794 | 0.686 | 0.625 |
| raw_era5land_center | 244 | 209 | 4.604 | 1.772 | 0.473 | 0.323 | 0.196 | 0.875 | 0.800 | 0.125 |
| downscalrain_cnn | 244 | 209 | 3.880 | 1.072 | -0.656 | 0.506 | 0.053 | 0.218 | 0.429 | 0.250 |
| downscalrain_cnn_firecalibrated | 244 | 209 | 3.902 | 1.003 | -0.808 | 0.507 | 0.038 | 0.066 | 0.429 | 0.250 |
| downscalrain_cnn_fireguard | 244 | 209 | 4.291 | 1.065 | -1.065 | 0.298 | 0.000 | 0.000 | 0.057 | 0.250 |

## test_puechabon_150km_jjas_dry

Held-out dry station-days within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 209 | 209 | 2.906 | 0.767 | 0.734 | 0.544 | 0.120 | 0.794 | 0.000 | 0.000 |
| raw_era5land_center | 209 | 209 | 2.179 | 0.833 | 0.816 | 0.343 | 0.196 | 0.875 | 0.000 | 0.000 |
| downscalrain_cnn | 209 | 209 | 0.404 | 0.208 | 0.158 | 0.277 | 0.053 | 0.218 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 209 | 209 | 0.351 | 0.102 | 0.006 | 0.222 | 0.038 | 0.066 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 209 | 209 | 0.189 | 0.059 | -0.059 | nan | 0.000 | 0.000 | 0.000 | 0.000 |

## puechabon_100km_jjas_all

Diagnostic only: nearest stations are train/val, not held-out.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 603 | 517 | 6.891 | 2.150 | -0.108 | 0.492 | 0.118 | 0.931 | 0.570 | 0.519 |
| raw_era5land_center | 603 | 517 | 8.294 | 2.789 | 0.822 | 0.351 | 0.176 | 1.401 | 0.721 | 0.556 |
| downscalrain_cnn | 603 | 517 | 7.184 | 1.733 | -0.883 | 0.413 | 0.075 | 0.367 | 0.605 | 0.259 |
| downscalrain_cnn_firecalibrated | 603 | 517 | 7.190 | 1.724 | -0.945 | 0.413 | 0.072 | 0.312 | 0.593 | 0.259 |
| downscalrain_cnn_fireguard | 603 | 517 | 7.991 | 1.771 | -1.626 | 0.091 | 0.002 | 0.025 | 0.058 | 0.111 |

## puechabon_100km_jjas_imerg_halo_all

Diagnostic only: Puéchabon-near IMERG false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 61 | 61 | 11.137 | 7.353 | 7.353 | -0.032 | 1.000 | 7.493 | 0.000 | 0.000 |
| raw_era5land_center | 61 | 61 | 10.449 | 6.971 | 6.913 | 0.041 | 0.721 | 7.052 | 0.000 | 0.000 |
| downscalrain_cnn | 61 | 61 | 4.084 | 2.256 | 2.133 | -0.022 | 0.426 | 2.272 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 61 | 61 | 4.084 | 2.217 | 2.040 | -0.028 | 0.426 | 2.180 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 61 | 61 | 1.659 | 0.348 | 0.070 | -0.070 | 0.016 | 0.209 | 0.000 | 0.000 |

## puechabon_100km_jjas_era5_halo_all

Diagnostic only: Puéchabon-near ERA5-Land false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 91 | 91 | 9.046 | 4.680 | 4.602 | -0.024 | 0.484 | 4.742 | 0.000 | 0.000 |
| raw_era5land_center | 91 | 91 | 10.258 | 7.507 | 7.507 | 0.121 | 1.000 | 7.646 | 0.000 | 0.000 |
| downscalrain_cnn | 91 | 91 | 3.383 | 1.671 | 1.569 | -0.002 | 0.363 | 1.708 | 0.000 | 0.000 |
| downscalrain_cnn_firecalibrated | 91 | 91 | 3.381 | 1.620 | 1.429 | -0.022 | 0.341 | 1.568 | 0.000 | 0.000 |
| downscalrain_cnn_fireguard | 91 | 91 | 1.367 | 0.280 | 0.001 | -0.060 | 0.011 | 0.140 | 0.000 | 0.000 |
