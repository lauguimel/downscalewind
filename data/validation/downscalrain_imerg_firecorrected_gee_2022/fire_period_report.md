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
| imerg_firebalanced | 1085 | 925 | 4.854 | 1.626 | -0.217 | 0.250 | 0.124 | 0.656 | 0.431 | 0.158 |
| imerg_firecorrected | 1085 | 925 | 4.702 | 1.280 | -0.968 | 0.215 | 0.005 | 0.123 | 0.075 | 0.105 |

## test_jjas_mediterranean_dry

Station dry days inside the held-out Mediterranean fire window.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 925 | 925 | 3.227 | 0.900 | 0.850 | 0.227 | 0.137 | 0.904 | 0.000 | 0.000 |
| raw_era5land_center | 925 | 925 | 2.689 | 0.822 | 0.783 | 0.216 | 0.165 | 0.837 | 0.000 | 0.000 |
| downscalrain_cnn | 925 | 925 | 1.209 | 0.297 | 0.246 | 0.242 | 0.058 | 0.300 | 0.000 | 0.000 |
| imerg_firebalanced | 925 | 925 | 2.305 | 0.652 | 0.602 | 0.228 | 0.124 | 0.656 | 0.000 | 0.000 |
| imerg_firecorrected | 925 | 925 | 1.095 | 0.160 | 0.069 | 0.066 | 0.005 | 0.123 | 0.000 | 0.000 |

## test_jjas_mediterranean_imerg_halo

Station dry, IMERG wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 127 | 127 | 8.695 | 6.094 | 6.094 | 0.140 | 1.000 | 6.254 | 0.000 | 0.000 |
| raw_era5land_center | 127 | 127 | 6.401 | 3.427 | 3.386 | 0.108 | 0.583 | 3.546 | 0.000 | 0.000 |
| downscalrain_cnn | 127 | 127 | 3.184 | 1.550 | 1.471 | 0.190 | 0.331 | 1.630 | 0.000 | 0.000 |
| imerg_firebalanced | 127 | 127 | 6.199 | 4.290 | 4.290 | 0.140 | 0.906 | 4.450 | 0.000 | 0.000 |
| imerg_firecorrected | 127 | 127 | 2.911 | 0.706 | 0.405 | 0.020 | 0.039 | 0.565 | 0.000 | 0.000 |

## test_jjas_mediterranean_era5_halo

Station dry, ERA5-Land wet: direct false-rain halo subset.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 153 | 153 | 7.267 | 3.757 | 3.682 | 0.148 | 0.484 | 3.831 | 0.000 | 0.000 |
| raw_era5land_center | 153 | 153 | 6.592 | 4.483 | 4.483 | 0.108 | 1.000 | 4.631 | 0.000 | 0.000 |
| downscalrain_cnn | 153 | 153 | 2.827 | 1.264 | 1.185 | 0.151 | 0.255 | 1.334 | 0.000 | 0.000 |
| imerg_firebalanced | 153 | 153 | 5.210 | 2.690 | 2.616 | 0.147 | 0.477 | 2.764 | 0.000 | 0.000 |
| imerg_firecorrected | 153 | 153 | 2.660 | 0.651 | 0.396 | 0.022 | 0.033 | 0.545 | 0.000 | 0.000 |

## test_puechabon_150km_jjas

Held-out stations within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 244 | 209 | 4.625 | 1.572 | 0.503 | 0.480 | 0.120 | 0.794 | 0.686 | 0.625 |
| raw_era5land_center | 244 | 209 | 4.604 | 1.772 | 0.473 | 0.323 | 0.196 | 0.875 | 0.800 | 0.125 |
| downscalrain_cnn | 244 | 209 | 3.880 | 1.072 | -0.656 | 0.506 | 0.053 | 0.218 | 0.429 | 0.250 |
| imerg_firebalanced | 244 | 209 | 3.984 | 1.282 | 0.022 | 0.482 | 0.110 | 0.572 | 0.600 | 0.500 |
| imerg_firecorrected | 244 | 209 | 4.390 | 1.140 | -0.895 | 0.287 | 0.000 | 0.055 | 0.057 | 0.250 |

## test_puechabon_150km_jjas_dry

Held-out dry station-days within 150 km of Puéchabon, Jun-Sep.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 209 | 209 | 2.906 | 0.767 | 0.734 | 0.544 | 0.120 | 0.794 | 0.000 | 0.000 |
| raw_era5land_center | 209 | 209 | 2.179 | 0.833 | 0.816 | 0.343 | 0.196 | 0.875 | 0.000 | 0.000 |
| downscalrain_cnn | 209 | 209 | 0.404 | 0.208 | 0.158 | 0.277 | 0.053 | 0.218 | 0.000 | 0.000 |
| imerg_firebalanced | 209 | 209 | 2.008 | 0.545 | 0.513 | 0.545 | 0.110 | 0.572 | 0.000 | 0.000 |
| imerg_firecorrected | 209 | 209 | 0.242 | 0.099 | -0.004 | 0.045 | 0.000 | 0.055 | 0.000 | 0.000 |

## puechabon_100km_jjas_all

Diagnostic only: nearest stations are train/val, not held-out.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 603 | 517 | 6.891 | 2.150 | -0.108 | 0.492 | 0.118 | 0.931 | 0.570 | 0.519 |
| raw_era5land_center | 603 | 517 | 8.294 | 2.789 | 0.822 | 0.351 | 0.176 | 1.401 | 0.721 | 0.556 |
| downscalrain_cnn | 603 | 517 | 7.184 | 1.733 | -0.883 | 0.413 | 0.075 | 0.367 | 0.605 | 0.259 |
| imerg_firebalanced | 603 | 517 | 6.860 | 1.975 | -0.576 | 0.482 | 0.093 | 0.666 | 0.547 | 0.407 |
| imerg_firecorrected | 603 | 517 | 8.116 | 1.841 | -1.545 | 0.067 | 0.002 | 0.110 | 0.058 | 0.074 |

## puechabon_100km_jjas_imerg_halo_all

Diagnostic only: Puéchabon-near IMERG false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 61 | 61 | 11.137 | 7.353 | 7.353 | -0.032 | 1.000 | 7.493 | 0.000 | 0.000 |
| raw_era5land_center | 61 | 61 | 10.449 | 6.971 | 6.913 | 0.041 | 0.721 | 7.052 | 0.000 | 0.000 |
| downscalrain_cnn | 61 | 61 | 4.084 | 2.256 | 2.133 | -0.022 | 0.426 | 2.272 | 0.000 | 0.000 |
| imerg_firebalanced | 61 | 61 | 7.772 | 5.114 | 5.105 | -0.032 | 0.787 | 5.245 | 0.000 | 0.000 |
| imerg_firecorrected | 61 | 61 | 4.163 | 0.671 | 0.392 | -0.070 | 0.016 | 0.532 | 0.000 | 0.000 |

## puechabon_100km_jjas_era5_halo_all

Diagnostic only: Puéchabon-near ERA5-Land false-rain halo days.

| model | n | n dry | RMSE | MAE | bias | corr | false wet/dry | mean rain on dry | wet recall | heavy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_imerg_center | 91 | 91 | 9.046 | 4.680 | 4.602 | -0.024 | 0.484 | 4.742 | 0.000 | 0.000 |
| raw_era5land_center | 91 | 91 | 10.258 | 7.507 | 7.507 | 0.121 | 1.000 | 7.646 | 0.000 | 0.000 |
| downscalrain_cnn | 91 | 91 | 3.383 | 1.671 | 1.569 | -0.002 | 0.363 | 1.708 | 0.000 | 0.000 |
| imerg_firebalanced | 91 | 91 | 6.316 | 3.286 | 3.209 | -0.021 | 0.473 | 3.349 | 0.000 | 0.000 |
| imerg_firecorrected | 91 | 91 | 3.412 | 0.520 | 0.315 | -0.046 | 0.011 | 0.455 | 0.000 | 0.000 |
