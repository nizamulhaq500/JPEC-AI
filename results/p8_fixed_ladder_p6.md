# RD benchmark -- kodak

24 images. Anchor: **jpegai**. Negative BD-rate = fewer bits for equal quality = better.

AVG is the unweighted mean of the per-metric BD-rates, matching the paper's Tables III-VI (verified in docs/05).

BD-rate is interpolated with a monotone PCHIP over the *shared* quality range, not a global cubic -- see `jpegai/eval/bdrate.py` for why that choice changes answers by tens of percent on the saturating metrics.

## BD-rate vs jpegai

| codec | AVG | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|

`overlap` = how many of jpegai's rate points lie inside that codec's shared quality range. BD-rate averages over the overlap only, so a low count means the number rests on few anchor points.

## Rate points (dataset averages)

| codec | quality | bpp | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim |
|---|---|---|---|---|---|---|---|---|---|
| jpegai | 0.0002 | 0.1153 | 0.9413 | 0.2907 | 0.9223 | 54.1508 | 0.2865 | 22.9420 | 0.9153 |
| jpegai | 0.0005 | 0.2076 | 0.9650 | 0.3578 | 0.9531 | 67.4283 | 0.2315 | 25.2491 | 0.9502 |
| jpegai | 0.001 | 0.3080 | 0.9759 | 0.4041 | 0.9691 | 74.5775 | 0.1991 | 26.8930 | 0.9655 |
| jpegai | 0.002 | 0.4275 | 0.9819 | 0.4386 | 0.9793 | 79.4045 | 0.1780 | 28.1517 | 0.9738 |
| jpegai | 0.005 | 0.5954 | 0.9898 | 0.4950 | 0.9893 | 84.9368 | 0.1469 | 30.5100 | 0.9832 |
| jpegai | 0.012 | 0.9173 | 0.9942 | 0.5630 | 0.9946 | 90.1201 | 0.1154 | 33.4976 | 0.9907 |
| jpegai | 0.03 | 1.3303 | 0.9965 | 0.6244 | 0.9973 | 92.2771 | 0.0923 | 36.2649 | 0.9946 |
| jpegai | 0.075 | 1.8283 | 0.9977 | 0.6770 | 0.9985 | 93.7795 | 0.0762 | 38.5683 | 0.9965 |
| jpegai | 0.2 | 2.4317 | 0.9985 | 0.7243 | 0.9991 | 94.2532 | 0.0641 | 40.6526 | 0.9976 |

`quality` column: **jpegai** -- 9 trained rate points

## PSNR (dB, reported only -- never part of AVG)

| codec | quality | bpp | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|---|---|
| jpegai | 0.0002 | 0.1153 | 26.12 | 26.83 | 37.85 | 36.92 |
| jpegai | 0.0005 | 0.2076 | 27.67 | 28.32 | 39.92 | 38.95 |
| jpegai | 0.001 | 0.3080 | 28.63 | 29.26 | 41.00 | 40.24 |
| jpegai | 0.002 | 0.4275 | 29.25 | 29.84 | 41.98 | 41.30 |
| jpegai | 0.005 | 0.5954 | 30.61 | 31.24 | 43.38 | 42.54 |
| jpegai | 0.012 | 0.9173 | 32.69 | 33.39 | 44.79 | 44.25 |
| jpegai | 0.03 | 1.3303 | 34.62 | 35.41 | 46.13 | 45.76 |
| jpegai | 0.075 | 1.8283 | 36.00 | 36.92 | 46.81 | 46.77 |
| jpegai | 0.2 | 2.4317 | 37.29 | 38.32 | 47.68 | 47.74 |
