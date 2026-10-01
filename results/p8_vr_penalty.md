# RD benchmark -- kodak

24 images. Anchor: **jpegai**. Negative BD-rate = fewer bits for equal quality = better.

AVG is the unweighted mean of the per-metric BD-rates, matching the paper's Tables III-VI (verified in docs/05).

BD-rate is interpolated with a monotone PCHIP over the *shared* quality range, not a global cubic -- see `jpegai/eval/bdrate.py` for why that choice changes answers by tens of percent on the saturating metrics.

## BD-rate vs jpegai

| codec | AVG | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|
| jpegai-vr | **-5.7%** | +0.9% | -5.3% | -3.1% | -13.8% | -8.4% | -9.1% | -0.9% | 6/9 |

`overlap` = how many of jpegai's rate points lie inside that codec's shared quality range. BD-rate averages over the overlap only, so a low count means the number rests on few anchor points.

**Caveat:** `jpegai-vr` (6/9) span only part of jpegai's range. Their AVG is not measured over the same ground as the other rows. The fix is lower-rate points in the ladder.

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
| jpegai-vr | -1069 | 0.1893 | 0.9567 | 0.3338 | 0.9477 | 66.2999 | 0.2446 | 24.6984 | 0.9382 |
| jpegai-vr | -800 | 0.3066 | 0.9756 | 0.4058 | 0.9707 | 77.4051 | 0.1943 | 27.2831 | 0.9651 |
| jpegai-vr | -600 | 0.4242 | 0.9840 | 0.4569 | 0.9815 | 83.1113 | 0.1641 | 29.2733 | 0.9770 |
| jpegai-vr | -400 | 0.5701 | 0.9894 | 0.5047 | 0.9883 | 86.9423 | 0.1394 | 31.2039 | 0.9847 |
| jpegai-vr | -200 | 0.7440 | 0.9928 | 0.5482 | 0.9926 | 89.4899 | 0.1197 | 32.9696 | 0.9895 |
| jpegai-vr | 0 | 0.9412 | 0.9949 | 0.5840 | 0.9952 | 90.9712 | 0.1052 | 34.4726 | 0.9923 |
| jpegai-vr | 200 | 1.1566 | 0.9961 | 0.6110 | 0.9967 | 91.8808 | 0.0951 | 35.6132 | 0.9940 |
| jpegai-vr | 450 | 1.4491 | 0.9970 | 0.6326 | 0.9977 | 92.5032 | 0.0874 | 36.5867 | 0.9951 |
| jpegai-vr | 702 | 1.7790 | 0.9973 | 0.6444 | 0.9982 | 92.8746 | 0.0833 | 37.1612 | 0.9956 |

`quality` column: **jpegai** -- 9 trained rate points; **jpegai-vr** -- one checkpoint swept over 9 Delta_beta values, -1069..+702

### PSNR BD-rate vs jpegai (diagnostic, not in AVG)

Separates the two branches: `psnr_y` is the luma branch, `psnr_u`/`psnr_v` the chroma one. None of these saturates, so they are the most robust rows in this file.

| codec | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|
| jpegai-vr | -18.3% | -16.8% | -19.1% | -22.3% |

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
| jpegai-vr | -1069 | 0.1893 | 27.48 | 27.96 | 40.58 | 39.81 |
| jpegai-vr | -800 | 0.3066 | 29.17 | 29.68 | 42.05 | 41.39 |
| jpegai-vr | -600 | 0.4242 | 30.50 | 31.06 | 43.05 | 42.52 |
| jpegai-vr | -400 | 0.5701 | 31.75 | 32.39 | 43.89 | 43.46 |
| jpegai-vr | -200 | 0.7440 | 32.82 | 33.53 | 44.60 | 44.24 |
| jpegai-vr | 0 | 0.9412 | 33.66 | 34.43 | 45.19 | 44.87 |
| jpegai-vr | 200 | 1.1566 | 34.25 | 35.05 | 45.66 | 45.30 |
| jpegai-vr | 450 | 1.4491 | 34.69 | 35.54 | 45.97 | 45.62 |
| jpegai-vr | 702 | 1.7790 | 34.92 | 35.79 | 46.13 | 45.82 |
