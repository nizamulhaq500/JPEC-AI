# RD benchmark -- kodak

24 images. Anchor: **jpegai-vr**. Negative BD-rate = fewer bits for equal quality = better.

AVG is the unweighted mean of the per-metric BD-rates, matching the paper's Tables III-VI (verified in docs/05).

BD-rate is interpolated with a monotone PCHIP over the *shared* quality range, not a global cubic -- see `jpegai/eval/bdrate.py` for why that choice changes answers by tens of percent on the saturating metrics.

## BD-rate vs jpegai-vr

| codec | AVG | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|

`overlap` = how many of jpegai-vr's rate points lie inside that codec's shared quality range. BD-rate averages over the overlap only, so a low count means the number rests on few anchor points.

## Rate points (dataset averages)

| codec | quality | bpp | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim |
|---|---|---|---|---|---|---|---|---|---|
| jpegai-vr | -1069 | 0.1893 | 0.9567 | 0.3338 | 0.9477 | 66.2999 | 0.2446 | 24.6984 | 0.9382 |
| jpegai-vr | -800 | 0.3066 | 0.9756 | 0.4058 | 0.9707 | 77.4051 | 0.1943 | 27.2831 | 0.9651 |
| jpegai-vr | -600 | 0.4242 | 0.9840 | 0.4569 | 0.9815 | 83.1113 | 0.1641 | 29.2733 | 0.9770 |
| jpegai-vr | -400 | 0.5701 | 0.9894 | 0.5047 | 0.9883 | 86.9423 | 0.1394 | 31.2039 | 0.9847 |
| jpegai-vr | -200 | 0.7440 | 0.9928 | 0.5482 | 0.9926 | 89.4899 | 0.1197 | 32.9696 | 0.9895 |
| jpegai-vr | 0 | 0.9412 | 0.9949 | 0.5840 | 0.9952 | 90.9712 | 0.1052 | 34.4726 | 0.9923 |
| jpegai-vr | 200 | 1.1566 | 0.9961 | 0.6110 | 0.9967 | 91.8808 | 0.0951 | 35.6132 | 0.9940 |
| jpegai-vr | 450 | 1.4491 | 0.9970 | 0.6326 | 0.9977 | 92.5032 | 0.0874 | 36.5867 | 0.9951 |
| jpegai-vr | 702 | 1.7790 | 0.9973 | 0.6444 | 0.9982 | 92.8746 | 0.0833 | 37.1612 | 0.9956 |

`quality` column: **jpegai-vr** -- one checkpoint swept over 9 Delta_beta values, -1069..+702

## PSNR (dB, reported only -- never part of AVG)

| codec | quality | bpp | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|---|---|
| jpegai-vr | -1069 | 0.1893 | 27.48 | 27.96 | 40.58 | 39.81 |
| jpegai-vr | -800 | 0.3066 | 29.17 | 29.68 | 42.05 | 41.39 |
| jpegai-vr | -600 | 0.4242 | 30.50 | 31.06 | 43.05 | 42.52 |
| jpegai-vr | -400 | 0.5701 | 31.75 | 32.39 | 43.89 | 43.46 |
| jpegai-vr | -200 | 0.7440 | 32.82 | 33.53 | 44.60 | 44.24 |
| jpegai-vr | 0 | 0.9412 | 33.66 | 34.43 | 45.19 | 44.87 |
| jpegai-vr | 200 | 1.1566 | 34.25 | 35.05 | 45.66 | 45.30 |
| jpegai-vr | 450 | 1.4491 | 34.69 | 35.54 | 45.97 | 45.62 |
| jpegai-vr | 702 | 1.7790 | 34.92 | 35.79 | 46.13 | 45.82 |
