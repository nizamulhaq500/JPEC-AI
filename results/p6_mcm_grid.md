# RD benchmark -- kodak

24 images. Anchor: **jpeg**. Negative BD-rate = fewer bits for equal quality = better.

AVG is the unweighted mean of the per-metric BD-rates, matching the paper's Tables III-VI (verified in docs/05).

BD-rate is interpolated with a monotone PCHIP over the *shared* quality range, not a global cubic -- see `jpegai/eval/bdrate.py` for why that choice changes answers by tens of percent on the saturating metrics.

## BD-rate vs jpeg

| codec | AVG | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim | overlap |
|---|---|---|---|---|---|---|---|---|---|
| vvc | **-45.9%** | -49.1% | -53.9% | -44.4% | -41.1% | -50.6% | -34.2% | -47.9% | 9/11 |
| ours-ladder_p5 | **+1.5%** | -26.3% | -4.4% | -16.8% | +27.8% | +6.1% | +30.1% | -5.9% | 6/11 |
| ours-ladder_p5_cont200 | **+nan%** | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | 0/0 |
| ours-ladder_p6a_mcm1 | **+nan%** | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | 0/0 |
| ours-ladder_p6a_mcm1_200 | **+nan%** | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | 0/0 |

`overlap` = how many of jpeg's rate points lie inside that codec's shared quality range. BD-rate averages over the overlap only, so a low count means the number rests on few anchor points.

**Caveat:** `ours-ladder_p5` (6/11) span only part of jpeg's range. Their AVG is not measured over the same ground as the other rows. The fix is lower-rate points in the ladder.

## Rate points (dataset averages)

| codec | quality | bpp | ms_ssim | vif | fsim | vmaf | nlpd | psnr_hvs | iw_ssim |
|---|---|---|---|---|---|---|---|---|---|
| jpeg | 10 | 0.2555 | 0.9288 | 0.2931 | 0.9253 | 67.2849 | 0.2682 | 25.0887 | 0.9125 |
| jpeg | 18 | 0.4163 | 0.9637 | 0.3827 | 0.9650 | 80.1484 | 0.2007 | 28.3371 | 0.9572 |
| jpeg | 25 | 0.5334 | 0.9746 | 0.4284 | 0.9770 | 84.8236 | 0.1724 | 30.1492 | 0.9709 |
| jpeg | 32 | 0.6383 | 0.9807 | 0.4615 | 0.9833 | 87.5803 | 0.1536 | 31.5489 | 0.9784 |
| jpeg | 40 | 0.7433 | 0.9847 | 0.4894 | 0.9873 | 89.4556 | 0.1393 | 32.7826 | 0.9833 |
| jpeg | 50 | 0.8692 | 0.9880 | 0.5184 | 0.9906 | 91.0563 | 0.1254 | 34.1263 | 0.9872 |
| jpeg | 62 | 1.0388 | 0.9909 | 0.5521 | 0.9933 | 92.4858 | 0.1109 | 35.7094 | 0.9905 |
| jpeg | 75 | 1.3443 | 0.9939 | 0.6040 | 0.9960 | 94.0006 | 0.0918 | 38.1183 | 0.9939 |
| jpeg | 85 | 1.8356 | 0.9963 | 0.6739 | 0.9978 | 95.2425 | 0.0717 | 41.1679 | 0.9965 |
| jpeg | 92 | 2.5589 | 0.9980 | 0.7650 | 0.9989 | 96.0849 | 0.0522 | 44.6181 | 0.9982 |
| jpeg | 96 | 3.6961 | 0.9990 | 0.8645 | 0.9994 | 96.5551 | 0.0360 | 47.9179 | 0.9991 |
| vvc | 47 | 0.0521 | 0.8854 | 0.2353 | 0.8705 | 48.2074 | 0.3201 | 22.6565 | 0.8397 |
| vvc | 44 | 0.0856 | 0.9188 | 0.2916 | 0.9101 | 60.3699 | 0.2718 | 24.4509 | 0.8915 |
| vvc | 41 | 0.1370 | 0.9459 | 0.3532 | 0.9412 | 70.6368 | 0.2272 | 26.3625 | 0.9313 |
| vvc | 38 | 0.2223 | 0.9658 | 0.4217 | 0.9636 | 79.2307 | 0.1855 | 28.6022 | 0.9588 |
| vvc | 35 | 0.3501 | 0.9792 | 0.4945 | 0.9784 | 85.6786 | 0.1482 | 31.0243 | 0.9763 |
| vvc | 32 | 0.5255 | 0.9876 | 0.5681 | 0.9875 | 89.8940 | 0.1173 | 33.4605 | 0.9866 |
| vvc | 29 | 0.7543 | 0.9925 | 0.6413 | 0.9928 | 92.5214 | 0.0924 | 35.8173 | 0.9923 |
| vvc | 26 | 1.0451 | 0.9953 | 0.7123 | 0.9958 | 94.1450 | 0.0730 | 38.0005 | 0.9955 |
| vvc | 22 | 1.6091 | 0.9977 | 0.8081 | 0.9979 | 95.3967 | 0.0521 | 40.6924 | 0.9979 |
| vvc | 18 | 2.3286 | 0.9989 | 0.8859 | 0.9989 | 96.0446 | 0.0375 | 42.6930 | 0.9989 |
| ours-ladder_p5 | 0.002 | 0.4835 | 0.9790 | 0.4248 | 0.9773 | 78.5183 | 0.1888 | 27.3522 | 0.9708 |
| ours-ladder_p5 | 0.012 | 0.8829 | 0.9932 | 0.5357 | 0.9939 | 88.4400 | 0.1278 | 32.2715 | 0.9883 |
| ours-ladder_p5 | 0.03 | 1.2691 | 0.9961 | 0.6045 | 0.9970 | 91.4325 | 0.0993 | 35.3771 | 0.9936 |
| ours-ladder_p5 | 0.075 | 1.7470 | 0.9974 | 0.6580 | 0.9983 | 93.5429 | 0.0823 | 37.7363 | 0.9959 |
| ours-ladder_p5 | 0.2 | 2.2802 | 0.9982 | 0.6996 | 0.9989 | 93.8225 | 0.0708 | 39.4149 | 0.9971 |
| ours-ladder_p5_cont200 | 0.012 | 0.9620 | 0.9950 | 0.5856 | 0.9953 | 90.7444 | 0.1044 | 34.5769 | 0.9924 |
| ours-ladder_p6a_mcm1 | 0.012 | 0.9177 | 0.9941 | 0.5617 | 0.9946 | 88.7697 | 0.1155 | 33.4507 | 0.9907 |
| ours-ladder_p6a_mcm1 | 0.03 | 1.3345 | 0.9965 | 0.6236 | 0.9973 | 91.5843 | 0.0925 | 36.1940 | 0.9945 |
| ours-ladder_p6a_mcm1_200 | 0.012 | 0.9636 | 0.9950 | 0.5861 | 0.9953 | 90.6573 | 0.1044 | 34.5916 | 0.9925 |

### PSNR BD-rate vs jpeg (diagnostic, not in AVG)

Separates the two branches: `psnr_y` is the luma branch, `psnr_u`/`psnr_v` the chroma one. None of these saturates, so they are the most robust rows in this file.

| codec | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|
| vvc | -59.7% | -56.2% | -64.5% | -58.4% |
| ours-ladder_p5 | +13.8% | +27.9% | -54.7% | -47.1% |
| ours-ladder_p5_cont200 | -- | -- | -- | -- |
| ours-ladder_p6a_mcm1 | -- | -- | -- | -- |
| ours-ladder_p6a_mcm1_200 | -- | -- | -- | -- |

## PSNR (dB, reported only -- never part of AVG)

| codec | quality | bpp | psnr | psnr_y | psnr_u | psnr_v |
|---|---|---|---|---|---|---|
| jpeg | 10 | 0.2555 | 26.67 | 28.02 | 35.30 | 35.10 |
| jpeg | 18 | 0.4163 | 28.80 | 29.98 | 38.09 | 37.95 |
| jpeg | 25 | 0.5334 | 29.89 | 31.04 | 39.31 | 39.16 |
| jpeg | 32 | 0.6383 | 30.70 | 31.85 | 40.11 | 39.92 |
| jpeg | 40 | 0.7433 | 31.42 | 32.58 | 40.82 | 40.66 |
| jpeg | 50 | 0.8692 | 32.17 | 33.38 | 41.45 | 41.30 |
| jpeg | 62 | 1.0388 | 33.08 | 34.37 | 42.05 | 41.95 |
| jpeg | 75 | 1.3443 | 34.52 | 35.98 | 42.98 | 42.96 |
| jpeg | 85 | 1.8356 | 36.46 | 38.27 | 44.05 | 44.16 |
| jpeg | 92 | 2.5589 | 38.86 | 41.43 | 45.15 | 45.37 |
| jpeg | 96 | 3.6961 | 41.35 | 45.26 | 46.33 | 46.61 |
| vvc | 47 | 0.0521 | 26.40 | 27.01 | 37.85 | 37.06 |
| vvc | 44 | 0.0856 | 27.74 | 28.40 | 38.92 | 38.10 |
| vvc | 41 | 0.1370 | 29.03 | 29.82 | 39.54 | 38.76 |
| vvc | 38 | 0.2223 | 30.68 | 31.57 | 40.67 | 39.91 |
| vvc | 35 | 0.3501 | 32.44 | 33.53 | 41.63 | 40.93 |
| vvc | 32 | 0.5255 | 34.23 | 35.58 | 42.49 | 41.77 |
| vvc | 29 | 0.7543 | 35.97 | 37.67 | 43.26 | 42.64 |
| vvc | 26 | 1.0451 | 37.57 | 39.71 | 43.92 | 43.30 |
| vvc | 22 | 1.6091 | 39.49 | 42.49 | 44.60 | 44.01 |
| vvc | 18 | 2.3286 | 41.04 | 44.98 | 45.21 | 44.59 |
| ours-ladder_p5 | 0.002 | 0.4835 | 28.67 | 29.23 | 41.72 | 40.97 |
| ours-ladder_p5 | 0.012 | 0.8829 | 31.59 | 32.20 | 44.30 | 43.85 |
| ours-ladder_p5 | 0.03 | 1.2691 | 33.79 | 34.51 | 45.74 | 45.37 |
| ours-ladder_p5 | 0.075 | 1.7470 | 35.38 | 36.20 | 46.76 | 46.48 |
| ours-ladder_p5 | 0.2 | 2.2802 | 36.47 | 37.39 | 47.36 | 47.34 |
| ours-ladder_p5_cont200 | 0.012 | 0.9620 | 33.74 | 34.52 | 45.19 | 44.77 |
| ours-ladder_p6a_mcm1 | 0.012 | 0.9177 | 32.69 | 33.38 | 44.80 | 44.27 |
| ours-ladder_p6a_mcm1 | 0.03 | 1.3345 | 34.60 | 35.39 | 46.09 | 45.77 |
| ours-ladder_p6a_mcm1_200 | 0.012 | 0.9636 | 33.73 | 34.54 | 45.18 | 44.61 |

## Skipped metrics

- `ours-ladder_p5_cont200`: ms_ssim: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: vif: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: fsim: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: vmaf: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: nlpd: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: psnr_hvs: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p5_cont200`: iw_ssim: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1`: ms_ssim: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: vif: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: fsim: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: vmaf: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: nlpd: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: psnr_hvs: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1`: iw_ssim: need >= 4 distinct-quality points, got 11 and 2
- `ours-ladder_p6a_mcm1_200`: ms_ssim: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: vif: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: fsim: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: vmaf: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: nlpd: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: psnr_hvs: need >= 4 distinct-quality points, got 11 and 1
- `ours-ladder_p6a_mcm1_200`: iw_ssim: need >= 4 distinct-quality points, got 11 and 1
