### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.02, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0670 | 0.0282 | 44.16 | 0.9839 | 0.3003 | 0.758 | 2.7 |
| bilinear | 0.0512 | 0.0211 | 46.56 | 0.9915 | 0.2361 | 0.230 | 3.0 |
| bicubic | 0.0510 | 0.0233 | 46.51 | 0.9906 | 0.2232 | 0.321 | 3.0 |
| classical | 0.1861 | 0.0327 | 34.87 | 0.9525 | 0.4516 | 0.520 | 246.1 |
| deep | 0.0615 | 0.0268 | 44.90 | 0.9863 | 0.2492 | 0.833 | 6342.9 |
| hybrid_fixed02 | 0.0605 | 0.0265 | 45.04 | 0.9866 | 0.2449 | 0.833 | 6711.4 |
| hybrid_adaptive | 0.0793 | 0.0259 | 42.35 | 0.9779 | 0.2702 | 0.577 | 6557.4 |
