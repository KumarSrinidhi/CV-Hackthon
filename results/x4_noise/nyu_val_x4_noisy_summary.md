### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.01, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0383 | 0.0155 | 49.06 | 0.9944 | 0.1781 | 0.295 | 2.4 |
| deep | 0.0208 | 0.0127 | 54.06 | 0.9973 | 0.0670 | 0.931 | 2081.7 |
