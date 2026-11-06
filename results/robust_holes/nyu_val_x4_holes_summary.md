### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0475 | 0.0146 | 47.33 | 0.9926 | 0.2219 | 0.321 | 2.6 |
| deep | 0.0474 | 0.0092 | 47.55 | 0.9941 | 0.2175 | 0.887 | 1838.4 |
