### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0372 | 0.0118 | 49.41 | 0.9949 | 0.1778 | 0.295 | 3.1 |
| deep | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 | 2084.8 |
