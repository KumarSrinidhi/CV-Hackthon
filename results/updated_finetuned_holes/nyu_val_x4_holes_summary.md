### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 42})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0472 | 0.0145 | 47.37 | 0.9927 | 0.2215 | 0.319 | 3.2 |
| deep | 0.0217 | 0.0065 | 54.50 | 0.9983 | 0.0983 | 0.949 | 2086.2 |
