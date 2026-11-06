### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.01, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0306 | 0.0136 | 51.00 | 0.9956 | 0.1524 | 0.211 | 2.0 |
| deep | 0.0165 | 0.0108 | 55.83 | 0.9981 | 0.0545 | 0.884 | 1792.9 |
