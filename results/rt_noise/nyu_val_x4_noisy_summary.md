### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.01, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0306 | 0.0136 | 51.00 | 0.9956 | 0.1524 | 0.211 | 1.5 |
| deep | 0.0141 | 0.0078 | 57.53 | 0.9988 | 0.0555 | 0.929 | 1789.8 |
