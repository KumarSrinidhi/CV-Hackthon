### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0475 | 0.0146 | 47.33 | 0.9926 | 0.2219 | 0.321 | 4.1 |
| deep | 0.0560 | 0.0102 | 46.09 | 0.9924 | 0.2568 | 0.865 | 2090.1 |
