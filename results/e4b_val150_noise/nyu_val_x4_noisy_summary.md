### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.02, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0345 | 0.0193 | 49.72 | 0.9940 | 0.1532 | 0.212 | 2.6 |
| deep | 0.0227 | 0.0153 | 53.10 | 0.9964 | 0.0669 | 0.855 | 1865.1 |
