### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.05, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0529 | 0.0381 | 45.64 | 0.9831 | 0.1587 | 0.218 | 1.9 |
| deep | 0.0741 | 0.0548 | 42.61 | 0.9551 | 0.1153 | 0.594 | 1850.7 |
| deep_adaptive | 0.0293 | 0.0192 | 50.85 | 0.9958 | 0.0988 | 0.708 | 1835.7 |
