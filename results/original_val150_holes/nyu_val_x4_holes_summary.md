### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0374 | 0.0120 | 49.47 | 0.9943 | 0.1901 | 0.231 | 3.0 |
| deep | 0.0438 | 0.0084 | 48.25 | 0.9940 | 0.2189 | 0.817 | 1876.4 |
