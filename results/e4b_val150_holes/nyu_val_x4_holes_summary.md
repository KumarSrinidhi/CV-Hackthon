### nyu / val / x4  (n=150, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0374 | 0.0120 | 49.47 | 0.9943 | 0.1901 | 0.231 | 5.0 |
| deep | 0.0344 | 0.0071 | 50.33 | 0.9961 | 0.1736 | 0.855 | 1875.0 |
