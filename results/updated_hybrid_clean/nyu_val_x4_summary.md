### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0372 | 0.0118 | 49.41 | 0.9949 | 0.1778 | 0.295 | 2.4 |
| deep | 0.0147 | 0.0046 | 58.09 | 0.9991 | 0.0647 | 0.969 | 2081.0 |
| hybrid | 0.0141 | 0.0045 | 58.32 | 0.9992 | 0.0626 | 0.969 | 2455.6 |
