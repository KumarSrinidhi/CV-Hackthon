### nyu / val / x4  (n=3, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0335 | 0.0100 | 49.90 | 0.9957 | 0.1888 | 0.291 | 5.7 |
| classical | 0.0534 | 0.0167 | 45.77 | 0.9900 | 0.2937 | 0.726 | 307.5 |
| deep | 0.0099 | 0.0030 | 60.26 | 0.9996 | 0.0533 | 0.983 | 2475.3 |
| hybrid | 0.0270 | 0.0066 | 51.69 | 0.9974 | 0.1585 | 0.897 | 2457.7 |
