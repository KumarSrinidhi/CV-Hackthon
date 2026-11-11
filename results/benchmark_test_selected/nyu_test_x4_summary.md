### nyu / test / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0342 | 0.0120 | 49.84 | 0.9935 | 0.1975 | 0.778 | 1.9 |
| bilinear | 0.0263 | 0.0091 | 52.16 | 0.9964 | 0.1557 | 0.066 | 2.1 |
| bicubic | 0.0232 | 0.0083 | 53.27 | 0.9970 | 0.1336 | 0.107 | 2.0 |
| classical | 0.0355 | 0.0142 | 49.49 | 0.9931 | 0.1894 | 0.731 | 289.1 |
| deep | 0.0068 | 0.0026 | 63.77 | 0.9996 | 0.0340 | 0.959 | 1882.1 |
| hybrid_fixed02 | 0.0068 | 0.0027 | 63.74 | 0.9996 | 0.0341 | 0.956 | 2096.4 |
| hybrid_adaptive | 0.0200 | 0.0060 | 54.62 | 0.9980 | 0.1206 | 0.822 | 2094.0 |
| hybrid_selected_clean | 0.0068 | 0.0027 | 63.76 | 0.9996 | 0.0341 | 0.956 | 2101.6 |
