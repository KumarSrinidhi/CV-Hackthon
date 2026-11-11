### nyu / val / x8  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0875 | 0.0314 | 41.92 | 0.9748 | 0.3941 | 0.462 | 1.8 |
| bilinear | 0.0701 | 0.0253 | 43.96 | 0.9867 | 0.3231 | 0.038 | 2.2 |
| bicubic | 0.0634 | 0.0238 | 44.81 | 0.9872 | 0.2928 | 0.057 | 2.2 |
| classical | 0.0931 | 0.0355 | 41.62 | 0.9756 | 0.3873 | 0.569 | 158.9 |
| deep | 0.0259 | 0.0089 | 53.07 | 0.9974 | 0.1117 | 0.930 | 1341.1 |
| hybrid_fixed02 | 0.0259 | 0.0089 | 53.07 | 0.9974 | 0.1118 | 0.929 | 1454.3 |
| hybrid_adaptive | 0.0530 | 0.0167 | 46.54 | 0.9914 | 0.2401 | 0.749 | 1458.0 |
