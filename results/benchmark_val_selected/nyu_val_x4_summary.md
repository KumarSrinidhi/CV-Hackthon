### nyu / val / x4  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0545 | 0.0165 | 46.06 | 0.9898 | 0.2629 | 0.808 | 1.7 |
| bilinear | 0.0422 | 0.0130 | 48.32 | 0.9941 | 0.2071 | 0.214 | 1.9 |
| bicubic | 0.0372 | 0.0118 | 49.41 | 0.9949 | 0.1778 | 0.295 | 1.8 |
| classical | 0.0599 | 0.0199 | 45.49 | 0.9882 | 0.2764 | 0.754 | 255.0 |
| deep | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 | 1874.3 |
| hybrid_fixed02 | 0.0123 | 0.0037 | 59.55 | 0.9994 | 0.0538 | 0.975 | 2059.5 |
| hybrid_adaptive | 0.0338 | 0.0087 | 50.44 | 0.9966 | 0.1670 | 0.875 | 2060.8 |
| hybrid_selected_clean | 0.0123 | 0.0037 | 59.57 | 0.9994 | 0.0537 | 0.975 | 2047.2 |
