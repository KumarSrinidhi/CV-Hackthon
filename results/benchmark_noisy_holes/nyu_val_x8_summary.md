### nyu / val / x8  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.02, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.1034 | 0.0423 | 40.45 | 0.9696 | 0.4330 | 0.429 | 1.6 |
| bilinear | 0.0816 | 0.0334 | 42.61 | 0.9845 | 0.3539 | 0.039 | 1.9 |
| bicubic | 0.0806 | 0.0351 | 42.69 | 0.9836 | 0.3402 | 0.062 | 2.0 |
| classical | 0.2603 | 0.0565 | 31.92 | 0.9387 | 0.6119 | 0.392 | 124.6 |
| deep | 0.0878 | 0.0341 | 42.10 | 0.9849 | 0.3309 | 0.712 | 1351.7 |
| hybrid_fixed02 | 0.0865 | 0.0337 | 42.23 | 0.9851 | 0.3259 | 0.711 | 1417.0 |
| hybrid_adaptive | 0.1166 | 0.0367 | 39.07 | 0.9690 | 0.3675 | 0.439 | 1436.4 |
