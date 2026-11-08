### hf_nyu_val / test / x4  (n=100, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0535 | 0.0162 | 48.27 | 0.9890 | 0.2465 | 0.739 | 2.7 |
| bilinear | 0.0415 | 0.0132 | 50.53 | 0.9934 | 0.1943 | 0.205 | 2.7 |
| bicubic | 0.0367 | 0.0121 | 51.62 | 0.9942 | 0.1680 | 0.259 | 2.7 |
| classical | 0.0584 | 0.0191 | 47.59 | 0.9874 | 0.2591 | 0.671 | 458.9 |
| deep | 0.0101 | 0.0030 | 62.64 | 0.9994 | 0.0434 | 0.960 | 1900.1 |
| hybrid | 0.0298 | 0.0077 | 53.85 | 0.9966 | 0.1389 | 0.804 | 2455.0 |
