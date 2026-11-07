### hf_nyu_val / test / x4  (n=645, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0391 | 0.0131 | 49.80 | 0.9938 | 0.1722 | 0.273 | 2.3 |
| deep | 0.0102 | 0.0033 | 61.31 | 0.9994 | 0.0411 | 0.968 | 7678.2 |
| deep_adaptive | 0.0102 | 0.0034 | 61.30 | 0.9994 | 0.0412 | 0.968 | 1845.6 |
