### nyu / val / x2  (n=30, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0297 | 0.0079 | 51.34 | 0.9970 | 0.1469 | 0.918 | 2.0 |
| bilinear | 0.0240 | 0.0064 | 53.23 | 0.9981 | 0.1188 | 0.610 | 2.4 |
| bicubic | 0.0205 | 0.0057 | 54.59 | 0.9986 | 0.1009 | 0.672 | 2.5 |
| classical | 0.0356 | 0.0103 | 50.01 | 0.9956 | 0.1726 | 0.870 | 706.5 |
| hybrid_fixed02 | not evaluated | | | | | | NotImplementedError: hybrid unavailable: deep branch not available: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself. |
| hybrid_adaptive | not evaluated | | | | | | NotImplementedError: hybrid unavailable: deep branch not available: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself. |
| deep | not evaluated | | | | | | all samples failed, e.g. ['nyu_val_0000: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself.'] |
