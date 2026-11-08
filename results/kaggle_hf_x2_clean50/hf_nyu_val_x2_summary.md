### hf_nyu_val / test / x2  (n=50, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0392 | 0.0105 | 50.88 | 0.9951 | 0.1585 | 0.906 | 1.8 |
| bilinear | 0.0320 | 0.0090 | 52.71 | 0.9968 | 0.1286 | 0.562 | 1.4 |
| bicubic | 0.0276 | 0.0080 | 54.03 | 0.9975 | 0.1102 | 0.617 | 1.6 |
| classical | 0.0441 | 0.0130 | 49.66 | 0.9941 | 0.1805 | 0.795 | 1429.8 |
| hybrid | not evaluated | | | | | | UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself. |
| deep | not evaluated | | | | | | all samples failed, e.g. ['00001: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself.'] |

saved: results/kaggle_hf_x2_clean50/hf_nyu_val_x2_per_image.csv, results/kaggle_hf_x2_clean50/hf_nyu_val_x2_summary.json, results/kaggle_hf_x2_clean50/hf_nyu_val_x2_summary.md
hf_nyu_val: 645 samples available, evaluating up to 50, methods: ['nearest', 'bilinear', 'bicubic', 'classical', 'deep', 'hybrid']
  evaluated 10/50
  evaluated 20/50
  evaluated 30/50
  evaluated 40/50
  evaluated 50/50
