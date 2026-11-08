### hf_nyu_val / test / x4  (n=100, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0535 | 0.0162 | 48.27 | 0.9890 | 0.2465 | 0.739 | 2.0 |
| bilinear | 0.0415 | 0.0132 | 50.53 | 0.9934 | 0.1943 | 0.205 | 1.4 |
| bicubic | 0.0367 | 0.0121 | 51.62 | 0.9942 | 0.1680 | 0.259 | 1.8 |
| classical | 0.0571 | 0.0183 | 47.86 | 0.9890 | 0.2566 | 0.665 | 589.5 |
| hybrid | not evaluated | | | | | | DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4. |
| deep | not evaluated | | | | | | all samples failed, e.g. ["00001: DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4."] |

saved: results/kaggle_hf_x4_clean100/hf_nyu_val_x4_per_image.csv, results/kaggle_hf_x4_clean100/hf_nyu_val_x4_summary.json, results/kaggle_hf_x4_clean100/hf_nyu_val_x4_summary.md

hf_nyu_val: 645 samples available, evaluating up to 50, methods: ['nearest', 'bilinear', 'bicubic', 'classical', 'deep', 'hybrid']
  evaluated 10/50
  evaluated 20/50
  evaluated 30/50
  evaluated 40/50
  evaluated 50/50
