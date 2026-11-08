### hf_nyu_val / test / x8  (n=50, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.1151 | 0.0424 | 41.41 | 0.9615 | 0.4272 | 0.447 | 1.8 |
| bilinear | 0.0913 | 0.0360 | 43.47 | 0.9785 | 0.3439 | 0.066 | 1.2 |
| bicubic | 0.0826 | 0.0336 | 44.36 | 0.9793 | 0.3116 | 0.102 | 1.7 |
| classical | 0.1247 | 0.0479 | 40.88 | 0.9641 | 0.4312 | 0.544 | 367.9 |
| hybrid | not evaluated | | | | | | DeepModelUnavailableError: No SGNet x8 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X8_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X8. |
| deep | not evaluated | | | | | | all samples failed, e.g. ["00001: DeepModelUnavailableError: No SGNet x8 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X8_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X8."] |

saved: results/kaggle_hf_x8_clean50/hf_nyu_val_x8_per_image.csv, results/kaggle_hf_x8_clean50/hf_nyu_val_x8_summary.json, results/kaggle_hf_x8_clean50/hf_nyu_val_x8_summary.md

  evaluated 10/50
  evaluated 20/50
  evaluated 30/50
  evaluated 40/50
  evaluated 50/50
