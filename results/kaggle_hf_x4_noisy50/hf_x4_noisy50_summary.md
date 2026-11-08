### hf_nyu_val / test / x4  (n=50, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.02, 'noise_relative': False, 'hole_ratio': 0.0, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0538 | 0.0258 | 47.10 | 0.9894 | 0.1925 | 0.345 | 1.6 |
| classical | 0.0773 | 0.0271 | 44.82 | 0.9836 | 0.2988 | 0.703 | 581.7 |
| hybrid | not evaluated | | | | | | DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4. |
| deep | not evaluated | | | | | | all samples failed, e.g. ["00001: DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4."] |

  evaluated 10/50
  evaluated 20/50
  evaluated 30/50
  evaluated 40/50
  evaluated 50/50
