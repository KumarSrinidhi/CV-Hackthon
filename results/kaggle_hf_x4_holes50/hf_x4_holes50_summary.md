### hf_nyu_val / test / x4  (n=50, unit=m, data_range=10.0, border=6px, degradation={'mode': 'bicubic', 'noise_std': 0.0, 'noise_relative': False, 'hole_ratio': 0.1, 'n_block_holes': 0, 'block_frac': 0.15, 'seed': 0})

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0628 | 0.0203 | 46.79 | 0.9881 | 0.2416 | 0.360 | 3.0 |
| classical | 0.1483 | 0.0305 | 39.34 | 0.9722 | 0.4948 | 0.645 | 570.0 |
| hybrid | not evaluated | | | | | | DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4. |
| deep | not evaluated | | | | | | all samples failed, e.g. ["00001: DeepModelUnavailableError: No SGNet x4 checkpoint found in '/kaggle/working/CV-Hackthon/checkpoints/sgnet'. Download the official *_R.pth files (link in the SGNet README) and place e.g. SGNet_X4_R.pth there, or set SGNET_CKPT_DIR / SGNET_CKPT_X4."] |


  evaluated 10/20
  evaluated 20/20
