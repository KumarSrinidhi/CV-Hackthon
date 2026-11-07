| dataset | split | scale | method | status | n | rmse | mae | psnr | ssim | boundary_rmse | edge_f1 | latency_ms_median | latency_ms_p95 | peak_ram_mb | peak_gpu_alloc_mb | peak_gpu_reserved_mb | repeat_max_abs_diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nyu | val | 2 | nearest | ok | 30 | 0.0297 | 0.0079 | 51.3440 | 0.9970 | 0.1469 | 0.9182 | 1.4429 | 1.5254 | 832.5664 | 0.0000 | 0.0000 | 0.0000 |
| nyu | val | 2 | bilinear | ok | 30 | 0.0240 | 0.0064 | 53.2265 | 0.9981 | 0.1188 | 0.6098 | 1.6450 | 1.7531 | 832.5664 | 0.0000 | 0.0000 | 0.0000 |
| nyu | val | 2 | bicubic | ok | 30 | 0.0205 | 0.0057 | 54.5889 | 0.9986 | 0.1009 | 0.6723 | 1.8743 | 1.9398 | 832.5664 | 0.0000 | 0.0000 | 0.0000 |
| nyu | val | 2 | classical | ok | 30 | 0.0356 | 0.0103 | 50.0093 | 0.9956 | 0.1726 | 0.8705 | 711.2034 | 716.8264 | 832.5664 | 0.0000 | 0.0000 | 0.0000 |
| nyu | val | 2 | deep | not evaluated: all samples failed, e.g. ['nyu_val_0000: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself.'] |  |  |  |  |  |  |  |  |  |  |  |  |  |
| nyu | val | 2 | hybrid_fixed02 | not evaluated: NotImplementedError: hybrid unavailable: deep branch not available: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself. |  |  |  |  |  |  |  |  |  |  |  |  |  |
| nyu | val | 2 | hybrid_adaptive | not evaluated: NotImplementedError: hybrid unavailable: deep branch not available: UnsupportedScaleError: Deep branch (SGNet) has verified checkpoints only for scales (4, 8, 16); got x2. Use the classical branch for this scale or train an SGNet x2 model yourself. |  |  |  |  |  |  |  |  |  |  |  |  |  |
| nyu | val | 4 | nearest | ok | 30 | 0.0545 | 0.0165 | 46.0582 | 0.9898 | 0.2629 | 0.8075 | 0.9478 | 0.9561 | 1630.0977 | 655.7563 | 4458.0000 | 0.0000 |
| nyu | val | 4 | bilinear | ok | 30 | 0.0422 | 0.0130 | 48.3177 | 0.9941 | 0.2071 | 0.2142 | 1.3502 | 1.4288 | 1630.0977 | 655.7563 | 4458.0000 | 0.0000 |
| nyu | val | 4 | bicubic | ok | 30 | 0.0372 | 0.0118 | 49.4112 | 0.9949 | 0.1778 | 0.2952 | 1.5786 | 1.7813 | 1630.0977 | 655.7563 | 4458.0000 | 0.0000 |
| nyu | val | 4 | classical | ok | 30 | 0.0599 | 0.0199 | 45.4948 | 0.9882 | 0.2764 | 0.7536 | 287.3611 | 302.7067 | 1630.0977 | 655.7563 | 4458.0000 | 0.0000 |
| nyu | val | 4 | deep | ok | 30 | 0.0123 | 0.0036 | 59.5689 | 0.9994 | 0.0537 | 0.9763 | 1748.6887 | 1749.9689 | 1630.0977 | 3468.8613 | 4458.0000 | 0.0000 |
| nyu | val | 4 | hybrid_fixed02 | ok | 30 | 0.0123 | 0.0037 | 59.5534 | 0.9994 | 0.0538 | 0.9751 | 2059.1576 | 2061.2038 | 1630.0977 | 3468.8613 | 4458.0000 | 0.0000 |
| nyu | val | 4 | hybrid_adaptive | ok | 30 | 0.0338 | 0.0087 | 50.4355 | 0.9966 | 0.1670 | 0.8751 | 2160.1359 | 2215.5289 | 1630.0977 | 3468.8613 | 4458.0000 | 0.0000 |
| nyu | val | 8 | nearest | ok | 30 | 0.0875 | 0.0314 | 41.9158 | 0.9748 | 0.3941 | 0.4622 | 1.0763 | 1.1006 | 1799.5156 | 1232.5381 | 4802.0000 | 0.0000 |
| nyu | val | 8 | bilinear | ok | 30 | 0.0701 | 0.0253 | 43.9566 | 0.9867 | 0.3231 | 0.0375 | 1.6521 | 1.9167 | 1799.5156 | 1232.5381 | 4802.0000 | 0.0000 |
| nyu | val | 8 | bicubic | ok | 30 | 0.0634 | 0.0238 | 44.8100 | 0.9872 | 0.2928 | 0.0572 | 1.2954 | 1.4642 | 1799.5156 | 1232.5381 | 4802.0000 | 0.0000 |
| nyu | val | 8 | classical | ok | 30 | 0.0931 | 0.0355 | 41.6216 | 0.9756 | 0.3873 | 0.5693 | 154.0175 | 157.1629 | 1799.5156 | 1232.5381 | 4802.0000 | 0.0000 |
| nyu | val | 8 | deep | ok | 30 | 0.0259 | 0.0089 | 53.0736 | 0.9974 | 0.1117 | 0.9303 | 1248.8177 | 1250.7165 | 1799.5156 | 3380.9399 | 4802.0000 | 0.0000 |
| nyu | val | 8 | hybrid_fixed02 | ok | 30 | 0.0259 | 0.0089 | 53.0686 | 0.9974 | 0.1118 | 0.9286 | 1411.6161 | 1412.2172 | 1799.5156 | 3380.8794 | 4802.0000 | 0.0000 |
| nyu | val | 8 | hybrid_adaptive | ok | 30 | 0.0530 | 0.0167 | 46.5407 | 0.9914 | 0.2401 | 0.7492 | 1449.0925 | 1451.1481 | 1799.5156 | 3380.9399 | 4802.0000 | 0.0000 |
