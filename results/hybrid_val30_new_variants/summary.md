# Hybrid validation comparison

NYU validation, first 30 images, x4, clean bicubic degradation, CUDA
(NVIDIA GeForce RTX 5050 Laptop GPU). Metrics use the repository evaluation
protocol: 6-pixel border crop, fixed 10 m data range.

| Method | RMSE | MAE | PSNR | SSIM | Boundary RMSE | Edge F1 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical | 0.0599 | 0.0199 | 45.49 | 0.9882 | 0.2764 | 0.754 | 542.0 |
| DL | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 | 1992.9 |
| DL + consistency | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 | 2156.9 |
| Fixed alpha 0.01 | 0.0123 | 0.0036 | 59.58 | 0.9994 | 0.0537 | 0.976 | 2130.4 |
| Fixed alpha 0.02 | 0.0123 | 0.0036 | 59.58 | 0.9994 | 0.0537 | 0.975 | 2133.8 |
| Fixed fusion + consistency | 0.0123 | 0.0036 | 59.58 | 0.9994 | 0.0537 | 0.975 | 2145.4 |
| Adaptive fusion | 0.0124 | 0.0037 | 59.52 | 0.9994 | 0.0541 | 0.974 | 2148.1 |
| Adaptive + consistency | 0.0123 | 0.0037 | 59.53 | 0.9994 | 0.0541 | 0.974 | 2148.1 |
| Adaptive + boundary refinement | 0.0141 | 0.0046 | 58.17 | 0.9991 | 0.0608 | 0.964 | 2238.3 |

The selected validation candidate is fixed alpha 0.02 by RMSE
(0.0122816 in the evaluator JSON), but the improvement over DL is small.
The full per-image output was generated during evaluation but is not included
in this commit; rerun the documented command with `--cache-dir` to regenerate
it.
