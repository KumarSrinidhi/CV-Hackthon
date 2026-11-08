# Kaggle Run Analysis — Tesla T4, 2026-11-08

**Source notebook:** `notebookf065efbfd5.ipynb`
**Branch at run time:** `master` (old `classical` method, no checkpoints)
**GPU:** Tesla T4 | **torch:** 2.11.0+cu130 | **Python:** 3.13.16

---

## What this run tells us

The notebook ran on `master` before the `feat/research-driven-depth-completion`
improvements were merged. Consequently:

- `classical` here = old `reconstruct(..., method="classical")`, **not** the improved
  `classical_blend` default now in `classical_hr_depth()`.
- `deep` / `hybrid*` = **not evaluated** (no checkpoints attached).
- All numbers below are real GPU measurements on a Tesla T4.

---

## HF NYU val — x4 clean (n=100)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0535 | 0.0162 | 48.27 | 0.9890 | 0.2465 | 0.739 | 2.0 |
| bilinear | 0.0415 | 0.0132 | 50.53 | 0.9934 | 0.1943 | 0.205 | 1.4 |
| bicubic | 0.0367 | 0.0121 | 51.62 | 0.9942 | 0.1680 | 0.259 | 1.8 |
| classical (old) | 0.0571 | 0.0183 | 47.86 | 0.9890 | 0.2566 | 0.665 | 589.5 |

**Key finding:** old `classical` RMSE 0.0571 > bicubic 0.0367 (+56% worse).
The improved `classical_blend` (validated: RMSE 0.0250 on val150 clean) should
close this gap significantly when re-run on the new branch.

---

## HF NYU val — x2 clean (n=50)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0392 | 0.0105 | 50.88 | 0.9951 | 0.1585 | 0.906 | 1.8 |
| bilinear | 0.0320 | 0.0090 | 52.71 | 0.9968 | 0.1286 | 0.562 | 1.4 |
| bicubic | 0.0276 | 0.0080 | 54.03 | 0.9975 | 0.1102 | 0.617 | 1.6 |
| classical (old) | 0.0441 | 0.0130 | 49.66 | 0.9941 | 0.1805 | 0.795 | 1429.8 |

x2 classical latency: **1430 ms** (the JBU pixel-loop is O(h·w·scale²)).
`classical_blend` at x2 uses bicubic back-projection which is much faster.

---

## HF NYU val — x8 clean (n=50)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.1151 | 0.0424 | 41.41 | 0.9615 | 0.4272 | 0.447 | 1.8 |
| bilinear | 0.0913 | 0.0360 | 43.47 | 0.9785 | 0.3439 | 0.066 | 1.2 |
| bicubic | 0.0826 | 0.0336 | 44.36 | 0.9793 | 0.3116 | 0.102 | 1.7 |
| classical (old) | 0.1247 | 0.0479 | 40.88 | 0.9641 | 0.4312 | 0.544 | 367.9 |

---

## HF NYU val — x4 noise (noise_std=0.02, n=50)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0538 | 0.0258 | 47.10 | 0.9894 | 0.1925 | 0.345 | 1.6 |
| classical (old) | 0.0773 | 0.0271 | 44.82 | 0.9836 | 0.2988 | 0.703 | 581.7 |

---

## HF NYU val — x4 holes (hole_ratio=0.10, n=50)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0628 | 0.0203 | 46.79 | 0.9881 | 0.2416 | 0.360 | 3.0 |
| classical (old) | 0.1483 | 0.0305 | 39.34 | 0.9722 | 0.4948 | 0.645 | 570.0 |

**Key finding:** old classical RMSE 0.1483 under 10% holes — 2.4× worse than bicubic.
This confirms the known failure mode. `classical_blend` uses nearest-fill before
bicubic back-projection, which should be more robust to holes.

---

## Official testset — x2 (n=20, native missing depth)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.1601 | 0.0233 | 38.41 | 0.9882 | 0.6936 | 0.631 | 5.8 |
| bilinear | 0.1320 | 0.0159 | 39.99 | 0.9917 | 0.5937 | 0.669 | 5.4 |
| bicubic | 0.1390 | 0.0191 | 39.51 | 0.9903 | 0.6172 | 0.654 | 5.8 |
| classical (old) | 0.1991 | 0.0210 | 37.79 | 0.9868 | 0.5676 | 0.675 | 858.1 |

---

## Official testset — x4 (n=20, native missing depth)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.2497 | 0.0493 | 34.57 | 0.9699 | 0.9103 | 0.382 | 2.4 |
| bilinear | 0.2088 | 0.0353 | 36.16 | 0.9807 | 0.7976 | 0.456 | 2.2 |
| bicubic | 0.2178 | 0.0403 | 35.74 | 0.9771 | 0.8133 | 0.460 | 2.5 |
| classical (old) | 0.3089 | 0.0541 | 34.17 | 0.9698 | 0.8401 | 0.532 | 351.2 |

Note: testset x8 was not completed (cell-14 was still running when the session ended).

---

## What to re-run on the new branch

The next Kaggle run should use branch `feat/research-driven-depth-completion` with
checkpoints attached. Expected improvements vs this run:

1. **classical_blend** should beat bicubic on RMSE (validated: 0.0250 vs 0.0292 on val150).
2. **deep** (SGNet x4): RMSE ~0.0087 m on clean (from `results/e8_val150_clean/`).
3. **hybrid v2** with hole-band confidence: should recover the holes regime.
4. New metrics (absRel, delta1, iRMSE, iMAE, RMSE-hole) will appear in all tables.

Quote this run as: **"Kaggle Tesla T4, 2026-11-08, master branch, no checkpoints"**.
