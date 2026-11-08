# Evaluation Methodology

All methods are evaluated on the **same** held-out samples, depth units, scale
factors, evaluation masks, and degradation protocol.

## Metrics (`eval/metrics.py`)

| Metric | Definition |
|--------|-----------|
| RMSE | `sqrt(mean((pred - gt)^2))` over the evaluation mask (metres) |
| MAE | `mean(|pred - gt|)` over the evaluation mask |
| PSNR | `10 log10(data_range^2 / MSE)`, fixed `data_range` (10 m for NYU, 255 for Middlebury) |
| SSIM | mean SSIM over the mask-eroded window core |
| boundary RMSE | RMSE inside a `radius`-pixel band around **ground-truth depth** discontinuities |
| edge precision/recall/F1 | predicted depth edges vs ground-truth depth edges (tol-pixel tolerance) |

Depth discontinuities are defined **from depth only** (`|jump| > rel_thresh *
data_range`), never from RGB edges. A border of `DEFAULT_BORDER = 6` px is removed.

## Protocol

1. **Degradation** (`data/degradation.py`): bicubic downsample → optional noise →
   optional holes. `lr_valid_mask` marks invalid pixels; invalid depth is stored as 0.
2. **Splits**: proper train/val/test separation; the test split is locked behind
   `--final` in `eval/evaluate.py`.
3. **Unavailable methods**: a missing checkpoint or stub is reported as
   "not evaluated" with the reason, never silently dropped.
4. **Scale factors**: classical supports 2/4/8; SGNet supports 4/8/16 (no x2).
5. **Hybrid variants**: `classical`, `deep`, `dl_consistency`, `fixed_fusion`,
   `fixed_fusion_consistency`, `adaptive_fusion`, `adaptive_fusion_consistency`,
   and `adaptive_fusion_boundary` use identical samples and masks. Fixed-alpha
   sweeps use `fixed_0`, `fixed_0.005`, `fixed_0.01`, `fixed_0.02`, `fixed_0.05`,
   and `fixed_0.1`.
   Consistency sweeps use names such as `dl_consistency_s0.1_i1`,
   `dl_consistency_s0.25_i2`, and `dl_consistency_s1.0_i4`.
6. **Selection and uncertainty**: select on validation RMSE only. Summaries include
   paired per-image RMSE differences, win rate, and deterministic paired bootstrap
   95% intervals versus `deep`. Test evaluation remains locked behind `--final`.

## Commands

```bash
python -m classical.benchmark --scale 4
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods classical deep dl_consistency fixed_fusion fixed_fusion_consistency adaptive_fusion adaptive_fusion_consistency adaptive_fusion_boundary --cache-dir results/cache
```

Teammate-produced results are preserved under `results/` (NYU and Middlebury,
x4/x8, clean/noise/holes).
# Hybrid/PS4 evaluation protocol

All methods for a comparison consume the same dataset sample, RGB image,
degraded LR depth, validity mask, scale, and ground truth. Validation is used
for selecting fusion alpha and post-processing; the test split is reserved for
the final verification command.

Metrics:

- **RMSE**: square root of mean squared depth error over valid GT pixels.
- **MAE**: mean absolute depth error over the same mask.
- **PSNR**: `10 log10(data_range^2 / MSE)` with fixed dataset range (10 m for NYU).
- **SSIM**: structural similarity with the fixed data range and the shared mask.
- **Boundary RMSE**: RMSE in a two-pixel band around ground-truth depth
  discontinuities.
- **Edge F1**: F1 between predicted and ground-truth depth discontinuities,
  using the fixed relative threshold and one-pixel tolerance. RGB edges are not
  used as ground truth.
- **Latency**: wall-clock inference time; CUDA runs synchronize before timing.
- **Peak RAM/GPU memory**: process RSS high-water mark and CUDA allocated/reserved
  peak memory observed during the profiling sample. RAM is process-level, so it
  includes model/runtime allocations retained between method calls. `n/a` means
  the platform or method did not expose that measurement.

Reproduction:

```
python scripts/eval_hybrid_ablation.py --split val --out results/hybrid_ablation
python -m eval.benchmark --dataset nyu --split val --scales 2 4 8 --methods nearest bilinear bicubic classical deep hybrid_fixed02 hybrid_adaptive hybrid_selected_clean --out results/benchmark_val --runs 10
python -m eval.benchmark --dataset nyu --split val --scales 2 4 8 --methods nearest bilinear bicubic classical deep hybrid_fixed02 hybrid_adaptive --noise-std 0.02 --hole-ratio 0.10 --out results/benchmark_noisy_holes --runs 10
python -m eval.benchmark --dataset nyu --split test --scales 2 4 8 --methods nearest bilinear bicubic classical deep hybrid_fixed02 hybrid_adaptive hybrid_selected_clean --final --out results/benchmark_test --runs 10
```

Unsupported scales/checkpoints and missing datasets are recorded as
`not evaluated`; they are not replaced with classical or random outputs.
`hybrid_selected_clean` is intentionally limited to the clean bicubic protocol;
do not include it in noisy/hole-punched runs.
