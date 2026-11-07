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
