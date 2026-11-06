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

## Commands

```bash
python -m classical.benchmark --scale 4
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic classical deep hybrid
```

Teammate-produced results are preserved under `results/` (NYU and Middlebury,
x4/x8, clean/noise/holes).
