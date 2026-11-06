# Limitations

## Not locally executed

- **Deep branch** (`deep_hr_depth`): not run on this machine — CPU-only, no SGNet
  checkpoint, and no NYU/Middlebury dataset downloaded. The integration is
  code-verified and the teammate's `results/` are preserved, but local deep/hybrid
  inference on real data was not performed.
- **Gradio UI**: written but not launched (gradio is not installed here).

## Model / scale limitations

- SGNet has official checkpoints for **x4, x8, x16** only; **x2 is classical-only**.
- SGNet has no validity-mask input; invalid LR pixels are nearest-filled before
  inference (the hybrid decides how much to trust the deep output at those pixels).

## Datasets

- NYU Depth v2 and Middlebury are the intended benchmarks; **KITTI compatibility is
  not verified** in this work.
- Datasets and checkpoints are not committed (see `.gitignore`).

## Algorithmic

- Confidence weights are **normalized heuristics, not calibrated probabilities**.
- Measurement consistency is only exact under the known bicubic degradation
  operator; real sensor formation is not claimed to be invertible.
- The classical JBU is a hard-gated splat; for large images the per-valid-sample
  Python loop is the current implementation (correct, but not heavily optimized).
- The deep branch's tiled-inference fallback (used on OOM) is not bit-identical to
  full-image inference.

## Deployment

- No live hosted URL is provided yet (requires a hosting runtime + gradio).
- `eval_P4_T<teamID>.md` has not been generated because the Team ID and USN list
  were not supplied.
