# Hybrid Fusion

The hybrid is a **genuine** fusion of two HR reconstructions — it is not a
side-by-side display and not a global fixed blend.

## Confidence estimation

For each pixel, two normalized confidence scores `W_c` (classical) and `W_d`
(deep) are estimated with `W_c + W_d == 1`. They are **normalized confidence
scores, not calibrated probabilities**. Three evidence terms (each in `[0, 1]`)
are combined with a fixed convex combination:

```
edge_score   = normalized Sobel magnitude of D_c          (structure)
dist_score   = normalized distance to nearest valid LR px (measurement proximity)
agreement    = RGB edge ∩ depth-discontinuity band        (edge agreement)

W_c_raw = 0.4 * edge_score + 0.3 * (1 - dist_score) + 0.3 * agreement
W_d_raw = 1 - W_c_raw
```

Intuition:
- The classical branch (edge-aware JBU) is trusted **on structure/edges**, **near
  valid measurements**, and **where RGB and depth edges agree**.
- The deep branch (learned prior) is trusted **on flat regions** and **far from
  measurements** (where it supplies a prior).

Invalid deep pixels (`NaN`/`Inf`/`<= 0`) force `W_d = 0`, `W_c = 1`, so a bad
branch never corrupts the other.

## Adaptive fusion

```
D_f = W_c * D_c + W_d * D_d,   W_c + W_d == 1
```

The weights are **spatial** (per pixel), not one global scalar. `fuse_fixed` is a
fixed-weight ablation baseline.

## Boundary refinement

`D_f` is refined using depth discontinuities as primary evidence and RGB structure
as supporting evidence (reusing `classical.boundary_refinement.refine_boundaries`).
This preserves foreground/background boundaries and prevents RGB texture from being
copied into depth.

## Measurement consistency

When the forward degradation operator `S` is known (bicubic downsampling used to
build the synthetic LR measurements):

```
r_LR = D_LR - S(D_f)          # at valid LR pixels only
D_f' = D_f + upsample(r_LR)   # nearest (block-constant) upsample
```

For unknown real sensor formation processes this exact residual is **not** claimed;
it is applied only when `operator_exact=True` and with the matching degradation
mode (`bicubic`, `bilinear`, or `nearest`). Noisy or hole-punched measurements
must not use this correction because their forward process includes additional
operations beyond `S`.

The public pipeline exposes `consistency_mode` and `operator_exact`; it rejects
consistency requests when the operator is not exact rather than silently applying
the bicubic default.

## Ablations

`hybrid/ablations()` returns: `classical`, `deep`, `fixed_fusion`,
`confidence_fusion`, `confidence_boundary`, and `final` (confidence + boundary +
measurement consistency).

For validation tuning of a fixed classical contribution, run:

```
python scripts/eval_hybrid_ablation.py --split val --max-samples 30
```

This evaluates Deep alone and 0%, 2%, 5%, 10%, and 20% classical blends with
refinement and consistency disabled. It then selects the lowest-RMSE blend and
reports the best nonzero blend with full and selective refinement, consistency,
and both stages added independently/together. It also writes per-image CSV
rows. Add `--max-samples` only for a smoke run; validation tuning should use
the complete validation split.
The JSON report is written to `results/hybrid_ablation/summary.json`.
