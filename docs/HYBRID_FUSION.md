# Hybrid Fusion

The hybrid combines two HR reconstructions with fixed or spatially gated weights.
Weights are normalized scores, not calibrated probabilities.

## Frozen final configuration

The application runs the hybrid with the configuration in
`configs/default_config.yaml` (loaded by `app/config.py`, passed by `main.reconstruct`):

| Setting | Value |
|---|---|
| `classical_alpha` | `0.02` (classical 0.02, deep 0.98) |
| `use_consistency` | `true`, strength `0.25`, 1 iteration |
| `use_boundary_refinement` | `false` |

Evidence: `results/hybrid_val30_new_variants/summary.md` (NYU val, first 30 images, x4,
clean bicubic degradation). Fixed alpha 0.02 has the lowest RMSE (0.0122816), but the
difference to the deep branch alone (0.0123) is within noise for 30 images; the claim is
"no worse than deep, marginally better on this subset", not a large improvement. Consistency
is valid here only because the synthetic bicubic forward operator is known; for real sensor
data with an unknown degradation, set `use_consistency: false`.

Set `classical_alpha: null` to switch back to the adaptive confidence fusion below.

## Confidence estimation

Adaptive weighting is conservative and capped by `max_classical_weight` (default
`0.35`). The classical weight starts at a 0.01 floor and grows only where the
classical branch has boundary support (depth-gradient + RGB/depth edge agreement)
and explains the measured LR depth better than the deep branch (measurement
residual under the forward operator). These signals are heuristics, not proof of
higher HR accuracy. `adaptive=False` disables the gate and gives the deep branch
zero-alpha behavior. Invalid deep pixels still force classical fallback for
safety.

## Boundary refinement

Boundary refinement uses depth discontinuities as primary evidence and RGB as
support. It is retained for ablation, but is disabled by default because full or
selective refinement can worsen validation RMSE.

## Measurement consistency

When the forward degradation operator `S` is known:

```
r_LR = D_LR - S(D_f)          # at valid LR pixels only
D_f' = D_f + strength * upsample(r_LR)
```

`strength` and `iterations` are configurable. Residual upsampling is an
implementation approximation rather than an exact projection. Invalid LR pixels
are ignored on every iteration; noisy inputs should use a damped strength or the
uncorrected baseline.

## Ablations

`hybrid/ablations()` returns classical, deep, DL+consistency, fixed fusion,
fixed fusion+consistency, adaptive fusion, adaptive fusion+consistency, and the
boundary-refined final ablation.
