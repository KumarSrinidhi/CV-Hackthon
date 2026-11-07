# Hybrid Fusion

The hybrid combines two HR reconstructions with fixed or spatially gated weights.
Weights are normalized scores, not calibrated probabilities.

## Confidence estimation

Adaptive weighting is conservative and capped by `max_classical_weight` (default
`0.1`). Classical weight requires RGB/depth edge agreement, then uses valid
measurement support and the residual under the forward operator as modulators.
These signals are heuristics, not proof of higher HR accuracy. `adaptive=False`
disables the gate and gives the deep branch zero-alpha behavior. Invalid deep
pixels still force classical fallback for safety.

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
