# API Reference

All depth functions return `(H, W) float32` NumPy arrays in the input depth unit.
Inputs are never modified in place.

## Classical

```python
from classical import classical_hr_depth
from classical.pipeline import reconstruct

classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
reconstruct(lr_depth, hr_rgb, valid_mask=None, scale=4, method="jbu") -> (H, W) float32
```

Methods for `reconstruct`: `nearest`, `bilinear`, `bicubic`, `jbu`, `guided`,
`classical`, `adaptive_jbu`, `refined_jbu`.

## Deep

```python
from deep.inference import deep_hr_depth, profile_inference

deep_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4, *, norm_range=None,
              device=None, tile_lr=None, ckpt_path=None) -> (H, W) float32
```

Raises `DeepModelUnavailableError` without a checkpoint. Scales: 4, 8, 16.

## Hybrid

```python
from hybrid import hybrid_hr_depth, ablations

hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4, *, use_consistency=True,
                deep_fn=None, return_intermediates=False) -> (H, W) float32
```

Raises `NotImplementedError` when the deep branch is unavailable. `deep_fn` is an
injectable `deep_fn(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32`.

## Hybrid internals

```python
from hybrid.confidence import estimate_confidence   # -> (W_c, W_d)
from hybrid.fusion import fuse, fuse_fixed          # -> D_f
from hybrid.consistency import enforce_consistency  # -> D_f'
from hybrid.boundary_refinement import refine_hybrid_depth
```

## Evaluation

```python
from eval.metrics import compute_all_metrics   # -> dict of metrics
```

## Application

```bash
python main.py --rgb RGB.png --depth depth.npy [--mask mask.npy] \
    --scale 4 --method hybrid [--gt gt.npy] --output out/
```
