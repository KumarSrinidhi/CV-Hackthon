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
    --scale 4 --method hybrid [--gt gt.npy] [--compare] [--depth-scale 0.001] --output out/
```

```python
from main import reconstruct, compare, comparison_image
reconstruct(rgb, lr_depth, valid_mask=None, scale=4, method="hybrid", config=None)
# -> {"depth", "method", "requested_method", "fallback", "notes", "intermediates"}
compare(rgb, lr_depth, valid_mask, scale, methods=("bicubic", "classical", "hybrid"))
from app.config import load_config, hybrid_kwargs     # configs/default_config.yaml
from app.validation import validate_inputs            # clear ValueError messages
from app.ui import process                            # Gradio-free UI core
```

`reconstruct` validates inputs first, applies the frozen hybrid settings from the config, and
degrades hybrid to classical **explicitly** (`fallback=True`, reason in `notes`) when the deep
branch is unavailable or the scale is unsupported.
