# P4 — Edge-Preserving Depth Map Upsampling & Refinement

An edge-aware, hybrid depth reconstruction system that transforms **sparse or
low-resolution depth maps** into **dense, high-resolution depth**, preserving object
boundaries and depth discontinuities. It combines classical computer vision,
deep-learning depth super-resolution, and a confidence-guided hybrid fusion.

## What is implemented

| Layer | Module | Status |
|-------|--------|--------|
| Classical | `classical/` — nearest/bilinear/bicubic, joint bilateral upsampling (JBU), guided filter, hole filling, depth-discontinuity detection, boundary refinement, adaptive JBU | Implemented & tested |
| Deep | `deep/` — SGNet (guided depth super-resolution), dataset loaders, degradation, preprocessing, inference | Implemented (needs checkpoint) |
| Hybrid | `hybrid/` — confidence estimation, adaptive fusion, boundary refinement, measurement consistency | Implemented & tested (needs deep branch) |
| Evaluation | `eval/` — RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, benchmark | Implemented |
| Application | `main.py` (CLI) + `app/ui.py` (Gradio) + `app/visualization.py` (export) | Implemented |

## Quick start

```bash
# Environment (Python 3.12+)
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
# For the web UI only:
pip install gradio

# Run reconstruction
python main.py --rgb RGB.png --depth depth.npy --scale 4 --method hybrid --output out/

# Run tests
pytest tests/                            # 90 passed, 1 skipped (CUDA-only test)
```

## Reconstruction methods

- `nearest`, `bilinear`, `bicubic` — interpolation baselines
- `jbu` — joint bilateral upsampling
- `guided` — guided filter
- `classical`, `adaptive_jbu`, `refined_jbu` — classical edge-preserving variants
- `deep` — SGNet (scales 4/8/16; requires the official checkpoint)
- `hybrid` — confidence-guided fusion of classical + deep

## Stable interfaces

```python
classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
deep_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
```

Conventions (frozen in `data/preprocessing.py`):
- `hr_rgb` is `(H, W, 3)` RGB, uint8 or float32 in `[0, 1]`.
- `lr_depth` is `(h, w)` float32 in the dataset's metric unit (metres for NYU).
- `valid_mask` is `(h, w)` bool at low resolution; `None` means "derive from depth".
- Output is `(H, W)` float32 in the same unit as `lr_depth`.
- Invalid depth (<= 0, NaN, Inf) is never treated as a measurement.

## Hybrid fusion

The hybrid combines the classical reconstruction `D_c` and deep reconstruction `D_d`
per pixel:

```
D_f = W_c * D_c + W_d * D_d,   W_c + W_d == 1
```

`W_c`/`W_d` are normalized confidence scores from three pieces of evidence:
1. **Structure** — normalized depth-gradient magnitude (classical preferred on edges).
2. **Measurement proximity** — distance to the nearest valid LR pixel (classical
   preferred near measurements, deep preferred far away).
3. **RGB/depth edge agreement** — where an RGB edge agrees with a depth
   discontinuity (classical preferred).

Boundary refinement is optional and disabled by default. When the forward degradation
operator is known, the fused map can receive a configurable, damped measurement-consistency correction
`r_LR = D_LR - S(D_f)`. See `docs/HYBRID_FUSION.md`.

The weights are **normalized confidence scores, not calibrated probabilities**. A
deep branch that is unavailable (no checkpoint / unsupported scale) makes
`hybrid_hr_depth` raise `NotImplementedError`, so evaluation reports it as
"not evaluated" rather than silently substituting classical output.

## Deep-learning branch (SGNet)

The deep branch adapts the official **SGNet** guided depth super-resolution model
(Apache-2.0; see `deep/sgnet/LICENSE`). One checkpoint per scale: **x4, x8, x16**
(x2 has no released checkpoint). Set `SGNET_CKPT_DIR` (default `checkpoints/sgnet`)
or `SGNET_CKPT_X{scale}` to the official `*.pth` files. See `deep/README_DEEP.md`.

## Evaluation

```bash
python -m classical.benchmark --scale 4        # synthetic classical benchmark
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic classical deep hybrid
```

Metrics: RMSE, MAE, PSNR, SSIM (fixed data range, 10 m for NYU), boundary RMSE
(RMSE inside a band around **ground-truth depth** discontinuities), edge
precision/recall/F1 (depth edges, not RGB edges), latency.

Teammate-produced deep/eval results are preserved under `results/`.

## Datasets

- **NYU Depth v2** — primary RGB-D benchmark (metres, dense GT → synthetic sparse/LR).
- **Middlebury Stereo** — additional guided depth SR benchmark (8-bit disparity).
- **KITTI** — listed for compatibility; **not verified** in this work.

Datasets and checkpoints are **not** committed (see `.gitignore`); they must be
placed under `datasets/` and `checkpoints/` respectively.

## Limitations

- The deep branch was **not run on this machine** (CPU-only, no SGNet checkpoint,
  no dataset downloaded); its integration is code-verified and the teammate's
  `results/` are preserved, but local deep/hybrid inference was not executed.
- SGNet supports scales **4, 8, 16** only; scale **2** is classical-only.
- The web UI requires `gradio` (optional, not installed by default).
- Confidence weights are normalized heuristics, not calibrated probabilities.

See `docs/LIMITATIONS.md` and `docs/IMPLEMENTATION_STATUS.md`.
