# Deep Learning Track — P4 Depth Super-Resolution

Owner: Kumar (improvements). Model: SGNet (AAAI 2024, Apache-2.0), vendored under `deep/sgnet/`.

## Architecture

`SGNet` (see `deep/sgnet/sgnet_net.py`): RGB ResBlocks → depth ResidualGroups (RCAB channel attention) → GCM Sobel-gradient module (auxiliary gradient output `out_re`) → 3 SDM bridges (invertible InvBlock + Fourier-domain SDB fusion + attention) → tail ResidualGroups → DenseProjection upsampler → bicubic-residual output. Variants: x4 (36.4 M params), x8 (39.9 M), x16 (86.6 M). Input: RGB [0,1] + LR depth (native LR resolution, per-image valid-pixel min-max normalized); output: HR depth in metres, clamp ≥ 0.

## Inference paths

- **Fixed:** `deep.inference.deep_hr_depth(lr, rgb, valid, scale)` — original behaviour, unchanged contract `(H,W) float32`.
- **Adaptive (recommended):** `deep.adaptive.deep_hr_depth_adaptive(...)` — LR-only regime detection (holes → invalid LR pixels; noise → median |Laplacian| > 4e-2, val-calibrated) selects the original (clean) or robust-curriculum checkpoint. On clean data the output is bit-identical to the fixed path.

## Checkpoints

| slot | file | sha256 (16) | use |
|---|---|---|---|
| clean x4 | `checkpoints/sgnet/SGNet_X4_R.pth` | `ebab512732b2c629` | clean regime (official pretrained) |
| robust x4 | `runs/e7_6epoch/best.pth` | `9ed571170c55a3a9` | holes/noise regimes (fine-tuned, `deep/finetune_v2.py`) |
| x8 / x16 | `checkpoints/sgnet/SGNet_X{8,16}_R.pth` | see GPU_RUN_REPORT | fixed only (no robust fine-tune) |

## Training (improvements)

`deep/finetune_v2.py` (see `docs/DEEP_MODEL_IMPROVEMENTS.md` for the full experiment history):

- Curriculum: {clean, holes 0.1, noise 0.02, block holes, combined} via `data.degradation.DegradationConfig` (training operator = evaluation operator).
- Full-model fine-tune, Adam 1e-6, cosine schedule, grad-clip 1.0, 128 px patches, batch 1, flips-only augmentation.
- Validation: 150-image val split subsets (30 images per regime); selection on cross-regime mean; test split untouched.
- FP32 only (cuFFT rejects bf16 — verified empirically).

## Measured performance (x4, val150 / HF val)

| regime | original deep | improved deep (adaptive) |
|---|---|---|
| clean (HF n=645) | 0.0102 m / F1 0.968 | identical |
| holes 10% | 0.0438 m / F1 0.817 | 0.0333 m / F1 0.861 |
| noise σ=0.02 m | 0.0312 m | 0.0219 m |
| noise σ=0.05 m | 0.0741 m | 0.0293 m |

Latency ~1.85–1.9 s/image (RTX 5070, 480×640, fp32); peak VRAM ~3.5–4.5 GB full-image, tiled OOM fallback exists but produces seam artefacts (see LIMITATIONS).

## Known limitations

- Scales {4, 8, 16} only — no official x2 weights (scale-2 requests fall back to classical/bicubic baselines).
- Adaptive switching implemented for x4 only.
- Regime detection is calibrated for absolute Gaussian noise ≥ 0.02 m and random/block holes; unvalidated for other real-sensor artefacts.
- Boundary RMSE remains edge-density-dominated (see `docs/PERFORMANCE_ANALYSIS.md`).
