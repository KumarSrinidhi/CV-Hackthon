# P4 — Edge-Preserving Depth Map Upsampling & Refinement

An edge-aware, hybrid depth reconstruction system that transforms **sparse or
low-resolution depth maps** into **dense, high-resolution depth**, preserving object
boundaries and depth discontinuities. It combines classical computer vision,
deep-learning depth super-resolution, and a fixed-weight (alpha 0.02) hybrid fusion with measurement consistency.

## What is implemented

| Layer | Module | Status |
|-------|--------|--------|
| Classical | `classical/` — nearest/bilinear/bicubic, joint bilateral upsampling (JBU), guided filter, hole filling, depth-discontinuity detection, boundary refinement, adaptive JBU | Implemented & tested |
| Deep | `deep/` — SGNet (guided depth super-resolution), dataset loaders, degradation, preprocessing, inference | Implemented (needs checkpoint) |
| Hybrid | `hybrid/` — confidence estimation, adaptive fusion, boundary refinement, measurement consistency | Implemented & tested (needs deep branch) |
| Evaluation | `eval/` — RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, benchmark | Implemented |
| Application | `main.py` (CLI) + `app/ui.py` (Gradio) + `app/visualization.py` (export/compare) + `app/config.py` / `app/validation.py` (config, input checks) | Implemented |

## Quick start

```bash
# Environment (Python 3.12+)
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
# For the web UI only:
pip install gradio

# Run reconstruction (add --compare for a side-by-side image, --gt for metrics + error map)
python main.py --rgb RGB.png --depth depth.npy --scale 4 --method hybrid --output out/

# Web UI (needs gradio)
python -m app.ui

# Run tests
pytest tests/
```

The full suite needs `torch` for one deep-dependent test
(`test_consistency_sweep_method_name_is_supported`); without torch that single test fails
and everything else passes. See `docs/APP_GUIDE.md` for the demo workflow and deployment.

## Reconstruction methods

- `nearest`, `bilinear`, `bicubic` — interpolation baselines
- `jbu` — joint bilateral upsampling
- `guided` — guided filter
- `classical`, `adaptive_jbu`, `refined_jbu` — classical edge-preserving variants
- `deep` — SGNet (scales 4/8/16; requires the official checkpoint)
- `hybrid` — **fixed alpha 0.02 fusion of classical + deep, then measurement consistency** (frozen config)

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

**Frozen final configuration** (`configs/default_config.yaml`, section `hybrid`):
`D_f = 0.02 * D_c + 0.98 * D_d`, followed by measurement-consistency correction
(`strength 0.25`, 1 iteration); boundary refinement off. It was chosen on NYU validation
(30 images, x4): fixed alpha 0.02 gave the lowest RMSE (0.0123, vs. 0.0123 for the deep
branch alone), so the **gain over deep-only is marginal** and the classical branch is a
safeguard more than a contributor. The adaptive confidence fusion described below remains
available by setting `hybrid.classical_alpha: null`.

The general form of the fusion is the classical reconstruction `D_c` and deep reconstruction `D_d`
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

Boundary refinement is optional and disabled by default. Because the forward degradation
operator is known (bicubic), the fused map receives a damped measurement-consistency correction
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

## Deploy to AWS App Runner

The repo ships a `Dockerfile` (+ `.dockerignore`, `requirements.web.txt`): Gradio binds to
`0.0.0.0:$PORT` (default **8080**), so the App Runner port needs no override.

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name p4-depth-app
docker build -t p4-depth-app .
docker tag p4-depth-app:latest <account>.dkr.ecr.<region>.amazonaws.com/p4-depth-app:latest
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/p4-depth-app:latest

# 2. Create the service (an ECR access role is required; the console walks you through it,
#    or pass --source-configuration with the ECR image URI + AccessRoleArn)
aws apprunner create-service --service-name p4-depth-app \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account>.dkr.ecr.<region>.amazonaws.com/p4-depth-app:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {"Port": "8080"}
    },
    "AuthenticationConfiguration": {"AccessRoleArn": "<ecr-access-role-arn>"},
    "AutoDeploymentsEnabled": false
  }'
```

Notes:

- Health check default (`GET /`) works as-is; first boot takes ~20-30 s (Gradio import).
- Start at **2 vCPU / 4 GB** — image processing is CPU-bound.
- The image intentionally ships **without torch / SGNet checkpoints** (both gitignored), so the
  hybrid pipeline runs its classical fallback (`deep_available=False`). To enable the deep
  branch, add a CPU `torch` wheel to `requirements.web.txt`, copy `checkpoints/sgnet/*.pth`
  into the image, and set `SGNET_CKPT_DIR`.
- App Runner source-from-GitHub also works (it builds this `Dockerfile`); it just needs an ECR
  repo + connection configured in the console.

## Limitations

- The deep branch was **not run on this machine** (CPU-only, no SGNet checkpoint,
  no dataset downloaded); its integration is code-verified and the teammate's
  `results/` are preserved, but local deep/hybrid inference was not executed.
- SGNet supports scales **4, 8, 16** only; scale **2** is classical-only.
- The web UI requires `gradio` (optional, not installed by default).
- Confidence weights are normalized heuristics, not calibrated probabilities.

See `docs/LIMITATIONS.md` and `docs/IMPLEMENTATION_STATUS.md`.
