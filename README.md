# P4 — Edge-Preserving Depth Map Upsampling & Refinement

An edge-aware, hybrid depth reconstruction system that transforms **sparse or
low-resolution depth maps** into **dense, high-resolution depth**, preserving object
boundaries and depth discontinuities. It combines classical computer vision,
deep-learning depth super-resolution, and input-adaptive hybrid fusion with measurement consistency.

## What is implemented

| Layer | Module | Status |
|-------|--------|--------|
| Classical | `classical/` — masked measurement fitting with RGB-guided depth regularization, plus interpolation, JBU and guided-filter baselines | Implemented & tested |
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
python main.py --rgb RGB.png --depth depth.npy --scale 4 --output out/

# Web UI (needs gradio)
python -m app.ui

# Run tests
pytest tests/
```

### Google Colab

Open [`CV_Hackthon_Colab.ipynb`](./CV_Hackthon_Colab.ipynb) in Google Colab and run
the cells from top to bottom. Select a GPU runtime for SGNet inference. The notebook
clones this repository, installs the runtime dependencies, uploads RGB/depth inputs,
runs the same `main.reconstruct` pipeline as the CLI and Gradio app, displays the
results, and downloads the generated depth files. SGNet checkpoints are not stored in
Git; upload them in the optional checkpoint cell to enable the deep branch. Without
them, the notebook reports and uses the explicit classical fallback.

The full suite includes PyTorch tests and runs CUDA checks when a GPU is
available. See `docs/APP_GUIDE.md` for the demo workflow and deployment.

## Reconstruction methods

- `nearest`, `bilinear`, `bicubic` — interpolation baselines
- `jbu` — joint bilateral upsampling
- `guided` — guided filter
- `classical` — iterative reconstruction that fits the observed LR samples and sharpens depth-supported edges with RGB guidance
- `adaptive_jbu`, `refined_jbu` — bilateral upsampling variants
- `deep` — SGNet (scales 4/8/16; requires the official checkpoint)
- `hybrid` — blends classical and deep predictions based on LR noise and missing measurements

## Stable interfaces

```python
classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4, *, mode="bicubic") -> (H, W) float32
deep_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
```

Conventions (frozen in `data/preprocessing.py`):
- `hr_rgb` is `(H, W, 3)` RGB, uint8 or float32 in `[0, 1]`.
- `lr_depth` is `(h, w)` float32 in the dataset's metric unit (metres for NYU).
- `valid_mask` is `(h, w)` bool at low resolution; `None` means "derive from depth".
- Output is `(H, W)` float32 in the same unit as `lr_depth`.
- Invalid depth (<= 0, NaN, Inf) is never treated as a measurement.

The classical solver defaults to the antialiased bicubic sampling model used by the
project's benchmarks. Pass `mode="bilinear"` or `mode="nearest"` when the LR data
was produced by that sampling model. Hole fills initialize the estimate, and only
valid LR pixels contribute to its measurement error.

## Hybrid fusion

The default `hybrid.classical_alpha: null` adapts the classical fraction to the
noise and missing pixels in the LR input. The blend starts at 2% for clean input
and can reach 65% for noisy input or 90% for incomplete input. A numeric
`classical_alpha` in `configs/default_config.yaml` selects a fixed fraction.

The measurement-consistency correction uses the bicubic sampling model. The
adaptive setting reduces its strength as estimated noise increases. Boundary
refinement is disabled by default. If SGNet is unavailable, the app reports the
fallback and returns the standalone classical reconstruction.

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
The [reconstruction audit](results/submission_audit/README.md) records the
current pipeline's cross-view robustness results and reproduction commands.

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

## Contributors

| Name | Area | GitHub |
|------|------|--------|
| Rakshitha | Deep learning, data & evaluation | [@Rakshithaa5](https://github.com/Rakshithaa5) |
| Tanush | Classical computer vision | [@tanushg07](https://github.com/tanushg07) |
| Rishita | Hybrid fusion | [@itsrishitacodeforspace](https://github.com/itsrishitacodeforspace) |
| Kumar | Application, integration & deployment | [@KumarSrinidhi](https://github.com/KumarSrinidhi) |

## Limitations

- The current [audit](results/submission_audit/README.md) verifies real SGNet
  and hybrid inference at ×4 and ×8. It uses cropped Middlebury disparity
  scenes with simulated LR inputs.
- SGNet supports scales **4, 8, 16** only; scale **2** is classical-only.
- The web UI requires `gradio` (optional, not installed by default).
- Confidence weights are normalized heuristics, not calibrated probabilities.

See `docs/LIMITATIONS.md` and `docs/IMPLEMENTATION_STATUS.md`.
