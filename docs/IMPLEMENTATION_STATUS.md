# Implementation Status

Legend: ✅ implemented & tested · ⚠️ implemented, not locally run · ❌ blocked

## Classical branch — ✅

- ✅ Nearest / bilinear / bicubic interpolation (`classical/interpolation.py`)
- ✅ Joint bilateral upsampling (`classical/jbu.py`)
- ✅ Guided filtering (`classical/guided_filter.py`)
- ✅ Depth-discontinuity detection + boundary refinement (`classical/boundary_refinement.py`)
- ✅ Hole filling / invalid-depth handling (`classical/hole_filling.py`)
- ✅ Adaptive JBU + RGB/depth edge agreement (`classical/adaptive_jbu.py`)
- ✅ Public wrapper `classical_hr_depth` + `reconstruct` (`classical/classical_hr_depth.py`)
- ✅ Synthetic benchmark (`classical/benchmark.py`)

Evidence: `tests/test_classical.py` (39 passed).

## Deep branch (SGNet) — ⚠️

- ✅ SGNet model + checkpoint verification (`deep/deep_model.py`)
- ✅ `deep_hr_depth` wrapper, device/tiled inference, profiling (`deep/inference.py`)
- ✅ Dataset loaders, degradation, preprocessing (`data/`)
- ✅ Evaluation metrics (`eval/metrics.py`)
- ✅ Fine-tuning scripts (`deep/finetune.py`, `deep/finetune_robust.py`)
- ⚠️ **Not run on this machine**: CPU-only, no SGNet checkpoint, no dataset downloaded.
- Preserved teammate results under `results/` (NYU/Middlebury, x4/x8).

Evidence: `tests/test_deep.py` passes (checkpoint-dependent tests use a temporary
checkpoint fixture; CUDA test is skipped). Local `deep_hr_depth` raises
`DeepModelUnavailableError` without a checkpoint, as designed.

## Hybrid branch — ✅ (logic) / ⚠️ (deep-dependent)

- ✅ Confidence estimation (`hybrid/confidence.py`)
- ✅ Adaptive + fixed-weight fusion (`hybrid/fusion.py`)
- ✅ Measurement consistency (`hybrid/consistency.py`)
- ✅ Boundary refinement (`hybrid/boundary_refinement.py`)
- ✅ Pipeline + ablations (`hybrid/pipeline.py`)
- ✅ Honest "not evaluated" reporting: raises `NotImplementedError` when deep is unavailable.

Evidence: `tests/test_hybrid.py` (13 passed) using an injected deep branch; `hybrid`
is correctly reported as "not evaluated" by `eval/evaluate.py` when no checkpoint.

## Application — ✅

- ✅ CLI (`main.py`) — validated inputs, PNG/NPY export, metrics + error map with GT, `--compare`.
- ✅ Config (`configs/default_config.yaml`, `app/config.py`) — frozen hybrid: fixed alpha 0.02 + consistency.
- ✅ Input validation (`app/validation.py`) — clear errors for shape/scale/mask/method problems.
- ✅ Visualization/export/side-by-side comparison (`app/visualization.py`).
- ✅ Hybrid wired with the frozen config; explicit classical fallback when deep is unavailable.
- ⚠️ Gradio UI (`app/ui.py`) — all logic is in the Gradio-free `process()` and is tested; the
  Blocks layout itself was **not launched** (gradio not installed in the build environment).

Evidence: `tests/test_app.py` (9) and `tests/test_integration.py` (16) pass; CLI verified on synthetic
data. Hybrid wiring is verified with an injected deep branch (alpha 0.02 + consistency reproduced
exactly); real deep/hybrid inference still needs torch + the SGNet checkpoint.

## Pending / blocked

- ❌ Live hosted URL (Hugging Face Spaces) — not deployed; requires gradio + a runtime.
- ❌ `eval_P4_T<teamID>.md` — Team ID + USN list not yet provided.
- ❌ Local deep/hybrid inference on real NYU/Middlebury data — needs dataset + checkpoint.
- ❌ KITTI compatibility — not verified.
