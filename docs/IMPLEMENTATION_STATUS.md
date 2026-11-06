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

- ✅ CLI (`main.py`) — reconstruction, PNG/NPY export, metrics with GT.
- ✅ Visualization/export (`app/visualization.py`).
- ⚠️ Gradio UI (`app/ui.py`) written but **not launched** (gradio not installed locally).

Evidence: `tests/test_app.py` (9 passed); CLI verified end-to-end on synthetic data.

## Pending / blocked

- ❌ Live hosted URL (Hugging Face Spaces) — not deployed; requires gradio + a runtime.
- ❌ `eval_P4_T<teamID>.md` — Team ID + USN list not yet provided.
- ❌ Local deep/hybrid inference on real NYU/Middlebury data — needs dataset + checkpoint.
- ❌ KITTI compatibility — not verified.
