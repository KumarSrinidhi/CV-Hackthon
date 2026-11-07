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

## Deep branch (SGNet) — ✅ (updated 2026-11-07)

- ✅ SGNet model + checkpoint verification (`deep/deep_model.py`)
- ✅ `deep_hr_depth` wrapper, device/tiled inference, profiling (`deep/inference.py`)
- ✅ Dataset loaders, degradation, preprocessing (`data/`)
- ✅ Evaluation metrics (`eval/metrics.py`)
- ✅ Fine-tuning scripts (`deep/finetune.py`, `deep/finetune_robust.py`, `deep/finetune_v2.py`)
- ✅ **Run on GPU (RTX 5070)**: baseline reproduced exactly (RMSE 0.0101 / F1 0.960); forensic analysis + improvement experiments completed (`docs/PERFORMANCE_ANALYSIS.md`, `docs/EXPERIMENT_LOG.md`).
- ✅ **Regime-aware adaptive inference** (`deep/adaptive.py`): LR-only detection selects original (clean) or robust-curriculum checkpoint (holes/noise). Clean outputs bit-identical to the original; degraded regimes improved 24–60% (`docs/EXPERIMENTAL_RESULTS.md`).
- ⚠️ Adaptive switching is x4-only; x8/x16 use official checkpoints as-is. Robust checkpoint trained on NYU bundle only (cross-dataset untested).

Evidence: full suite **116 passed** (`pytest tests/ -q`); per-regime results under `results/` (baseline_deep_repro, e4b/e7/e8_val150_*, e8_hf_clean100).

## Hybrid branch — ✅ (v2 fusion, updated 2026-11-07)

- ✅ Confidence estimation (`hybrid/confidence.py`) — legacy, preserved
- ✅ **v2 validation-calibrated fusion** (`hybrid/pipeline.py`, default): adaptive deep + constant classical weight under holes + consistency; legacy path via `fusion="legacy"`
- ✅ Adaptive + fixed-weight fusion (`hybrid/fusion.py`)
- ✅ Measurement consistency (`hybrid/consistency.py`) — verified numerically (residual 0.0049→0.0017)
- ✅ Boundary refinement (`hybrid/boundary_refinement.py`) — used by legacy path
- ✅ Pipeline + extended ablations (`hybrid/pipeline.py`)
- ✅ Honest "not evaluated" reporting: raises `NotImplementedError` when deep is unavailable.

Evidence: `tests/test_hybrid.py` + `tests/test_adaptive.py` (regression tests for v2 fusion, regime detection, hermeticity); weight-sweep calibration under `results/hybrid_v2_sweep/`.

## Application — ✅

- ✅ CLI (`main.py`) — reconstruction, PNG/NPY export, metrics with GT.
- ✅ Visualization/export (`app/visualization.py`).
- ⚠️ Gradio UI (`app/ui.py`) written but **not launched** (gradio not installed locally).

Evidence: `tests/test_app.py` (9 passed); CLI verified end-to-end on synthetic data.

## Pending / blocked

- ❌ Live hosted URL (Hugging Face Spaces) — not deployed; requires gradio + a runtime.
- ❌ `eval_P4_T<teamID>.md` — Team ID + USN list not yet provided.
- ✅ Local deep/hybrid inference on real NYU data — done (GPU session, 2026-11-06/07).
- ❌ KITTI compatibility — not verified.
- ⚠️ x8/x16 robust fine-tunes and cross-dataset (Middlebury) validation of the robust checkpoint — not done (documented in LIMITATIONS).
