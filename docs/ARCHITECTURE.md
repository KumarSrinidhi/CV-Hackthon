# P4 — System Architecture

## High-level data flow

```
            INPUT
   HR RGB + LR Depth + Validity Mask
                │
        ┌───────┴────────┐
        ▼                ▼
  Classical Branch    Deep Branch
  (classical/)        (deep/, SGNet)
        │                │
      D_c (HR)         D_d (HR)
        └───────┬────────┘
                ▼
         Hybrid Fusion (hybrid/)
         1. Confidence estimation  (W_c, W_d)
         2. Adaptive fusion        D_f = W_c*D_c + W_d*D_d
         3. Boundary refinement
         4. Measurement consistency
                │
                ▼
         Final HR depth (H, W) float32
                │
        ┌───────┴────────┐
        ▼                ▼
    CLI (main.py)    Gradio UI (app/ui.py)
        + export (app/visualization.py)
```

## Module responsibilities

### `classical/` (classical branch)
- `interpolation.py` — nearest / bilinear / bicubic baselines with missing-depth support.
- `jbu.py` — joint bilateral upsampling (hard colour gate + optional depth gate).
- `guided_filter.py` — validity-aware guided filter (He et al., box-filter based).
- `hole_filling.py` — `prepare_depth` (validity resolution) and conservative hole filling.
- `boundary_refinement.py` — depth-discontinuity detection and edge-aware refinement.
- `adaptive_jbu.py` — depth-vs-RGB edge agreement.
- `classical_hr_depth.py` — public `classical_hr_depth` wrapper + `reconstruct` router.
- `benchmark.py` — self-contained synthetic benchmark.

### `deep/` (deep branch, SGNet)
- `deep_model.py` — SGNet construction + checkpoint loading/verification.
- `inference.py` — `deep_hr_depth` wrapper, device handling, tiled inference, profiling.
- `settings.py` — supported scales (4/8/16), checkpoint resolution, exceptions.
- `sgnet/` — vendored SGNet implementation (Apache-2.0).
- `finetune.py`, `finetune_robust.py` — training/fine-tuning scripts (GPU).

### `hybrid/` (hybrid branch)
- `confidence.py` — `estimate_confidence` → normalized `W_c`, `W_d`.
- `fusion.py` — `fuse` (adaptive) and `fuse_fixed` (ablation).
- `consistency.py` — measurement-consistency correction.
- `boundary_refinement.py` — `refine_hybrid_depth`.
- `pipeline.py` — `hybrid_hr_depth` orchestrator + `ablations`.

### `data/` (shared conventions)
- `preprocessing.py` — frozen input conventions, RGB/depth conversion, masks, geometry.
- `degradation.py` — the forward operator `S` (bicubic downsample) + noise/holes.
- `dataset.py`, `middlebury_raw.py` — dataset loaders (NYU/Middlebury).

### `eval/`
- `metrics.py` — RMSE, MAE, PSNR, SSIM, boundary RMSE, edge precision/recall/F1.
- `evaluate.py` — masked, same-sample evaluation; honest "not evaluated" reporting.
- `benchmark.py` — latency/memory benchmarking.

### `app/` + `main.py`
- `visualization.py` — colormaps, boundary/error/confidence overlays, PNG/NPY export.
- `ui.py` — Gradio UI (lazy import).
- `main.py` — CLI entry point + method dispatch.

## Key invariant

Original depth measurements and validity masks are **never modified in place** and
**never confused with filled values**. Depth discontinuities are detected from depth
first; RGB edges are supporting evidence only.
