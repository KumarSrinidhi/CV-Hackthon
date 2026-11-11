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

## Design specification vs. what is implemented

The design specification (the "Complete System Architecture" document) is the target;
this table records where the shipped code follows it and where it deliberately differs.

| Spec item | Status |
|---|---|
| Input validation (dims, scale, invalid depth, RGB-depth alignment) | Implemented in `app/validation.py` |
| Original depth / validity mask preserved | Implemented; branches work on copies |
| Classical branch (interpolation, JBU, GF, hole filling, boundary refinement) | Implemented (`classical/`); no separate `joint_bilateral.py`, JBU lives in `jbu.py` |
| Deep branch | **SGNet** (x4/x8/x16). FGGNet is not implemented. |
| Confidence estimation + confidence-guided fusion | Implemented, but the **deployed default is fixed alpha 0.02** (validation winner); adaptive fusion is opt-in via config |
| Measurement consistency `r_LR = D_LR - S(D_f)` | Implemented, enabled (strength 0.25) because the bicubic operator is known |
| Adaptive boundary refinement (`B*JBF + (1-B)*GF`) | A refinement exists but is **off by default**: it raised validation RMSE (0.0141 vs 0.0123) |
| Scales 2x / 4x / 8x | Classical x2/4/8/16; deep and hybrid x4/8/16 (x2 falls back to classical, flagged) |
| Output layer: RGB, LR depth, classical, deep, hybrid, error map | Side-by-side `comparison` image (shared colour range) + error map with GT |
| Export PNG / NPY | Implemented |
| Export PLY point cloud | Not implemented (optional; needs camera intrinsics) |
| Model uncertainty in confidence | Not available from SGNet |
| Peak GPU memory / ablation scripts | In `eval/` (Rakshitha) |

**Deployment principle from the spec:** the hybrid should show measurable improvement over both
standalone branches, otherwise the best standalone method should be deployed. On the 30-image NYU
validation subset the fixed-alpha hybrid (RMSE 0.0123) is statistically indistinguishable from the
deep branch alone (0.0123), so the improvement is **not demonstrated**. The hybrid is kept as the
default per the team decision; if a larger evaluation shows no gain, switch the app default to
`deep` by setting `app.default_method: deep` in `configs/default_config.yaml`.
