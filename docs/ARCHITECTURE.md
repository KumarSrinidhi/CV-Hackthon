# P4 — System Architecture & Workflow

## 1. High-Level Architecture Diagram

```text
                 INPUT
       ┌───────────────────────────┐
       │ HR RGB + LR Depth + Mask  │
       └─────────────┬─────────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
 ┌──────────────────┐  ┌──────────────────┐
 │ Classical Branch │  │   Deep Branch    │
 │     Tanush       │  │    Rakshitha     │
 └────────┬─────────┘  └────────┬─────────┘
          │                     │
          ▼                     ▼
    Classical HR           Deep HR
     Depth D_c              Depth D_d
          │                     │
          └──────────┬──────────┘
                     ▼
           ┌─────────────────────┐
           │   Hybrid Fusion     │
           │      Rishita        │
           │                     │
           │ Confidence Fusion   │
           │ Consistency Check   │
           │ Boundary Refinement │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │ Final HR Depth Map  │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │ UI / Visualization  │
           │ Kumar               │
           └─────────────────────┘
```

## 2. Module Responsibilities

1. **Deep Learning Branch (`deep/`) — Rakshitha**:
   - Neural network model (SGNet adapter) executing forward passes on RTX 5050 GPU.
   - Preserves shared public contract `deep_hr_depth(...)`.

2. **Classical Computer Vision Branch (`classical/`) — Tanush**:
   - Edge-preserving upsampling (Joint Bilateral Upsampling, Guided Image Filtering).
   - Conservative hole filling that preserves measured depth values.
   - Discontinuity preservation avoiding RGB texture bleed.

3. **Hybrid Fusion Branch (`hybrid/`) — Rishita**:
   - Per-pixel confidence weighting ($W_c + W_d = 1.0$) using edge alignment and residual consistency.
   - Backward measurement consistency projection ($r_{LR} = D_{LR} - S(D_f)$).
   - Adaptive boundary refinement.

4. **Application and Coordination (`app/`, `main.py`, `configs/`) — Kumar**:
   - Streamlit web interface with file upload and side-by-side visualization.
   - System integration, error handling, configuration management, and benchmark coordination.
