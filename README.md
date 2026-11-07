---
title: Edge-Preserving Depth Map Super-Resolution (P4)
emoji: 🏔️
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Map Super-Resolution and Refinement

A hybrid deep-learning and classical computer vision architecture for edge-preserving depth map upsampling (2×, 4×, 8×) with RGB structural guidance, backward measurement consistency enforcement, and adaptive boundary refinement.

---

## 1. System Architecture

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
 │ (Tanush)         │  │ (Rakshitha)      │
 │ JBU / Guided     │  │ EdgeGuidedSRNet  │
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
           │ (Rishita)           │
           │                     │
           │ Confidence Blending │
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
           │ (Kumar)             │
           └─────────────────────┘
```

---

## 2. Key Modules & Ownership

| Owner | Module | Responsibilities |
|---|---|---|
| **Tanush** | `classical/` | Nearest, bilinear, bicubic baselines; Joint Bilateral Upsampling (`jbu.py`); Guided Filtering (`guided_filter.py`); Conservative hole-filling (`hole_filling.py`); and Boundary refinement (`boundary_refinement.py`). |
| **Rakshitha** | `deep/`, `data/`, `eval/` | `EdgeGuidedSRNet` guided depth SR architecture; NYU-v2 / synthetic RGB-D pipelines; official metrics (`RMSE`, `MAE`, `PSNR`, `SSIM`, `Boundary RMSE`, `Edge F1`). |
| **Rishita** | `hybrid/` | Confidence-weighted blending ($W_c + W_d = 1.0$), fixed-weight ablation baseline ($\alpha=0.02$), backward measurement consistency projection ($r_{LR} = D_{LR} - S(D_f)$), and adaptive boundary refinement. |
| **Kumar** | `app/`, `main.py`, `configs/` | Visualization with Turbo/Plasma colormaps, side-by-side comparison figure export, CLI execution workflow, and repository integration. |

---

## 3. Quickstart & Usage

### Setup
Activate the environment and verify dependencies:
```bash
pip install -r requirements.txt
```

### Run Full Demonstration & Visualization
Executes single-scene reconstruction across Bicubic, Classical, Deep, and Hybrid tracks, printing metrics and saving `outputs/depth_sr_comparison.png`:
```bash
python3 main.py --mode demo --scale 4
```

### Run Benchmark Evaluation
Evaluates all methods across dataset samples and reports the metric comparison table:
```bash
python3 main.py --mode benchmark --scale 4 --num_samples 5
```

### Run Multi-Scale Profiling (2×, 4×, 8×)
Profiles latency and memory across multiple upsampling scale factors:
```bash
python3 main.py --mode scale_benchmark
```

### Run Full Unit Test Suite
Runs all 14 unit tests across classical, deep, hybrid, evaluation, and application modules:
```bash
pytest tests/
```

---

## 4. Benchmark Findings & Ablation Analysis

Validation on the NYU-v2 test set indicates:
1. **Hybrid Consistency Gain**: Damped measurement consistency enforcement ($D_f + \gamma \cdot S^\dagger(r_{LR})$) improves RMSE and PSNR over standalone deep learning by anchoring reconstructions to physically verified sensor measurements.
2. **Fixed-Weight Fusion ($\alpha=0.02$)**: Small classical injection ($\alpha \approx 0.02$) provides edge stability while preventing RGB texture bleed in planar depth regions.
3. **Selective vs. Full Refinement**: Full bilateral boundary refinement can over-smooth sharp depth steps; keeping boundary refinement selective to agreed discontinuities ($|\nabla D| > \tau_d \land |\nabla I| > \tau_c$) maintains higher Edge F1 scores.
