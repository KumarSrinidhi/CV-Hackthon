# Deep Model Improvements — P4 Deep Branch (SGNet)

Companion to `docs/EXPERIMENT_LOG.md` (full experiment registry) and `docs/PERFORMANCE_ANALYSIS.md` (root-cause evidence). Governing plan: `plan.txt`.

## 1. Original architecture and training configuration (unchanged)

- **Model:** official SGNet (AAAI 2024, Apache-2.0), vendored in `deep/sgnet/`. x4 variant: 36.4 M params (140.1 MB fp32). Topology: RGB ResBlocks, 4 depth ResidualGroups (RCAB), GCM Sobel-gradient module, 3 SDM bridges (invertible InvBlock + Fourier SDB fusion + channel attention), bicubic-residual head.
- **Original checkpoints:** official pretrained `checkpoints/sgnet/SGNet_X4_R.pth` (sha256 `ebab512732b2c629…`), X8, X16. Never modified.
- **Original training (by SGNet authors):** synthetic NYU-v2, clean bicubic LR only. Repo fine-tuning before this work: masked L1, Adam 1e-5, head-only (`deep/finetune.py`); the auxiliary gradient output `out_re` was unsupervised.

## 2. Confirmed weaknesses (evidence in PERFORMANCE_ANALYSIS.md)

1. Deep loses to bicubic under 10% random holes (0.0438 vs 0.0374, val150) — no missing-measurement prior; error +41% near holes.
2. Deep under strong noise (σ=0.05 m) is far worse than bicubic (0.0741 vs 0.0529).
3. Boundary error dominates clean RMSE (edge regions 12× smooth; per-scene RMSE↔edge-density Spearman 0.945).
4. The hybrid fusion (legacy evidence weights) damaged the deep branch 3× on clean data.
5. Hybrid-vs-deep regressions concentrate on deep's hardest scenes.

## 3. Implemented improvements

### 3.1 E4 — Degradation-curriculum fine-tuning (`deep/finetune_v2.py`)

New training script (original `finetune.py` untouched): pluggable losses (masked L1 / smooth-L1 / BerHu + Sobel-gradient L1 on the main output and/or `out_re` + multiscale), degradation curriculum via `data.degradation.DegradationConfig` (training operator == evaluation operator), multi-regime validation, cosine schedule, regime weighting.

- **E4a (head scope): REJECTED** — holes −30% but clean +51% (head cannot behave regime-conditionally).
- **E4b (full model, lr 1e-6, cosine, 3 epochs): KEPT (provisional)** — val150: holes 0.0438→0.0344, noise 0.02: 0.0312→0.0227, noise 0.05: 0.0741→0.0315; clean cost 0.0087→0.0090 (+3.4%).
- **E5 (+gradient supervision λ_g=0.1): REJECTED** — no meaningful change (−0.1%, within noise). `out_re` supervision left off.
- **E6 (clean-weighted sampling 0.40): REJECTED** — no meaningful change; the clean drift is inherent to fine-tuning on the NYU bundle, not the regime mix.
- **E7 (6 epochs): REJECTED as a single checkpoint; KEPT as the robust-slot checkpoint** — best holes/noise performer (val150 holes 0.0333, noise 0.0219) at clean 0.0130 (30-img subset).

### 3.2 E8 — Regime-aware checkpoint selection (`deep/adaptive.py`)

Deployable LR-only regime detection (no GT):

- **holes** = any invalid LR pixel;
- **noise** = median |Laplacian| of valid LR pixels > 4e-2 (calibrated on val: clean ∈ [0.0077, 0.0245], noise σ=0.02 ∈ [0.0640, 0.0818] — perfectly separated; threshold = geometric midpoint).

Selects original checkpoint (clean specialist) or E7 robust checkpoint. Measured (`results/e8_val150_*/`, `results/e8_hf_clean100/`):

| regime | original deep | deep_adaptive |
|---|---|---|
| clean (HF, n=645) | 0.0102 / F1 0.968 | **0.0102 / F1 0.968 (identical)** |
| holes 10% (val150) | 0.0438 | **0.0333** (−24%, beats bicubic) |
| noise 0.02 (val150) | 0.0312 | **0.0219** (−30%) |
| noise 0.05 (val150) | 0.0741 | **0.0293** (−60%) |

**Checkpoint provenance:** robust slot = `runs/e7_6epoch/best.pth`, SHA256 `9ed571170c55a3a9…` (`runs/` is gitignored; reproduce with the E7 config in EXPERIMENT_LOG.md). Clean slot = original official checkpoint. `SGNET_CKPT_ROBUST` env overrides the robust slot; `SGNET_CKPT_DIR`/`SGNET_CKPT_X4` isolation disables the ambient robust default (hermetic tests).

## 4. Mathematical formulation of the training objective (finetune_v2)

`L = λ_d·L_depth + λ_g·L_grad + λ_m·L_multiscale`

- `L_depth`: masked L1 (or smooth-L1 β=0.05 / BerHu with c = 0.2·max|e|) in valid-LR-normalized depth.
- `L_grad`: |Sobel(pred) − Sobel(target)|₁ over the valid interior (central-difference kernels identical to SGNet's `Get_gradient_nopadding_d`), optionally also on `out_re`.
- `L_multiscale`: masked L1 on 1/2 and 1/4 avg-pooled predictions (weight 0.5 each).
- Kept configuration: λ_d=1, λ_g=0, λ_m=0 (masked L1) — auxiliary losses showed no measurable benefit (E5).

## 5. Before/after summary (val150 unless noted)

| regime | bicubic | original deep | improved deep (E8) |
|---|---|---|---|
| clean (HF n=645) | 0.0391 (F1 0.273) | 0.0102 (F1 0.968) | **0.0102 (F1 0.968)** |
| clean (HF n=100 subset) | 0.0367 (F1 0.259) | 0.0101 (F1 0.960) | **0.0101 (F1 0.960)** |
| holes 10% | 0.0374 | 0.0438 | **0.0333** |
| noise 0.02 m | 0.0345 | 0.0312 | **0.0219** |
| noise 0.05 m | 0.0529 | 0.0741 | **0.0293** |

Clean benchmark preserved exactly; robustness improved 24–60%. The clean-benchmark RMSE/F1 targets (≤0.0101 / ≥0.960) are met by the improved deep branch (E8 routes clean to the original checkpoint).

## 6. Limitations

- Regime detection covers holes (exact) and Gaussian sensor noise (validated for σ ≥ 0.02 absolute); other real-sensor artefacts (e.g., structured missing regions with valid masks present, correlated noise) are not explicitly detected.
- The robust checkpoint was fine-tuned on the NYU SGNet train bundle only; cross-dataset transfer is untested.
- Boundary-error improvements from gradient supervision were not realized (E5); boundary RMSE remains dominated by scene edge density.
- x8/x16 have no robust fine-tune (E8 adaptive switching is x4-only by construction; x8/x16 use official checkpoints).
