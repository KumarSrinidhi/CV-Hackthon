# Performance Analysis — Forensic Root-Cause Investigation (Phase 1)

Governing plan: `plan.txt` §6. Every probe below has an entry in `docs/EXPERIMENT_LOG.md` and reusable code in `scripts/forensic_probes.py` / `scripts/error_analysis.py`. Raw probe outputs: `tmp/forensics/*.json`; per-scene artifacts: `results/error_analysis/`.

**Baseline context (reproduced, P0-02):** deep RMSE 0.0101 m, edge F1 0.960, 1900 ms (100 HF NYU val samples, clean bicubic x4, checkpoint `ebab512732b2c629…`).

Verdict legend: **CONFIRMED** (evidence sufficient), **NOT A PROBLEM** (evidence against), **UNCERTAIN** (insufficient evidence — no conclusion forced).

---

## 1. Pipeline trace (verified by direct code inspection)

LR depth → nearest-fill invalid → per-image valid-pixel min-max normalize → SGNet fp32 forward (RGB + LR input; bicubic-residual head; FFT/global blocks) → denormalize → clamp ≥ 0. Eval: 6 px border crop, fixed data_range 10 m, depth-only edges at |Δ| > 0.3 m, deterministic per-sample degradation (downsample → noise → holes).

## 2. A — Signal-processing and degradation-operator analysis

### A1 — Frequency response of the operator (`data/degradation.py::downsample`, PIL bicubic x4)

Method: (A1.1) empirical impulse response; (A1.2) exactly-periodic synthetic sinusoids + step edge; (A1.3) radial power-spectrum ratio on 5 real scenes. A naive single-scene FFT-ratio characterization was deliberately **not** used as the primary evidence (non-stationary depth signals; grid change).

Key measurements (`tmp/forensics/a1_operator.json`):

| HR frequency (cyc/px) | LR gain (sinusoid amp) | round-trip RMSE (down→bicubic-up) |
|---|---|---|
| 0.008 | 1.000 | 0.006 |
| 0.031 | 0.995 | 0.024 |
| 0.063 | 0.940 | 0.065 |
| 0.094 | 0.767 | 0.291 |
| 0.109 | 0.638 | 0.437 |
| 0.125 (=LR Nyquist) | 0.913* | 0.431 |
| 0.156 (aliased) | 0.230 | 0.689 |
| 0.250 | ~0 | 0.707 |

\* at exactly Nyquist the tone lands on the folded bin; treat 0.094–0.125 as the rolloff region.

- Impulse response: 4×4 LR support, weights sum 1/16 per HR delta (= 1/scale², consistent with per-output normalized antialias kernel); DC exact (constants preserved to 0.0).
- Step edge: ~5% overshoot, ~4 px transition.
- Real scenes: LR retains ~0.53–0.65 of HR radial power across 0.004–0.20 cyc/px (coarse, scene-mixed diagnostic; the synthetic sinusoid curve is the cleaner operator characterization).

**Interpretation.** The operator preserves ≤0.06 cyc/px nearly losslessly and destroys most content above ~0.12 cyc/px, with partial aliasing of the 0.13–0.16 band (gain ≈ 0.23). Any depth SR method must reconstruct the 0.06–0.12 band from context; that is where boundary sharpness lives.

**Verdict:** CONFIRMED as characterization (operator is strongly band-limiting above ~0.06 cyc/px with mild aliasing). **No information-theoretic limit is claimed** (plan.txt A1/A3d).

### A2 — Depth-noise physics

Current protocol: additive Gaussian, absolute σ in metres (optional relative mode exists). Measured RMSE-vs-depth after bicubic up for three parameterizations on 10 real scenes (`tmp/forensics/a2_noise.json`): all curves grow with depth (upsampling error grows too); at 9 m: absolute σ=0.02 → 0.255, relative 2% → 0.308, quadratic (Kinect-v1-like, σ∝z², c=0.0015) → 0.276.

**Verdict:** UNCERTAIN as to which model is "physically correct" in general; CONFIRMED that absolute noise slightly under-covers far range relative to depth-dependent models. **Decision:** keep the absolute-σ curriculum (it matches the existing evaluation protocol `--noise-std` exactly); document the caveat; do not silently change the degradation definition.

### A3 — Sanity checks, controlled reconstruction, oracles (25 samples; `tmp/forensics/a3_diagnostics.json`)

| Diagnostic | Result | Verdict |
|---|---|---|
| A3a identity pipeline (pred=GT through metrics) | RMSE 0.0, edge F1 1.0; nearest-fill/normalize round-trip err ≤ 9.5e-7 | NOT A PROBLEM (plumbing is exact). Sanity check only — **never evidence of x4 capability** |
| A3b operator round trip: bicubic-up vs SGNet on same LR | bicubic 0.0449 / F1 0.377 vs SGNet 0.0120 / F1 0.971 | SGNet reconstructs 3.7× better than bicubic under the same operator (controlled comparison, not a bound) |
| A3c-1 GT-range normalization oracle | 0.012004 vs deployable 0.012005 (≤0.01%) | NOT A PROBLEM — per-image valid-LR minmax costs nothing measurable on clean data (non-deployable diagnostic) |
| A3c-2 spectral oracle (bicubic-up + GT highpass, σ=0.75 px) | 0.0293 / F1 0.825 — **worse than SGNet** | UNCERTAIN / rejected as a bound: the graft intervention does not dominate SGNet, so no upper bound was established. SGNet already recovers high-frequency structure beyond naive spectral grafting |
| A3c-3 GT-edge-mask oracle | covered by per-image edge precision/recall split (§C): precision 0.986 / recall 0.957 — localization error is small and balanced | reported as diagnostic decomposition |
| A3d irreducible floor | **not claimed** (per plan.txt); headroom statements below refer only to measured diagnostics | — |

## 3. B — Numerical and evaluation audit

### B1 — Metric correctness (`tmp/forensics/b1_metric_audit.json`)

Independently reimplemented RMSE/MAE/PSNR/edge/boundary metrics match `eval/metrics.py` to **max abs diff 0.0** on 3 scenes. Degenerate handling correct (constant depth → NaN edge F1; NaN pred excluded; border crop right).

Threshold sensitivity (3 scenes, mean edge F1): rel_thresh 0.01→0.98 (tol 2), 0.03→0.948 (tol 1, the repo convention), 0.05→0.918 (tol 1).

**Verdict:** NOT A PROBLEM (implementation correct). CONFIRMED caveat: edge F1 is threshold-sensitive; the historical convention (0.3 m, tol 1) is preserved unchanged for all comparisons.

### B2 — Fusion-equation audit (25 samples × clean/holes; `tmp/forensics/b2_fusion_audit.json`)

Equation verified in code: `wc = 0.4·edge + 0.3·(1−dist) + 0.3·agreement` (all terms ∈[0,1], wc∈[0,1]).

| Regime | deep-only | classical-only | fused (current) | wc_scale=0 (deep-only+consistency) | oracle per-pixel selector (diagnostic) |
|---|---|---|---|---|---|
| clean | 0.0119 | 0.0705 | 0.0357 | 0.0119 | 0.0102 |
| holes 10% | 0.0671 | 0.2066 | 0.0924 | 0.0671 | 0.0287 |

- Pixels where fusion is worse than deep-only: **69.6%** (clean), 55.5% (holes).
- RMSE increases monotonically with classical weight in BOTH regimes.
- Consistency enforcement math verified: LR residual 0.0049 → 0.0017 after projection.

**Verdict:** CONFIRMED — the current confidence equation damages the deep branch on this benchmark; classical is never competitive here (its edge strength does not offset its much larger depth error). The oracle selector shows only ~14% headroom over deep-only on clean data but ~57% under holes — the potential value of fusion is concentrated in degraded regimes, and realizing it requires *regime-aware* trust, not fixed weights. (GT-informed selector is a diagnostic, never deployable.)

### B3 — Inference numerical audit (25 samples; `tmp/forensics/b3_inference_audit.json`)

| Check | Result | Verdict |
|---|---|---|
| Normalization variants (valid-LR minmax vs 0.5–99.5 pct vs fixed [0,10] m) | 0.012005 / 0.012009 / 0.012053 | NOT A PROBLEM (≤0.4% spread on clean data) |
| Holes: error near vs far from missing LR pixels | 0.0212 vs 0.0151 (+41% near holes) | CONFIRMED — missing-depth weakness is spatially localized at hole boundaries |
| Tiling (tile_lr=32 vs full) | max per-pixel seam diff up to 0.48 m; aggregate RMSE delta ≈ −7e-5 | CONFIRMED (edge case) — the OOM-only tiled fallback has large localized seam errors; document, do not use for benchmarking |
| bf16 autocast | `RuntimeError: Unsupported dtype BFloat16` (cuFFT path) | CONFIRMED constraint — fp32 is required, as documented |

## 4. C — Empirical per-scene / per-region analysis (100 samples; `results/error_analysis/`)

### C0 — Rankings and correlations (`rankings.json`)

- Spearman(deep RMSE, GT edge density) = **0.945**; Spearman(deep RMSE, depth range) = 0.921; (valid fraction) = −0.24.
- Worst-RMSE scenes: 00119 (0.0390), 00186 (0.0331), 00088, 00089, 00091, 00084, 00118 (0.0236–0.0306). Worst boundary RMSE: 00186 (0.164), 00172 (0.102).
- Worst edge F1: 00063 (0.800), 00153 (0.831) — floor is high; most scenes ≥ 0.89.
- **Deep beats bicubic on every one of the 100 scenes** (max "regression" −0.0027).
- Hybrid's largest regressions vs deep occur exactly on deep's hardest scenes (00029 +0.108, 00091, 00088, 00089, 00119) — consistent with B2.

### C1 — Region-wise RMSE (mean masked RMSE, deep unless noted)

| Region | bicubic | classical | deep | hybrid |
|---|---|---|---|---|
| Smooth surfaces | 0.0160 | 0.0284 | **0.0054** | 0.0112 |
| Edge band (±2 px of GT discontinuity) | 0.1169 | 0.2027 | **0.0217** | 0.0883 |
| On GT edges | 0.2536 | 0.3298 | **0.0668** | 0.2016 |
| Thin structures | 0.1881 | 0.1784 | **0.0345** | 0.1068 |
| Strong depth edges (RGB-aligned) | 0.3312 | 0.4695 | 0.0961 | 0.2873 |
| Depth edges with weak RGB edges | 0.3218 | 0.4609 | 0.1086 | 0.2281 |
| RGB-texture / flat depth | 0.0267 | 0.0439 | 0.0068 | 0.0221 |

**CONFIRMED:** boundary regions carry 4–12× the smooth-region error; thin structures 6.4×; weak-RGB edges are worse than RGB-aligned edges (RGB guidance is doing real work at boundaries). Texture-copying is mild (texture/flat-depth only 1.26× smooth).

### C2 — Visual atlas

Panels for the 6 worst-RMSE scenes in `results/error_analysis/atlas_*.png` (GT, LR, RGB, per-method predictions, shared-scale absolute error maps, GT-vs-predicted edge overlay).

### C3 — RGB-guidance investigation (25 samples; `rgb_probe/rgb_probe_summary.json`)

| RGB input | RMSE | boundary RMSE | edge F1 | edge recall |
|---|---|---|---|---|
| original | 0.0120 | 0.0522 | 0.971 | 0.957 |
| constant grey | 0.0226 | 0.1012 | 0.858 | 0.771 |
| blurred | 0.0220 | 0.0977 | 0.868 | 0.789 |
| block-shuffled | 0.0252 | 0.1149 | 0.886 | 0.830 |
| other sample | 0.0248 | 0.1140 | 0.909 | 0.867 |

Interpretation with the distribution-shift caveat (plan.txt C3): all artificial RGB variants roughly double RMSE, so part of the drop is input-shift, not proof of geometric use. But the *specific* failure signature is informative: edge **recall** collapses (0.96 → 0.77–0.87) while **precision** stays ≥0.96 even for mismatched RGB — RGB drives boundary *detection*, and misaligned RGB does not create many false edges. Texture-copying into flat regions is mild (C1).

**Verdict:** CONFIRMED that RGB provides genuine geometric value concentrated in edge recall; CONFIRMED (mild) texture leakage; module-level causality within SGNet not claimed.

## 5. Confirmed problems, ruled-out hypotheses, uncertainties

### Confirmed (ranked per plan.txt §16)
1. **[Fusion correctness/impact] Current hybrid fusion degrades the deep branch on clean data 3× and under holes ~1.4×** (B2, C0). Root cause: fixed evidence weights trust the classical branch on any structure, but classical depth error (0.07 clean / 0.21 holes) dwarfs deep error. → Address in Phase 8 with validation-calibrated, regime-aware weights; the oracle diagnostic bounds the possible gain (~14% clean, ~57% holes).
2. **[Robustness] Deep error is spatially concentrated around missing measurements (+41% near holes)** (B3) and the deep branch offers no completion prior for large holes (historical: deep ≈ bicubic under holes — to be re-verified in Phase 6). → E4 robustness fine-tuning is the highest-priority training experiment.
3. **[Reconstruction] Boundary error dominates clean RMSE**: edges carry 12× smooth-region error, per-scene RMSE correlates 0.94 with edge density, and worst scenes combine high edge density with large depth range (C0/C1). → E1 (gradient supervision of the unsupervised `out_re` head) and E2 (loss upgrade) target this directly.
4. **[Edge case] Tiled OOM fallback produces seam errors up to 0.48 m** (B3) — never triggered by the benchmark (no OOM), documented; any future latency work must not enable tiling silently.

### Ruled out (NOT A PROBLEM)
- Metric implementation bugs (B1: exact match to independent reimplementation).
- Pipeline/masking/normalization plumbing (A3a exact; A3c-1 and B3: normalization choice ≤0.4% effect).
- Texture-copying as a major error source (C1: 1.26× smooth; C3: precision stays high).
- Bicubic-style operator mismatch in the harness (degradation is the exact known operator; `operator_exact` verified True).

### Uncertain
- Whether a spectral GT-informed bound can be constructed at all that dominates SGNet (A3c-2 failed to).
- Physically correct noise parameterization for the intended application (A2).
- Scene-level identity of failure causes beyond edge density/range proxies (no scene labels available in the HF samples).

## 6. Ranked improvement roadmap (feeds Phases 3–8)

| Priority | Action | Targets | Expected gain bound (measured diagnostics) |
|---|---|---|---|
| 1 | E4 robustness fine-tuning (holes/noise curriculum via `DegradationConfig`) | holes-regime collapse | oracle selector: up to ~57% RMSE reduction under holes is achievable by better branch selection; model-side completion should recover a substantial part |
| 2 | E1 gradient supervision (`out_re`) + E2 loss upgrade | boundary error (12× smooth) | A1: 0.06–0.12 cyc/px band is partially recoverable; boundary RMSE worst scenes ≥ 0.16 |
| 3 | E3 full-model fine-tune (only if head saturates) | global reconstruction | A3b shows large bicubic→SGNet gap already exploited; gains likely small |
| 4 | Phase 8: regime-aware calibrated fusion | hybrid damage (3× clean) | oracle selector: clean ceiling 0.0102 vs deep 0.0119; holes ceiling 0.0287 vs 0.0671 |
| 5 | Phase 7 latency (last) | 1.9 s/img | quality-gated |

*No claim in this document relies on an assumed irreducible floor; all headroom figures refer to measured oracle diagnostics (non-deployable).*
