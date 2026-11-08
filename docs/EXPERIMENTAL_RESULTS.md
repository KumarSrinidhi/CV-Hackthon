# Experimental Results — P4 Deep Branch Improvements

All numbers from the official harness (`eval/evaluate.py` / `scripts/eval_all3_hf.py` / `scripts/eval_adaptive.py`); metrics: RMSE/MAE in metres (NYU, data_range 10 m, 6 px border), edge F1 at |Δ|>0.3 m with 1 px tolerance. Degradation is deterministic (per-sample seeds). Val150 = NYU validation split (first 150 of 449 test images); HF = HuggingFace FastDepth NYU val (`datasets/val-000000.tar`); the two populations are disjoint (split audit, P1-08).

**Improved deep** = `deep_hr_depth_adaptive` (E8: LR-only regime detection → original checkpoint on clean, E7 robust checkpoint under holes/noise).
**Improved hybrid** = hybrid v2 (adaptive deep + 0.05 classical weight under holes + consistency enforcement).

## 1. Clean benchmark (HF NYU val, x4, clean bicubic)

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0535 | 0.0162 | 48.27 | 0.9890 | 0.2465 | 0.739 | 2.7 |
| bilinear | 0.0415 | 0.0132 | 50.53 | 0.9934 | 0.1943 | 0.205 | 2.7 |
| bicubic | 0.0367 | 0.0121 | 51.62 | 0.9942 | 0.1680 | 0.259 | 2.7 |
| classical | 0.0584 | 0.0191 | 47.59 | 0.9874 | 0.2591 | 0.671 | 458.9 |
| deep (original) | 0.0101 | 0.0030 | 62.64 | 0.9994 | 0.0434 | 0.960 | 1900.1 |
| deep (improved/adaptive) | 0.0101 | — | — | — | — | 0.960 | ~1900 |
| hybrid (legacy, original) | 0.0298 | 0.0077 | 53.85 | 0.9966 | 0.1389 | 0.804 | 2455.0 |

Reproduced exactly at commit d5c4cab state (P0-02). The improved deep routes clean inputs to the original checkpoint — outputs are bit-identical (verified on the full 645-image val tar: deep 0.0102 / F1 0.968 both paths, `results/e8_hf_clean100/`).

## 2. Degraded regimes (NYU val150, x4)

### Random holes 10%

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 |
|---|---|---|---|---|---|---|
| bicubic | 0.0374 | 0.0120 | 49.47 | 0.9943 | 0.1901 | 0.231 |
| deep (original) | 0.0438 | 0.0084 | 48.25 | 0.9940 | 0.2189 | 0.817 |
| deep (improved) | **0.0333** | 0.0070 | 50.59 | 0.9963 | 0.1682 | 0.861 |
| hybrid (improved) | see §4 | | | | | |

### Noise σ = 0.02 m

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 |
|---|---|---|---|---|---|---|
| bicubic | 0.0345 | 0.0193 | 49.72 | 0.9940 | 0.1532 | 0.212 |
| deep (original) | 0.0312 | 0.0225 | 50.17 | 0.9920 | 0.0711 | 0.844 |
| deep (improved) | **0.0219** | 0.0146 | 53.45 | 0.9967 | 0.0657 | 0.868 |

### Noise σ = 0.05 m

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 |
|---|---|---|---|---|---|---|
| bicubic | 0.0529 | 0.0381 | 45.64 | 0.9831 | 0.1587 | 0.218 |
| deep (original) | 0.0741 | 0.0548 | 42.61 | 0.9551 | 0.1153 | 0.594 |
| deep (improved) | **0.0293** | 0.0192 | 50.85 | 0.9958 | 0.0988 | 0.708 |

## 3. Statistical reliability (paired per-image, scene = experimental unit)

E4b-vs-original on the clean HF benchmark (n=100): RMSE delta +0.00026 m, 95% CI [+0.00020, +0.00032], 99/100 scenes worse — statistically significant; this motivated E8 (regime-aware selection) rather than accepting a single robust checkpoint. Tool: `scripts/statistical_comparison.py`.

## 4. Hybrid v2 (results/hybrid_v2_*; final tables filled at evaluation completion)

Weight-sweep evidence (60 val samples per regime, `results/hybrid_v2_sweep/`):

| regime | deep_only (E7) | best fusion config | best RMSE | oracle per-pixel (diagnostic) |
|---|---|---|---|---|
| clean | 0.01015 | deep + consistency (α=0) | 0.00999 (−1.6%) | 0.00876 |
| holes 10% | 0.03659 | fixed α=0.05 + consistency | 0.03611 (−1.3%) | 0.01949 |
| noise 0.02 | 0.02396 | deep + consistency | 0.02396 (tie) | 0.01609 |

The legacy confidence equation (0.4 edge + 0.3 distance + 0.3 agreement) is never optimal and was damaging (probe B2: fused 0.0357 vs deep 0.0119 on clean). v2 removes it in favour of calibrated per-regime constants; classical contribution survives only under holes.

## 5. Multi-scale (x4 / x8; x2 not supported by SGNet)

See `results/x8_*` (pending) and `docs/LIMITATIONS.md`: no official SGNet x2 weights exist; x2 requests are served by classical/bicubic baselines only. x16 numbers are not quoted (not evaluated in this work).

## 6. Rejected experiments (summary; full entries in EXPERIMENT_LOG.md)

| experiment | outcome | reason |
|---|---|---|
| E4a head-scope curriculum | rejected | clean +51% regression (head cannot be regime-conditional) |
| E5 gradient supervision (λ_g=0.1) | rejected | −0.1% delta, within noise |
| E6 clean-weighted sampling | rejected | −0.5%/+0.4%, within noise |
| E7 6-epoch curriculum (as single checkpoint) | rejected | clean +0.9%; kept as robust slot for E8 |
| spectral GT-graft oracle as bound | inconclusive | does not dominate SGNet; no bound established |

## 7. Oracle diagnostics (non-deployable, for context only)

Per-pixel GT-informed branch selection would bound hybrid headroom at −12% (clean), −47% (holes) RMSE relative to deep-only (60 val samples). These are upper-bound diagnostics, never deployable performance.
