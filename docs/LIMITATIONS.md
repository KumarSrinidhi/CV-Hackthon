# Limitations

Updated 2026-11-07 after the deep-branch improvement session (`docs/EXPERIMENT_LOG.md`, `docs/PERFORMANCE_ANALYSIS.md`).

## Deep branch — current state

- **Executed on GPU (RTX 5070)** since 2026-11-06: the "not locally executed" limitation above is historical. Baseline reproduced exactly; improvements measured.
- **Scales:** official checkpoints exist for **x4, x8, x16** only; **x2 is classical-only** (no official SGNet x2 weights; documented, not trained in this work). Adaptive (regime-aware) switching is implemented for **x4 only**.
- **Regime detection** covers random/block holes (exact, from the LR validity mask) and absolute Gaussian noise ≥ 0.02 m (median |Laplacian| threshold, val-calibrated). Real-sensor artefacts beyond these (correlated noise, structured dropouts with valid masks present, rolling shutter) are **not detected** and fall through to the clean checkpoint.
- **Robust checkpoint** (`runs/e7_6epoch/best.pth`, sha256 `9ed571170c55a3a9…`) was fine-tuned on the NYU SGNet train bundle only; cross-dataset transfer (Middlebury disparity) is **untested**.
- **Boundary error remains edge-density-dominated** (Spearman 0.945): gradient supervision and multiscale losses produced no measurable gain (E5) and were rejected; worst scenes combine high edge density with large depth range.
- **Tiled OOM fallback** produces seam errors up to ~0.48 m at tile boundaries (aggregate RMSE impact negligible); never triggered by the benchmark, but avoid forcing `tile_lr`.
- **FP32 only** — the cuFFT path rejects bf16 (verified: `Unsupported dtype BFloat16`).
- **No irreducible-floor claim:** operator frequency analysis and GT-informed oracles (non-deployable) inform headroom discussion only; the true floor is unknown.

## Hybrid — current state

- **v2 fusion** (default) replaces the legacy evidence weights, which measurably damaged the deep branch (3× clean RMSE). v2 = adaptive deep + 0.05 classical weight under holes + measurement consistency. The legacy path remains available (`fusion="legacy"`).
- The classical branch's contribution is now small by measurement: deep dominates on clean and noise; classical adds ~1.3% under holes. A per-pixel oracle (diagnostic only) shows larger theoretical headroom (−47% under holes) that no deployable confidence design realized in this work.
- Confidence weights remain **normalized heuristics, not calibrated probabilities**.
- Measurement consistency is only exact under the known bicubic degradation operator; real sensor formation is not claimed to be invertible.
- The classical JBU per-valid-sample Python loop is still the latency hotspot for large images (~300 ms @ 480×640 ×4).

## Datasets / deployment

- NYU Depth v2 and Middlebury are the intended benchmarks; **KITTI compatibility is not verified**.
- Datasets and checkpoints are not committed (see `.gitignore`); the robust checkpoint's SHA256 is recorded in `docs/DEEP_MODEL_IMPROVEMENTS.md` for provenance.
- No live hosted URL; `eval_P4_T<teamID>.md` still pending Team ID/USN.
- Gradio UI written but not launched in this environment.
