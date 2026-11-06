# Experiment Log — P4 Deep Branch Improvement

Registry format per plan.txt §4. One entry per experiment. Verdicts: CONFIRMED / NOT A PROBLEM / UNCERTAIN for probes; decisions: keep / reject / repeat / inconclusive for experiments.

Entry fields: Experiment ID, Date/time, Research question, Hypothesis, Confirmed failure under investigation, Exact code change, Configuration and hyperparameters, Dataset and split, Sample identifiers, Checkpoint + SHA256, Random seed, Software/hardware environment, Metrics, Per-image results, Runtime, Peak GPU memory, Visual observations, Statistical analysis, Limitations, Final decision.

---

## P0-01 — Repository, environment and checkpoint verification (Phase 0, read-only)

- **Date/time:** 2026-11-06 ~21:50 local
- **Research question:** Do the reported baseline facts (plan.txt §2) match the actual repository?
- **Method:** Direct inspection and probing; no code modified.
- **Environment recorded:** Windows, Python 3.12.14, torch 2.11.0+cu128, CUDA available, NVIDIA GeForce RTX 5070 Laptop GPU, 7.96 GB VRAM.
- **Git state:** branch `hybrid`, HEAD `d5c4cab` ("Add 3-track comparison..."), remote `origin https://github.com/KumarSrinidhi/CV-Hackthon.git`. Untracked: `.python-version`, `docs/HYBRID_AUDIT_REPORT.md`, `nyu_data/`, `plan.txt`, `src/`, `tests/test_data.py`, `tests/test_metrics.py`, `tests/test_skeleton.py` (pre-existing user work — untouched).
- **Checkpoints (SHA256, first 16 hex):**
  - `checkpoints/sgnet/SGNet_X4_R.pth` — 146,859,807 B — `ebab512732b2c629` ✔ matches historical record `ebab512732b2...`
  - `checkpoints/sgnet/SGNet_X8_R.pth` — 160,727,455 B — `f650dd2f107ad665`
  - `checkpoints/sgnet/SGNet_X16_R.pth` — 347,508,191 B — `ceda2db815c9c001`
- **Test baseline:** `pytest tests/ -q` → **106 passed, 0 skipped** (11.55 s). Note: `docs/GPU_RUN_REPORT.md` says "35 passed" — stale; docs also say "90 passed, 1 skip". Actual current suite: 106.
- **Discrepancies found (plan.txt §19 reporting duty):**
  1. HF val tar location is `datasets/val-000000.tar`; plan.txt §2.7 reports `datasets/ps4/val-000000.tar`. Content unaffected (same file found via glob).
  2. Stale test counts in docs (above).
  3. `checkpoints/` directory: `deep/settings.py` default resolves to `<repo>/checkpoints/sgnet` — verified present, no env override needed.
- **Decision:** proceed to Phase 0 baseline reproduction.

---

## P0-02 — Reproduction of the historical baseline evaluation (Phase 0)

- **Date/time:** 2026-11-06, started ~21:50 local
- **Research question:** Can the reported deep-branch baseline (RMSE 0.0101 m, edge F1 0.960, ~1899 ms/img) be reproduced with the exact historical command?
- **Command:** `python scripts/eval_all3_hf.py --tars datasets/val-000000.tar --scale 4 --max-samples 100 --out results/baseline_deep_repro`
- **Dataset/split:** first 100 `.h5` members of `datasets/val-000000.tar` (HF FastDepth NYU val), clean bicubic DegradationConfig, x4.
- **Methods:** nearest, bilinear, bicubic, classical, deep, hybrid (all six).
- **Checkpoint:** `checkpoints/sgnet/SGNet_X4_R.pth`, SHA256 `ebab512732b2c629…`
- **Seed:** degradation default seed 0 (deterministic per-sample).
- **Results (n=100, unit=m, data_range=10.0, border=6px, clean bicubic, seed 0):**

| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| nearest | 0.0535 | 0.0162 | 48.27 | 0.9890 | 0.2465 | 0.739 | 2.7 |
| bilinear | 0.0415 | 0.0132 | 50.53 | 0.9934 | 0.1943 | 0.205 | 2.7 |
| bicubic | 0.0367 | 0.0121 | 51.62 | 0.9942 | 0.1680 | 0.259 | 2.7 |
| classical | 0.0584 | 0.0191 | 47.59 | 0.9874 | 0.2591 | 0.671 | 458.9 |
| **deep** | **0.0101** | **0.0030** | **62.64** | **0.9994** | **0.0434** | **0.960** | **1900.1** |
| hybrid | 0.0298 | 0.0077 | 53.85 | 0.9966 | 0.1389 | 0.804 | 2455.0 |

- **Comparison to historical (GPU_RUN_REPORT.md §5):** deep RMSE 0.0101 vs 0.0101 ✔; edge F1 0.960 vs 0.960 ✔; latency 1900.1 vs 1899 ms ✔; bicubic 0.0367/0.259 ✔; classical 0.0584/0.671 ✔; hybrid 0.0298/0.804 ✔. **Exact reproduction — no discrepancy to investigate.**
- **Artifacts:** `results/baseline_deep_repro/` (summary.md/json + per_image.csv). Historical directories untouched.
- **Decision:** keep. Phase 0 exit gate met (reproducible baseline + verified checkpoint `ebab512732b2c629…` + green suite 106 passed). Proceed to Phase 1.

---

## P1-01 — Probe A1: degradation-operator frequency analysis (Phase 1)

- **Date/time:** 2026-11-06 ~22:10
- **Research question:** What information does the PIL-bicubic x4 operator suppress, preserve or alias?
- **Method:** `scripts/forensic_probes.py a1` — impulse response, exactly-periodic sinusoids (k/512, 512 px), step edge, radial power-spectrum ratio on 5 real scenes.
- **Environment:** Python 3.12.14, numpy/PIL; CPU only. Checkpoint: n/a.
- **Results:** passband gain 1.00 (≤0.031 cyc/px) → 0.94 (0.063) → 0.77 (0.094) → 0.64 (0.109) → 0.91 (Nyquist bin); aliased band 0.23 (0.156); ~0 (0.25). Impulse support 4×4 LR px, Σw = 1/16; DC exact; step overshoot 5%, transition 4 px. Real-scene radial ratio 0.53–0.65 across 0.004–0.20 cyc/px.
- **Statistical analysis:** deterministic synthetic inputs; real-scene values are medians over 5 scenes (coarse, scene-mixed diagnostic).
- **Limitations:** radial ratio mixes operator attenuation with scene spectra; single-image FFT ratios deliberately not used.
- **Decision/verdict:** CONFIRMED characterization — operator is band-limiting above ~0.06 cyc/px with mild aliasing. No information-theoretic limit claimed.

## P1-02 — Probe A2: depth-noise physics (Phase 1)

- **Date/time:** 2026-11-06 ~22:25
- **Research question:** Does the noise model represent the intended degradation conditions?
- **Method:** `scripts/forensic_probes.py a2` — absolute (σ=0.02 m), relative (2%), quadratic Kinect-v1-style (σ=0.0015·z²) noise on 10 real scenes → bicubic-up RMSE per 0.5 m depth bin.
- **Results:** all models' error grows with depth; at 9 m: abs 0.255, rel 0.308, quad 0.276. Absolute slightly under-covers far range.
- **Limitations:** no sensor-verified noise data available; constants chosen to give comparable mid-range severity.
- **Decision:** UNCERTAIN physically; keep absolute σ for the curriculum (matches eval protocol exactly); document caveat.

## P1-03 — Probe A3: sanity checks, round trip, oracles (Phase 1)

- **Date/time:** 2026-11-06 ~22:35
- **Research question:** Is the pipeline exact, and how much headroom do GT-informed oracles indicate?
- **Method:** `scripts/forensic_probes.py a3` — 25 samples; identity check; bicubic-up vs SGNet round trip; GT-range normalization oracle; spectral-graft oracle (σ=0.75 px lowpass at LR Nyquist).
- **Checkpoints:** `SGNet_X4_R.pth` (`ebab512732b2c629…`). GPU: RTX 5070.
- **Results:** identity exact (fill round-trip ≤9.5e-7); round trip: bicubic 0.0449/F1 0.377 vs SGNet 0.0120/F1 0.971; GT-norm oracle 0.012004 (≈ no change); spectral oracle 0.0293 (worse than SGNet → rejected as a bound).
- **Decision:** plumbing NOT A PROBLEM; normalization NOT A PROBLEM; spectral bound UNCERTAIN/rejected. No floor claim.

## P1-04 — Probe B1: metric correctness audit (Phase 1)

- **Date/time:** 2026-11-06 ~22:45
- **Research question:** Are the official metrics correct and how threshold-sensitive is edge F1?
- **Method:** `scripts/forensic_probes.py b1` — independent metric reimplementation vs `eval/metrics.py` on 3 scenes; rel_thresh {0.01–0.05} × tol {0,1,2} sweep; degenerate inputs.
- **Results:** max abs diff 0.0 across all metrics/scenes; degenerate cases handled safely (NaN edge F1 for empty edge mask, NaN preds excluded); edge F1 varies 0.79–0.98 across thresholds.
- **Decision:** NOT A PROBLEM (implementation exact). Historical convention (0.3 m, tol 1) preserved for all comparisons.

## P1-05 — Probe B2: fusion-equation audit (Phase 1)

- **Date/time:** 2026-11-06 ~22:55
- **Research question:** Does the current confidence equation help or hurt, and how much fusion headroom exists?
- **Method:** `scripts/forensic_probes.py b2` — 25 samples × {clean, holes 10%}; classical/deep/fused RMSE; per-pixel worse-fractions; wc-scale sweep {0–2}; GT-informed per-pixel oracle selector (diagnostic only); consistency residual check.
- **Results:** clean: deep 0.0119 / classical 0.0705 / fused 0.0357 (69.6% of pixels worse than deep); holes: deep 0.0671 / classical 0.2066 / fused 0.0924. RMSE monotonic in classical weight. Oracle selector: 0.0102 clean, 0.0287 holes. Consistency residual 0.0049→0.0017.
- **Decision:** CONFIRMED — fusion as configured damages results in both regimes; headroom concentrated in degraded regimes. Address in Phase 8 with validation-calibrated regime-aware weights.

## P1-06 — Probe B3: inference numerical audit (Phase 1)

- **Date/time:** 2026-11-06 ~23:00
- **Research question:** Are inference-time numerical choices (normalization, filling, tiling, precision) limiting quality?
- **Method:** `scripts/forensic_probes.py b3` — 25 samples × 3 normalization variants; 5 samples × holes 10% with near/far error split; tiling (tile_lr=32) on 2 samples; bf16 autocast attempt.
- **Results:** normalization variants within 0.4%; near-hole RMSE 0.0212 vs far 0.0151 (+41%); tiling seam max diff 0.477 m (aggregate −7e-5); bf16 rejected by cuFFT path (`Unsupported dtype BFloat16`).
- **Decision:** normalization NOT A PROBLEM; hole-localized error CONFIRMED (drives E4); tiling seam CONFIRMED as OOM-only edge case (documented); fp32 required CONFIRMED.

## P1-07 — Probe C: per-scene/per-region error analysis + RGB guidance (Phase 1)

- **Date/time:** 2026-11-06 ~22:00–22:30
- **Research question:** Where does the deep branch fail, and does RGB guidance help or hurt?
- **Method:** `scripts/error_analysis.py` — 100 samples × {bicubic, classical, deep, hybrid}; per-image metrics; 8 region masks; rankings; Spearman correlations; visual atlas (6 worst scenes); RGB probe (original/grey/blurred/shuffled/other-sample) on 25 samples.
- **Checkpoints:** `SGNet_X4_R.pth`. GPU: RTX 5070; ~20 min total.
- **Results (key):** Spearman(RMSE, edge density)=0.945, (RMSE, depth range)=0.921; edge-band RMSE 0.0217 vs smooth 0.0054; on-edge 0.0668 (12× smooth); thin 0.0345; weak-RGB edges 0.1086 > RGB-aligned 0.0961; texture/flat-depth 0.0068 (1.26× smooth). Deep beats bicubic on all 100 scenes. Hybrid regressions concentrate on deep's hardest scenes. RGB probe: removing/degrading RGB collapses edge recall 0.957→0.77–0.87 while precision stays ≥0.96.
- **Artifacts:** `results/error_analysis/` (per_image_metrics.csv, per_region_rmse.csv, region_summary.csv, rankings.json, atlas_*.png, rgb_probe/).
- **Limitations:** region definitions are heuristic proxies; RGB probe subject to input-distribution shift (caveated).
- **Decision:** CONFIRMED — boundary error dominates clean RMSE; RGB guidance genuinely helps edge recall; texture-copying mild. Feeds E1/E2 (boundary) and Phase 8 (fusion).

## P1-08 — Phase 2 (early): split overlap audit

- **Date/time:** 2026-11-06 ~22:35
- **Research question:** Do the 100-sample HF benchmark and the 150-image NYU val split overlap?
- **Method:** `tmp/debugging/split_overlap_check.py` — name matching + shape/depth-signature (percentile) matching between the two populations.
- **Results:** 100 HF samples vs 150 NYU val: 0 name matches, 0 signature collisions → disjoint populations.
- **Artifacts:** `tmp/forensics/split_manifest.json`.
- **Decision:** validation-based selection does not touch the benchmark; documented for §17.3.
