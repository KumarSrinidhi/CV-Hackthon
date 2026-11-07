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

---

# Phase 3 — Hypothesis-driven improvements

## P3-E4a — Robustness curriculum fine-tuning (head scope)

**Experiment proposal (plan.txt §8.1 gate):**

1. **Observed failure:** deep branch loses to bicubic under 10% random holes on the val split.
2. **Evidence:** `results/verify_holes_val150/` — deep RMSE 0.0438 vs bicubic 0.0374 (n=150, hole_ratio 0.1); P1-06: error +41% near holes vs far regions; P1-05: holes-regime headroom.
   *Note:* historical report (robust_val150) said deep 0.0371 ≈ bicubic 0.0374 (tie); this verification measures deep 0.0438 (worse than tie). Discrepancy recorded honestly; both agree deep's holes advantage collapses.
3. **Hypothesized cause:** SGNet was trained on clean bicubic LR only; it has no missing-measurement completion prior, and nearest-filled hole boundaries create artifacts the network reproduces.
4. **Proposed intervention:** fine-tune with a degradation curriculum — training regimes {clean, holes(0.1), noise(σ=0.02), block holes(2), combined} drawn uniformly per step, using `DegradationConfig` so the training operator matches the eval operator. Head scope (only `last_conv`, 62,689 params).
5. **Why appropriate:** smallest controlled intervention targeting the confirmed failure; operator-consistent degradation prevents train/eval mismatch.
6. **Expected metric changes:** holes-regime val RMSE down (target: beat bicubic 0.0374 and original deep 0.0438); clean val RMSE approximately preserved (small risk of regression).
7. **Possible regressions:** clean RMSE/edge F1 degradation; overfit to 0.1 holes at the expense of clean.
8. **Dataset/split:** NYU train bundle (split train), validation on the 150-image val split (30-image subset per regime), test untouched.
9. **Estimated GPU time:** ~15 min.
10. **Memory:** ~4.5 GB peak (patch training).
11. **Success criteria:** mean val RMSE over {clean, holes} improves over pretrained baseline (0.0327 on the 30-image subset); holes RMSE < 0.0374 (bicubic); clean RMSE no worse than pretrained + 5%.
12. **Rejection criteria:** clean regression > 5% or holes improvement < 10% → reject, try scope=all next (E4b).

**Configuration:** `python -m deep.finetune_v2 --root nyu_data/nyu_data --epochs 3 --steps-per-epoch 200 --train-regimes clean,holes,noise,block,combined --val-regimes clean,holes --scope head --lr 1e-5 --schedule constant --clip 1.0 --depth-loss l1 --seed 42 --out runs/e4a_head_curriculum`
Checkpoint provenance: initialized from official `SGNet_X4_R.pth` (`ebab512732b2c629…`); output to `runs/e4a_head_curriculum/` (original checkpoint untouched).

**Results (30-image val subset, metres):**

| epoch | clean | holes | mean |
|---|---|---|---|
| pretrained | 0.01230 | 0.05601 | 0.03415 |
| 1 | 0.01421 | 0.04643 | 0.03032 |
| 2 | 0.01682 | 0.04185 | 0.02933 |
| 3 | 0.01861 | 0.03946 | 0.02903 |

Peak VRAM 3.47 GB; ~0.9 s/step incl. data.

**Decision: REJECT** (pre-registered criterion: clean regression > 5%). Holes improved 30% but clean regressed +51% — head-only adaptation monotonically trades clean accuracy for hole filling; it cannot learn regime-conditional behaviour. Proceed to pre-registered fallback E4b (scope=all, lr 1e-6).

## P3-E4b — Robustness curriculum fine-tuning (full model, low LR)

**Experiment proposal:** identical to P3-E4a except: **scope=all** (36.4 M params trainable), **lr=1e-6** (plan §8.4 range for full fine-tuning), **schedule=cosine**, epochs 3.
- Hypothesis: full-model capacity can learn hole-conditional behaviour (the network can detect nearest-filled hole statistics and respond) without the head-only clean/holes trade-off.
- Success: mean val RMSE improves AND clean regression ≤ 5% AND holes improvement ≥ 10%.
- Rejection: clean regression > 5% (vs pretrained 0.01230) or holes RMSE > 0.0504.

**Configuration:** `python -m deep.finetune_v2 --root nyu_data/nyu_data --epochs 3 --steps-per-epoch 200 --train-regimes clean,holes,noise,block,combined --val-regimes clean,holes --scope all --lr 1e-6 --schedule cosine --clip 1.0 --depth-loss l1 --seed 42 --out runs/e4b_all_curriculum`

**Results (30-image val subset, metres):**

| epoch | clean | holes | mean |
|---|---|---|---|
| pretrained | 0.01230 | 0.05601 | 0.03415 |
| 1 | 0.01266 | 0.04621 | 0.02944 |
| 2 | 0.01288 | 0.04346 | 0.02817 |
| 3 (best) | 0.01289 | 0.04333 | 0.02811 |

Peak VRAM 3.76 GB; ~2.7 s/step (full-model backward).

**Decision: KEEP (provisional, pending full-split confirmation).** Pre-registered success criteria met: mean improved 17.7%; holes −22.6%; clean regression +4.8% (≤ 5% limit, borderline).

**Full val150 confirmation, original vs E4b-best (n=150, `results/{original,e4b}_val150_*/`):**

| regime | bicubic | original deep | E4b deep | Δ vs original |
|---|---|---|---|---|
| clean | 0.0292 (F1 0.211) | 0.0087 (F1 0.959) | 0.0090 (F1 0.954) | +3.4% RMSE |
| holes 10% | 0.0374 (F1 0.231) | 0.0438 (F1 0.817) | **0.0344** (F1 0.855) | −21.5%, now beats bicubic |
| noise σ=0.02 | 0.0345 (F1 0.212) | 0.0312 (F1 0.844) | **0.0227** (F1 0.855) | −27.2% |
| noise σ=0.05 | 0.0529 (F1 0.218) | 0.0741 (F1 0.594, worse than bicubic) | **0.0315** (F1 0.677) | −57.5% |

The confirmed missing-depth/noise weaknesses are substantially fixed; clean cost is +3.4% RMSE (0.0087→0.0090) and −0.005 edge F1. The original's strong-noise performance was *worse than bicubic*; E4b restores a large margin.

**Artifacts:** `runs/e4b_all_curriculum/` (best.pth = epoch 3; `runs/` is gitignored — checkpoint SHA256 recorded in `docs/DEEP_MODEL_IMPROVEMENTS.md`), `results/e4b_val150_{clean,holes,noise,noise_strong}/`, `results/original_val150_{clean,holes,noise,noise_strong}/`, `results/verify_holes_val150/`.

E4b best.pth SHA256: `8413deabbf4e81b9…`

## P3-E5 — Gradient supervision added to the kept curriculum (isolates the loss factor)

**Experiment proposal:** identical to P3-E4b (same data curriculum, scope all, lr 1e-6, cosine, seed, 3 epochs) plus **λ_g=0.1 Sobel-gradient L1** applied to both the main output and the previously unsupervised auxiliary `out_re` head.
- **Observed failure targeted:** boundary error dominates clean RMSE (P1-07: edge regions 12× smooth; correlation 0.945 with edge density); `out_re` unsupervised (verified in code).
- **Hypothesis:** gradient supervision sharpens boundary reconstruction and recovers some of E4b's clean cost.
- **Success criteria (vs E4b on the same 30-image val subset):** clean val RMSE < 0.01289 AND holes val RMSE ≤ 0.04333 × 1.02 (no material robustness loss).
- **Rejection:** clean not improved or holes regression > 2%.
- **Config:** `python -m deep.finetune_v2 --root nyu_data/nyu_data --epochs 3 --steps-per-epoch 200 --train-regimes clean,holes,noise,block,combined --val-regimes clean,holes --scope all --lr 1e-6 --schedule cosine --clip 1.0 --depth-loss l1 --lam-g 0.1 --grad-target both --seed 42 --out runs/e5_grad_curriculum`

**Results (30-image val subset, vs E4b):**

| epoch | clean (E5 / E4b) | holes (E5 / E4b) |
|---|---|---|
| 3 (best) | 0.012878 / 0.012892 | 0.043268 / 0.043327 |

**Decision: REJECT (no meaningful change).** Differences are −0.1% on both regimes — within run-to-run noise on 30 images; gradient supervision at λ_g=0.1 with grad-target both neither helps nor hurts. λ_g=0 kept; no complexity without measurable benefit. (Rejection documented per plan §18.7.)

## P3-E4b-HF — E4b checkpoint on the clean HF 100-sample benchmark

- **Question:** does the kept robust checkpoint preserve clean-benchmark quality (target RMSE < 0.0101, F1 ≥ 0.960)?
- **Method:** `eval_all3_hf.py` with `SGNET_CKPT_X4=runs/e4b_all_curriculum/best.pth`, all six methods, n=100 → `results/e4b_hf_clean100/`.
- **Results:** deep RMSE **0.0104** (original 0.0101), edge F1 **0.952** (original 0.960), boundary RMSE 0.0449 (0.0434), latency 1881 ms; hybrid 0.0301/0.801 (unchanged behaviour — still loses to deep on clean).
- **Decision:** the robustness gain costs −0.0003 RMSE / −0.008 F1 on clean; acceptance target not yet met. Next single factor: clean-weighted training distribution (E6).

## P3-E6 — Clean-weighted curriculum (isolates the training-distribution factor)

**Experiment proposal:** identical to P3-E4b (scope all, lr 1e-6, cosine, 3 epochs, seed 42) except training-regime sampling weights **clean 0.40, holes 0.15, noise 0.15, block 0.15, combined 0.15** (E4b used uniform 0.2 each).
- **Hypothesis:** E4b's clean drift comes from 80% degraded samples; heavier clean exposure retains robustness while recovering clean accuracy.
- **Success criteria (30-image val subset vs E4b):** clean < 0.01289 AND holes ≤ 0.045; then confirm on val150 + HF benchmark.
- **Rejection:** clean ≥ E4b's or holes > 0.045.
- **Config:** `python -m deep.finetune_v2 --root nyu_data/nyu_data --epochs 3 --steps-per-epoch 200 --train-regimes clean,holes,noise,block,combined --regime-weights 0.4,0.15,0.15,0.15,0.15 --val-regimes clean,holes --scope all --lr 1e-6 --schedule cosine --clip 1.0 --depth-loss l1 --seed 42 --out runs/e6_cleanweighted`

**Results (30-image val subset, vs E4b):**

| epoch 3 (best) | E6 | E4b |
|---|---|---|
| clean | 0.012832 | 0.012892 |
| holes | 0.043520 | 0.043327 |

**Decision: REJECT (no meaningful change).** Clean −0.5%, holes +0.4% — both within noise. The clean drift is inherent to fine-tuning on the NYU train bundle, not the regime mix. Uniform sampling kept.

## P3-E7 — Training duration (6 epochs; isolates the duration factor)

**Experiment proposal:** identical to P3-E4b except **epochs 6** (fresh cosine over 6 epochs).
- **Hypothesis:** E4b's best was its final epoch (selection curve not saturated); longer training improves both clean and holes.
- **Success:** clean < 0.01289 AND holes < 0.04333 on the 30-image val subset.
- **Rejection:** either metric not improved.
- **Config:** `python -m deep.finetune_v2 --root nyu_data/nyu_data --epochs 6 --steps-per-epoch 200 --train-regimes clean,holes,noise,block,combined --val-regimes clean,holes --scope all --lr 1e-6 --schedule cosine --clip 1.0 --depth-loss l1 --seed 42 --out runs/e7_6epoch`

**Results (30-image val subset):**

| epoch | clean | holes | mean |
|---|---|---|---|
| pretrained | 0.01230 | 0.05601 | 0.03415 |
| 4 (best) | 0.01301 | 0.04205 | 0.02753 |
| 6 (last) | 0.01300 | 0.04206 | 0.02753 |

**Full val150 confirmation (E7-best):** holes **0.0333** (E4b: 0.0344), noise σ=0.02 **0.0219** (E4b: 0.0227).

**Decision: REJECT as a single-checkpoint replacement** (pre-registered criterion requires BOTH metrics improved; clean regressed +0.9%) but **KEEP as the robust-slot checkpoint** for E8: it is the best holes/noise performer measured, and clean quality under E8 is served by the original checkpoint. E7 best.pth SHA256: (recorded in docs/DEEP_MODEL_IMPROVEMENTS.md).

## P3-E8 — Regime-aware checkpoint selection (deployable, LR-only detection)

**Experiment proposal:**
- **Observed failures:** (i) E4b/E7 robustness gains are statistically significant but cost a significant clean regression (Phase 4 stats: HF clean RMSE +0.00026, CI [+0.00020,+0.00032], 99/100 scenes worse); (ii) B2/B3: the regime is inferable from the LR input (holes = invalid LR pixels; noise = local high-frequency energy).
- **Intervention:** `deep/adaptive.py::deep_hr_depth_adaptive` — detect regime from LR only: holes → robust checkpoint (E7 best); median |Laplacian| of valid LR > threshold → noise → robust checkpoint; else → original checkpoint. Threshold calibrated on val only.
- **Calibration (50 val images):** clean Laplacian score ∈ [0.0077, 0.0245]; noise σ=0.02 ∈ [0.0640, 0.0818]; σ=0.05 ∈ [0.153, 0.179]. Classes perfectly separated; threshold = 4e-2 (geometric midpoint of the clean/noise gap).
- **Success criteria:** HF clean benchmark equals original deep within noise (RMSE ≤ 0.0101 + 2σ, F1 ≥ 0.958); val150 holes/noise match or beat E7 numbers.
- **Rejection:** any regime misrouted in validation, or clean benchmark not preserved.
- **Config:** `python scripts/eval_adaptive.py --tars datasets/val-000000.tar --out results/e8_hf_clean100` and `--nyu-val --out results/e8_val150`.

**Results:**

- **HF clean benchmark (n=645 — the run evaluated the full val tar, giving stronger statistics than the historical n=100):** deep_adaptive RMSE 0.0102 / edge F1 0.968 — **identical to the original deep** (0.0102/0.968); every HF sample was routed to the clean checkpoint. (On the historical 100-sample subset this equals RMSE 0.0101 / F1 0.960.)
- **val150 regimes (deep_adaptive vs original deep):**

| regime | bicubic | original deep | deep_adaptive | Δ |
|---|---|---|---|---|
| clean | 0.0292 | 0.0087 | 0.0087 | exact match |
| holes 10% | 0.0374 | 0.0438 | **0.0333** | −24%, beats bicubic |
| noise 0.02 | 0.0345 | 0.0312 | **0.0219** | −30% |
| noise 0.05 | 0.0529 | 0.0741 (worse than bicubic) | **0.0293** | −60% |

**Decision: KEEP.** E8 = original-clean quality + E7-robustness with LR-only detection; no regime was misrouted (clean outputs bit-identical). Acceptance target on clean is preserved (0.0101/0.960 on the historical 100-subset protocol) while degraded regimes improve 24–60%.

**Artifacts:** `deep/adaptive.py`, `scripts/eval_adaptive.py`, `results/e8_hf_clean100/`, `results/e8_val150_{clean,holes,noise,noise_strong}/`.
