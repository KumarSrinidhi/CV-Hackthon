# Implementation and Experiment Report

Branch: `feat/research-driven-depth-completion`
Scope: understand the repository, study six reference research repositories, improve the
classical / deep / hybrid depth-completion approaches, establish the evaluation
methodology, and package everything for GPU experiments on Kaggle.

Honesty rules applied throughout: **no fabricated numbers**, **no local GPU training**,
**no full-dataset runs**. Every number in §7 is a clearly-labelled CPU smoke run with
sample counts in the 2–4 range; the only trustworthy quality numbers remain those in
`docs/GPU_RUN_REPORT.md` (teammate RTX 5070) and the not-yet-run Kaggle notebook.

---

## 1. Execution order (as required)

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Understand all branches + PDF task statement → `PROJECT_UNDERSTANDING.md` | done (`8beeab6`), corrected after the merge |
| 2 | Clone 6 reference repos into `tmp/` → `RESEARCH_REFERENCE_ANALYSIS.md` | done (`8beeab6`) |
| 3 | Improve code: merge `origin/hybrid`, resolve conflicts | done (`4c9fb84`) |
| 4 | Build `KAGGLE_DEPTH_COMPLETION_EXPERIMENTS.ipynb` | done (`73847d7`), smoke-tested (§7) |
| 5 | Stash-based hole-band port + tests | done (`43264c5`) |
| 6 | CPU smoke tests only | done (§7) |
| 7 | This report | done |

Commits on the branch (oldest first):

```
4c9fb84 Merge origin/hybrid: unified hybrid pipeline with graceful deep fallback
8beeab6 docs: add project understanding and reference-research analysis
43264c5 feat: hole-band classical confidence around missing LR measurements
3bcd049 fix: pre-check deep method for missing checkpoint like hybrids
73847d7 feat: add Kaggle depth-completion experiments notebook
```

## 2. Merge of `origin/hybrid` — decisions

The merge resolved nine conflicted files and deleted three dead `ps4` test files.
Key decisions, all preserved in the merged tree:

1. **Deep-unavailable contract = graceful degrade** (not `raise`). `hybrid_hr_depth`
   returns `(depth, available)`; a missing checkpoint yields the classical output with
   `deep_available=False` and a `degrade_note`. Evaluation honesty is enforced
   *upstream* by a pre-check (`_needs_deep_checkpoint` + `resolve_checkpoint`) so a
   degraded classical result is never scored under a hybrid/deep name.
2. **Classical wrapper** = `origin/hybrid`'s `classical_hr_depth`, which delegates to
   `reconstruct(..., method="classical")`. HEAD's inline A1→A4 wrapper was dropped
   (single source of truth).
3. **Frozen config defaults**: `classical_alpha=0.02`, `consistency_strength=0.25`,
   `max_classical_weight=0.35`, `use_consistency=True`, `consistency_iterations=1`,
   `use_boundary_refinement=False`.
4. **Unified `hybrid_hr_depth` branch order**: degrade check → fixed-alpha path
   (bypasses v2) → `fusion=="v2"` regime-aware path → legacy confidence path.
   `regime == "noise"` disables consistency (`consistency_on = use_consistency and
   regime != "noise"`).
5. **Intermediates/telemetry keys** kept from both sides (`classical`, `deep`,
   `deep_available`, `confidence_*`, `fused`, `fused_before_consistency`,
   `classical_weight`, refinement/consistency flags, `fusion`, `regime`).
6. **Docs rewritten** to the graceful-degrade contract (no stale `raise` wording);
   `PROJECT_UNDERSTANDING.md` §4/§5/tree annotations corrected to match the merge.

## 3. Hole-band port (stash@{0}) — decisions

`stash@{0}` ("prework" on `feat/combined-hybrid`) was **partially ported**:

- **Ported**: the *hole-band concept* — inside `estimate_confidence(..., adaptive=True)`,
  dilate the invalid-LR region 3×3, upsample it to HR (INTER_NEAREST), and force the
  classical confidence to at least `hole_classical_weight * hole_zone *
  classical_supported` (default `hole_classical_weight=0.5`), plumbed through
  `hybrid_hr_depth` → `estimate_confidence`.
- **Not ported (deliberate)**: the stash's wholesale confidence-model replacement —
  it was built on a different evidence base and would have discarded the merge
  resolution. Fusion invalid-branch handling was already present via auto-merge.
- **Tests added** in `tests/test_hybrid.py`: invalid-branch fuse (`both invalid → 0`),
  classical preference near holes, hole weight requires classical support, no-hole
  no-boost, and an end-to-end hole-band pipeline test.
- **Docs**: hole-band paragraph in `docs/HYBRID_FUSION.md`, signature updated in
  `docs/API_REFERENCE.md`.
- `stash@{1}` (hybrid-v2 WIP) intentionally untouched.

## 4. Research-driven adoption status (A1–A12)

Full analysis: `RESEARCH_REFERENCE_ANALYSIS.md`. Post-implementation status:

| # | Idea (reference) | Status | Where / why not |
|---|---|---|---|
| A1 | Per-pixel softmax-competitive confidence (Sparse-Depth-Completion, PENet) | **pre-existing, verified** | `hybrid/confidence.py`, `hybrid/fusion.py` already fuse per-pixel |
| A2 | Hole-band classical boost + hard overwrite at measured pixels (stash, SpAgNet) | **partially adopted** | band boost implemented (`43264c5`); hard `C=1` overwrite not ported |
| A3 | Mask-normalised deep input (CFCNet `conv(d,1)/conv(mask,1)` + mask channel) | not adopted | would invalidate existing checkpoint distribution; needs retraining (Kaggle) |
| A4 | Never average zeros; nearest-valid downsample, nearest mask resampling (PENet, sparse-to-dense) | **pre-existing** | `data/degradation.py:99-101` nearest-filled downsample, `upsample_mask_nearest` |
| A5 | Clamp predictions to `[0, max_depth]` before scoring (Sparse-Depth-Completion) | not adopted | `eval/metrics.py` scores unclamped preds (≥0 clamp only in baselines); candidate follow-up |
| A6 | berHu loss, NaN/0-aware masking (sparse-to-dense) | not adopted | no training this session; applies to future finetune |
| A7 | Density-jittered training, edge-biased holes, Bernoulli thinning (CFCNet, sparse-to-dense) | not adopted | training-side; synthetic hole/noise regimes already exist for evaluation |
| A8 | CSPN-style learned guided propagation (PENet) | not adopted | new module + training; would need its own validation cycle |
| A9 | Per-pixel measurement re-injection with reliability gate (PENet) | not adopted | existing consistency step is global-strength (`consistency_strength`) |
| A10 | Eval harness discipline: per-image metrics, sync latency, density sweep (sparse-to-dense, SpAgNet) | **mostly pre-existing** | batch-1 per-image metrics + `torch.cuda.synchronize()` latency already in `eval/evaluate.py`; **density sweep (`patterns/*.npy`) not run** |
| A11 | iMAE / iRMSE inverse-depth metrics | not adopted | not in `METRIC_KEYS`; cheap follow-up if the rubric wants it |
| A12 | Own stash@{0} hole-band fusion + tests | **adopted (scoped)** | see §3 |

## 5. Evaluation methodology and honesty fixes

Protocol (unchanged, `docs/EVALUATION_METHODOLOGY.md`): masked metrics over valid GT
pixels, border 6 px, fixed `data_range=10 m`, depth-only edge extraction, unavailable
methods reported as *not evaluated* with the reason — never dropped, never relabelled.

Fix in this branch (`3bcd049`): `_needs_deep_checkpoint` now also covers the pure
`deep` method. Before, a missing checkpoint made `deep` fail per-sample and surface as
the noisy "all samples failed" row; now it reports the same clean
`DeepModelUnavailableError: No SGNet x4 checkpoint found …` reason as the hybrids.
The guard lives in the shared function, so the notebook, `scripts/eval_all3_hf.py`,
and every CLI path get it at once.

## 6. Kaggle notebook (`KAGGLE_DEPTH_COMPLETION_EXPERIMENTS.ipynb`)

Thin orchestration over the existing harness — no re-implemented metrics. Cells:

1. **Header** — contents + honesty protocol (quote as "Kaggle run, `<date>`, `<GPU>`";
   not interchangeable with `docs/GPU_RUN_REPORT.md` or local smoke tests).
2. **Environment** — python/torch/CUDA/GPU report (`torch.cuda.synchronize` for timing).
3. **Clone + install** — clones `KumarSrinidhi/CV-Hackthon`, installs
   `requirements.txt` **except torch/torchvision** (Kaggle ships a CUDA build) plus
   `h5py`, `huggingface_hub`.
4. **Checkpoint status** — optional `CKPT_DIR` → `SGNET_CKPT_DIR`; prints
   `resolve_checkpoint` for ×2/×4/×8/×16. Without checkpoints, `deep`/`hybrid*` are
   *not evaluated* (honest), baselines + classical always run.
5. **Data A** — locate `val-*.tar` under `/kaggle/input`, else download
   `sayakpaul/nyu_depth_v2` from the HF hub (Internet toggle noted).
6. **Experiment A** — HF NYU val100 ×4 clean via `scripts/eval_all3_hf.py`
   (primary; same protocol as local `val100` baselines).
7. **Experiment B** — ×2 and ×8, 50 samples (×2 deep/hybrid expected *not evaluated*:
   SGNet has no ×2 scale).
8. **Experiment C** — ×4 degraded (`noise_std=0.02`, `hole_ratio=0.10`), 50 samples.
9. **Data B** — extracts the official 20-sample testset from branch `kumarbro`
   (`git fetch origin kumarbro && git checkout FETCH_HEAD -- data/depthsr_testset`);
   native block-mean LR + sensor holes, **not** re-degraded
   (`operator_exact=False`).
10. **Experiment D** — `DepthSRTestset` subclass serving native `lr_depth_*.npy` /
    `lr_mask_*.npy`, ×2/×4/×8, with `hybrid` injected as `_hybrid_testset`
    (consistency off — the native operator is block-mean, not bicubic — and it
    raises `NotImplementedError` without a checkpoint so the harness reports
    *not evaluated* instead of mislabelled classical output).
11. **Results** — re-prints every `results/kaggle_*/*summary.md`; outputs are
    per-image CSV + summary JSON/MD per experiment, collected via Kaggle
    "Download output".

Usage: open in Kaggle, GPU runtime, Internet on (or attach tar/checkpoints as
datasets), run top to bottom. README gained a matching "### Kaggle" section.

## 7. Verification — CPU smoke only (this machine has no usable GPU)

**Not a quality evaluation.** Sample counts are 2–4, runs are CPU, checkpoints are
absent. Numbers exist only to prove the plumbing executes end-to-end; do not quote
them as results.

| Check | Command / cell | Result |
|---|---|---|
| Full test suite | `python -m pytest tests/ -q` | **145 passed, 1 skipped (CUDA)** in 90 s |
| Checkpoint cell | notebook cell 3 | ×4/×8/×16 → `DeepModelUnavailableError`, ×2 → `UnsupportedScaleError` (expected) |
| Experiment D (official testset ×4, n=4) | smoke script mirroring cells 8–10 | nearest 0.0750 / bilinear 0.0622 / bicubic 0.0662 / **classical 0.0604 RMSE**, edge F1 0.782; `deep`/`hybrid*` **not evaluated** with checkpoint reasons; native LR shapes (128×128, bool mask) verified |
| Experiment C (HF ×4 noise, n=3) | smoke script mirroring cell 7 | bicubic 0.0301 / classical 0.0389 RMSE; `deep` **not evaluated** (clean reason after `3bcd049`) |
| Experiment A (`all3_main`, ×4, n=2) | smoke script mirroring cell 5 | nearest 0.0204 / bilinear 0.0155 / bicubic 0.0136 / classical 0.0148; `deep`/`hybrid` **not evaluated**; all three result files written |
| App path | `reconstruct(rgb, lr, None, 4)` (earlier) | finite (16,16) float32, `deep_available=False`, degrade note present |

Reproduce the smoke: `python tmp/smoke_kaggle_paths.py` (testset must be extracted
first: `git archive -o tmp/testset.tar origin/kumarbro data/depthsr_testset && tar -xf tmp/testset.tar -C tmp`).

**Real numbers** (GPU, larger n) remain those in `docs/GPU_RUN_REPORT.md` — untouched,
never re-labelled by anything in this branch.

## 8. Known gaps / not done

- **No SGNet checkpoints locally** (`SGNet_X4_R.pth` etc. absent) → deep/hybrid
  quality is unverifiable here; the Kaggle notebook with attached checkpoints is the
  intended vehicle. Local deep smoke remains `python -m deep.inference --random-init`.
- **The Kaggle notebook has not been executed on Kaggle** — this machine has no
  Kaggle/GPU. All four experiment paths were smoke-tested locally instead (§7).
- **`eval_P4_<TeamID>.md` not produced** — requires the team USNs and deadline, which
  are unknown (PDF extraction had no USN list).
- **A3/A6/A7/A8/A9/A11 not implemented** (see §4) — training-side or new modules
  requiring their own validation cycles; A5 clamp and A11 inverse-depth metrics are
  cheap follow-ups if the rubric rewards them.
- **A10 density sweep not run** (`patterns/*.npy` masks).
- **`stash@{1}`** (denoise + default-method switch) left in place, unmerged.
- **`feat/classical` extras** (`tuned_jbu`, `masked_jbu`) not merged — out of scope
  for this branch's decisions.
- **Official testset is not committed** (dataset, not code); it is fetched on demand
  from `origin/kumarbro` by the notebook and by the smoke instructions.

## 9. Commands reference

```bash
# tests (CPU)
python -m pytest tests/ -q

# smoke the notebook's experiment paths (after extracting the testset, see §7)
python tmp/smoke_kaggle_paths.py

# HF NYU full method matrix (same engine the notebook uses)
python scripts/eval_all3_hf.py --tars datasets/val-000000.tar --max-samples 100 --scale 4 \
    --out results/kaggle_hf_x4_clean100

# local canonical benchmark (needs checkpoints for deep/hybrid)
python -m eval.benchmark --dataset nyu --split val --scales 2 4 8 \
    --methods nearest bilinear bicubic classical deep hybrid_fixed02 hybrid_adaptive hybrid_selected_clean \
    --out results/benchmark_val --runs 10
```
