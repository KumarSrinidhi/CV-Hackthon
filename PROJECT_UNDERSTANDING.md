# PROJECT_UNDERSTANDING — P4 Edge-Preserving Depth Map Upsampling & Refinement

> Phase-1 deliverable. Written from direct inspection of the repository on
> `master` @ `3cc1540` plus the unmerged branches (`feat/classical`,
> `origin/hybrid`, `feat/combined-hybrid`, `hybrid-v2`, `origin/kumarbro`) and the
> official problem statement (`Computer Vision Hackathon 2026.pdf`, P4).
> Items that could not be verified from the available evidence are marked
> **UNKNOWN** rather than guessed.

---

## 1. Problem statement and objectives

From the official instruction sheet (P4, "Edge-Preserving Depth Map Upsampling and
Refinement"):

- **Mandatory requirements:** three separately callable approaches — **classical**,
  **deep learning**, **hybrid** — for converting sparse / low-resolution depth into
  dense high-resolution depth while preserving object boundaries and depth
  discontinuities.
- **Advanced requirements:** support **2×, 4×, 8×** scale factors; handle **missing
  depth values** (sensor holes); use **RGB guidance**; provide **side-by-side
  comparison** of approaches; work on **unseen depth maps**; **preserve depth
  boundaries** (poor boundary performance loses points).
- **Evaluation metrics:** RMSE, MAE, PSNR, SSIM, edge/boundary preservation,
  performance across scale factors, **latency and memory**.
- **Datasets named in the sheet:** NYU Depth V2, KITTI, Middlebury Stereo.
- **Deliverables:** live hosted URL; GitHub repo with source, architecture diagram,
  design doc, comparative evaluation file (`eval_P4_<TeamID>.md`, first line =
> comma-separated USNs), ablation study and failure-case analysis.
- **Rubric:** Evaluation Metrics 25, Demo/viva 25, Implementation 20, UI/UX and
  deployment 15, Project documentation 15.

**Team / USNs, submission deadline, live-URL target: UNKNOWN** (not present in the
repository).

---

## 2. Datasets and data contracts

### Frozen input/output contract (`data/preprocessing.py`, restated in `README.md`)

| Field | Contract |
|---|---|
| `hr_rgb` | `(H, W, 3)` RGB, uint8 or float32 in `[0, 1]` |
| `lr_depth` | `(h, w)` float32, dataset units (metres for NYU) |
| `valid_mask` | `(h, w)` bool at **LR** resolution; `None` ⇒ derived from depth |
| output | `(H, W)` float32, same unit as `lr_depth` |
| invalid depth | `<= 0`, `NaN`, `Inf` are never treated as measurements |

All three public wrappers share this signature:
`*_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32`.

### Datasets

- **NYU Depth V2** — primary benchmark. Dense GT, depth degraded synthetically
  (bicubic downsample + optional noise/holes) to produce LR inputs.
  `datasets/val-000000.tar` (1 GB, HuggingFace val pack) is present locally;
  `datasets/` is gitignored.
- **Middlebury Stereo** — secondary benchmark (8-bit disparity); results under
  `results/middlebury_raw_x4/`.
- **KITTI** — listed in README for compatibility, explicitly **not verified** in
  this work. Status: **not used**.
- **`data/depthsr_testset/` (only on `origin/kumarbro`, NOT in the master working
  tree)** — 20-sample synthetic manual test set (512×512): `rgb.png`,
  `depth_hr.npy`, `lr_depth_{2,4,8}x.npy`, `lr_mask_{2,4}x…`, `meta.json`,
  plus `index.csv`. Samples 00–09 indoor (≈85 % valid LR), 10–19 geometric
  primitives (≈35–45 % valid, sharp discontinuities). Documented as *functional /
  sanity testing*, not benchmarking. This is the repository's only **real
  "unseen input" set** reachable from git and is a natural held-out set for the
  Kaggle notebook.
- Degradation operators live in `data/degradation.py` (bicubic downsample,
  Gaussian noise, random holes), applied deterministically with a seed.

---

## 3. Repository structure and module map

```
CV-Hackthon/
├── main.py                  CLI entry: --rgb --depth --scale --method --output [--gt --compare]
├── configs/default_config.yaml   frozen hybrid fusion + app defaults
├── classical/               classical branch
│   ├── interpolation.py     nearest / bilinear / bicubic
│   ├── jbu.py               joint bilateral upsampling
│   ├── guided_filter.py     guided filter
│   ├── hole_filling.py, boundary_refinement.py, depth_discontinuity.py
│   ├── adaptive_jbu.py, classical_hr_depth.py (public wrapper), pipeline.py
│   ├── benchmark.py         synthetic classical benchmark (python -m classical.benchmark)
│   ├── methods/             (feat/classical, hybrid-v2 only) tuned/adaptive/refined/masked JBU
│   ├── evaluation/          (feat/classical, hybrid-v2 only) classical eval suite
│   └── edge_upsampling.py   (from origin/hybrid) classical_blend upsampler
├── deep/                    deep branch
│   ├── deep_model.py        SGNet wrapper (get_model, per-scale checkpoints)
│   ├── inference.py         deep_hr_depth() + profile_inference() + CPU/GPU smoke CLI
│   ├── settings.py          scale support {4,8,16}, env vars SGNET_CKPT_*, DeepModelUnavailableError
│   ├── dataset.py, preprocessing, finetune.py, finetune_robust.py
│   ├── adaptive.py          (from origin/hybrid) E8 LR-only regime detection
│   ├── finetune_v2.py       (from origin/hybrid) E4 curriculum training
│   └── sgnet/               vendored official SGNet (Apache-2.0)
├── hybrid/                  hybrid branch
│   ├── confidence.py        pixel-wise branch confidence (W_c + W_d == 1)
│   ├── fusion.py            fuse() / fuse_fixed()
│   ├── consistency.py       measurement-consistency correction r_LR = D_LR - S(D_f)
│   ├── boundary_refinement.py, pipeline.py (hybrid_hr_depth, ablations)
├── eval/                    metrics.py, evaluate.py, benchmark.py
├── data/                    preprocessing, degradation, dataset loaders, middlebury_raw
├── app/                     ui.py (Gradio), visualization.py, config.py, validation.py
├── scripts/, tests/         6 test modules (app/classical/deep/hybrid/integration/pipeline)
├── docs/                    ARCHITECTURE, EVALUATION_METHODOLOGY, IMPLEMENTATION_STATUS,
│                            LIMITATIONS, HYBRID_FUSION, GPU_RUN_REPORT, APP_GUIDE, API_REFERENCE,
│                            TESTING_AND_VALIDATION (+ EXPERIMENTAL_RESULTS, EXPERIMENT_LOG,
│                            PERFORMANCE_ANALYSIS, DEEP_MODEL_IMPROVEMENTS from origin/hybrid)
├── results/                 tracked evaluation outputs (84 files)
├── tmp/                     gitignored scratch: 6 cloned reference repos (Phase 2), bench scripts
├── checkpoints/             gitignored; SGNet *.pth  (partial — see §12)
└── datasets/                gitignored
```

---

## 4. End-to-end pipeline

```
lr_depth + hr_rgb + valid_mask
        │
        ├─► classical_hr_depth() ───► D_c     (edge-aware upsampling; no ML)
        ├─► deep_hr_depth()      ───► D_d     (SGNet forward; needs checkpoint)
        │         │
        │         └─ preprocessing: resolve mask → nearest-fill invalid LR →
        │            per-image min-max over valid pixels → forward → denormalise → clamp ≥ 0
        │
        └─► hybrid_hr_depth()    ───► D_f
                  1. compute D_c, D_d
                  2. estimate_confidence() → (W_c, W_d), W_c + W_d == 1
                  3. fuse() → D_f = W_c·D_c + W_d·D_d   (or fixed α = 0.02)
                  4. optional boundary refinement (off by default)
                  5. measurement consistency: r_LR = D_LR − S(D_f), damped 0.25, 1 iter
                  6. deep unavailable → classical output + deep_available=False
                     (evaluation pre-checks the checkpoint and reports
                     "not evaluated" instead)
```

Degradation (synthetic) side: `data/degradation.py` builds LR depth from GT with a
known forward operator `S` (bicubic), which is what makes the consistency step legal
on synthetic data.

---

## 5. Classical branch

- **Public API:** `classical_hr_depth()` in `classical/classical_hr_depth.py`
  delegates to `reconstruct(..., method="classical")` — the edge-preserving
  pipeline (`prepare_depth` → dense/masked edge upsampling from
  `edge_upsampling.py`). `reconstruct()` accepts
  `{nearest, bilinear, bicubic, jbu, guided, classical, classical_blend,
  adaptive_jbu, refined_jbu}`; its default method is `jbu`.
- **Members:** interpolation baselines, joint bilateral upsampling (JBU), guided
  filter, hole filling, depth-discontinuity detection, boundary refinement,
  adaptive JBU. Benchmarked with `python -m classical.benchmark`.
- **On `feat/classical` / `hybrid-v2` (unmerged):** add
  `tuned_jbu / masked_jbu`, a classical evaluation suite and stress reports.
  `edge_upsampling.py` (classical_blend), measured **better than bicubic** on
  val150 clean (RMSE 0.0250 vs 0.0292), came in with `origin/hybrid` and is now
  merged here.
- **Known weakness (measured, master):** the current classical pipeline scores
  **worse than plain bicubic** on RMSE on NYU×4 clean (0.0584 vs 0.0367) while
  winning on edge F1 (0.671 vs 0.259) — it preserves edges but is not the RMSE
  winner. This is the classical-branch gap the research-driven revision targets.

---

## 6. Deep branch (SGNet)

- Vendored **SGNet** guided depth super-resolution (`deep/sgnet/`, Apache-2.0),
  one checkpoint per scale **x4 / x8 / x16** (x2 has no released checkpoint —
  deep scales are `[4, 8, 16]` in `configs/default_config.yaml`).
- `deep/settings.py` resolves checkpoints from `SGNET_CKPT_DIR` (default
  `checkpoints/sgnet`) or `SGNET_CKPT_X{scale}` and raises
  `DeepModelUnavailableError` when unavailable; `inference.py` supports
  `--random-init` for architecture/latency-only smoke tests.
- Contract notes: no mask input to SGNet (invalid LR pixels are nearest-filled on a
  working copy), fp32 forward (cuFFT-in-fp16 needs 2^n), tiled fallback on OOM
  (flagged as not bit-identical because of global FFT blocks), NaN/Inf → error.
- **Local status:** CPU-only machine, `torch 2.11.0+cu128` installed but
  **CUDA unavailable**. `checkpoints/sgnet/SGNet_X4_R.pth` is **missing**;
  present are `model/SGNet_Real.pth`, `SGNet_Real_R.pth`, `SGNet_X16.pth`,
  `SGNet_X16_R.pth` (partial). `runs/e7_6epoch/best.pth` (E7 robust ckpt) also
  absent. ⇒ local deep/hybrid **quality** evaluation is not possible as-is;
  latency/architecture smoke is possible via `--random-init`.
- **Prior deep results are real teammate measurements**, preserved under
  `results/` and `docs/GPU_RUN_REPORT.md` (RTX 5070, 479 images): never to be
  re-labelled or extrapolated.

---

## 7. Hybrid branch

- `hybrid_hr_depth()` in `hybrid/pipeline.py`; confidence in
  `hybrid/confidence.py` (evidence: depth-gradient structure, measurement
  proximity via EDT, RGB/depth edge agreement — combined by fixed convex
  combination; weights are **normalized heuristics, not calibrated
  probabilities**).
- **Frozen configuration** (`configs/default_config.yaml`):
  `D_f = 0.02·D_c + 0.98·D_d`, consistency strength 0.25 / 1 iteration,
  boundary refinement **off** (it worsened val RMSE).
  Chosen on NYU val30 ×4 clean: RMSE 0.01228 vs **0.0123 for deep alone** —
  the README states plainly that the gain is marginal and the classical branch is
  a safeguard more than a contributor.
- **Unmerged hybrid work:**
  - `stash@{0}` (branch `feat/combined-hybrid`, "prework") — a substantial,
    test-covered WIP: hole-band confidence (`hole_classical_weight=0.5`, classical
    gains weight in the band around missing LR measurements where the deep input is
    nearest-filled), invalid-branch handling in `fuse()`/`fuse_fixed()`,
    expanded `ablations()` keys, and 6 new hermetic tests in `tests/test_hybrid.py`.
    Its blobs are based on the `origin/hybrid` versions of those files.
  - `origin/hybrid` — regime-aware fusion, `edge_upsampling.py`, `deep/adaptive.py`
    (E8), `finetune_v2.py` (E4), statistical-comparison and forensic-probe scripts.
    **Latent bug:** `hybrid.pipeline.ablations()` on that branch references a
    removed `confidence_boundary` variable → `NameError` when called.
  - `stash@{1}` (branch `hybrid-v2`, "WIP: denoise + default-method switch") —
    median denoise of the working depth + default classical method switch
    (`jbu` → `classical`).

---

## 8. Evaluation methodology

- **Entry point:** `python -m eval.evaluate --dataset nyu --split val --scale 4
  --methods bicubic classical deep hybrid [--max-samples N --noise-std --hole-ratio
  --seed --out results]`; `--final` is required for `test`/`full` splits so no
  tuning happens on them.
- **Metrics** (`eval/metrics.py`): RMSE, MAE, PSNR, SSIM (fixed data range 10 m
  for NYU), boundary RMSE (band around **GT** depth discontinuities), edge
  precision/recall/F1 on depth edges (not RGB), latency. Memory profiling exists
  only in `deep.inference.profile_inference` (None on CPU).
- **Protocol rules** already written down in `docs/EVALUATION_METHODOLOGY.md`
  (train/val/test discipline, no tuning on test, fixed seeds). The methodology
  doc is the contract the new experiments must follow.
- **Stress dimensions already implemented:** Gaussian noise (`--noise-std`),
  random holes (`--hole-ratio`), seeds.
- Existing result directories (`results/benchmark_val*`, `hf_nyu_val_x4*`,
  `robust_*`, `x4_noise`, `x4_holes`, `middlebury_raw_*`) are tracked baselines.

**Measured baselines (NYU ×4, HF val100, clean) — reproduced from tracked results:**

| method | RMSE | MAE | edge F1 | latency |
|---|---|---|---|---|
| bicubic | 0.0367 | 0.0121 | 0.259 | — |
| classical (master) | 0.0584 | 0.0191 | 0.671 | — |
| deep (SGNet) | 0.0101 | 0.0030 | 0.960 | 1900 ms |
| hybrid legacy | 0.0298 | 0.0077 | 0.804 | — |
| hybrid adaptive | 0.0338 | — | — | — |
| hybrid fixed α=0.02 | ≈ deep | ≈ deep | — | — |

Robustness findings already measured: deep **loses to bicubic under 10 % holes**
(0.0438 vs 0.0374) and degrades under noise; classical is weak on RMSE; hybrid
fixed-α ≈ deep alone.

---

## 9. Existing results and provenance

- `results/` (84 tracked files) + `docs/GPU_RUN_REPORT.md` = **teammate's real
  GPU measurements** (RTX 5070, 479 images). Preserved, never fabricated.
- `results/benchmark_test_selected/` exists → a test-split run has already been
  produced (its protocol is documented in `docs/EVALUATION_METHODOLOGY.md`).
- Additional experiment docs on `origin/hybrid`:
  `EXPERIMENTAL_RESULTS.md`, `EXPERIMENT_LOG.md`, `DEEP_MODEL_IMPROVEMENTS.md`,
  `PERFORMANCE_ANALYSIS.md`, plus `results/e4b`, `results/e7`, `results/e8`,
  `results/error_analysis`.
- `feat/classical` carries `classical/REAL_DEPTH_RESULTS.md` and
  `RANDOM_STRESS_RESULTS.md` (real-data and stress evaluations of the
  classical variants).
- Anything this session measures itself must be labelled **new, CPU, smoke** and
  stored separately from these.

---

## 10. Application layer and deployment

- **CLI:** `python main.py --rgb RGB.png --depth depth.npy --scale N --method
  {nearest,bilinear,bicubic,jbu,guided,classical,adaptive_jbu,refined_jbu,deep,hybrid}
  --output out/ [--gt] [--compare]`.
- **Web UI:** Gradio (`python -m app/ui`), config in `app/config.py`
  (`P4_CONFIG` env overrides `configs/default_config.yaml`), input validation in
  `app/validation.py`, visualization/export in `app/visualization.py`.
- **Docker/AWS:** `Dockerfile` + `.dockerignore` + `requirements.web.txt`;
  Gradio binds `0.0.0.0:$PORT` (8080), App Runner instructions in README.
  The shipped image deliberately contains **no torch and no checkpoints**, so the
  deployed app runs the classical fallback (`deep_available=False`).
- **Colab:** `CV_Hackthon_Colab.ipynb` — clones the repo, runs
  `main.reconstruct`, checkpoints uploaded manually (not in git).
- Gradio is **not installed** in the local `.venv` (status UNKNOWN until checked).

---

## 11. Git topology, workstreams and uncommitted work

```
root ─ feat/implement ─ raksh-dl ─ raksh-hybrid ─ (merges) ─► master = 3cc1540 (= origin/master)
                                                              │
  feat/classical      fc624a0   5 ahead  NOT in master       │
  feat/combined-hybrid=origin/improve 95df9bb 2 ahead  NOT in master
  hybrid-v2=origin/hybrid-v2 476904f 1 ahead  NOT in master
  origin/hybrid       4022437  13 ahead  NOT in master  ← most advanced (contains origin/improve)
  origin/kumarbro     9705c2f  UNRELATED root history (5a517d7), deploy/UI + depthsr_testset
  orphan 482ba4b      in no branch
```

- In master: `feat/implement`, local `hybrid`, `origin/raksh-dl` (SGNet),
  `origin/raksh-hybrid`.
- Not in master: `feat/classical`, `feat/combined-hybrid`, `hybrid-v2`,
  **`origin/hybrid`**, `origin/kumarbro` (+ dangling `482ba4b`).
- Working tree on master: **clean** (`git status`: `## master...origin/master`).
- **Stashes (uncommitted work):**
  - `stash@{0}` on `feat/combined-hybrid` — "prework": hole-band confidence +
    invalid-branch fusion + `ablations()` key expansion + 6 tests (see §7).
  - `stash@{1}` on `hybrid-v2` — "WIP: denoise + default-method switch".
- **Risks in unmerged branches:** `origin/hybrid`'s `ablations()` NameError;
  `origin/kumarbro` shares no history with master (its files cannot be merged
  cleanly — must be `git show`/cherry-picked by file).
- Untracked-but-ignored: `tmp/` (reference clones, bench scripts,
  `health.html`, `v1_classical.json`, `v2_classical.json`), `checkpoints/`,
  `datasets/`, `runs/`, `outputs/`.
- **`.gitignore` ignores `tmp/`, `checkpoints/`, `datasets/`, `runs/`, `outputs/`;
  `results/` IS tracked.** Therefore cloned reference repos, datasets, weights and
  experiment outputs must NOT be committed; `results/` may be extended with new
  clearly-labelled entries.

---

## 12. Environment, tests, gaps and risks

### Environment (verified)

- Windows, PowerShell; project `.venv` **Python 3.14.7**, `torch 2.11.0+cu128`,
  **CUDA unavailable** (no usable GPU on this machine).
- Package installs must go through `uv pip install --python .venv/Scripts/python.exe <pkg>`
  (no pip in the venv; system python is 3.13). `pypdf` was installed this way.
- `dcg` (command guard) blocks shell redirects to dynamic/escaped paths — write
  files through Python `open()` or literal paths instead of `> tmp\...` / `2>$null`.

### Tests

- `tests/`: `test_app`, `test_classical`, `test_deep`, `test_hybrid`,
  `test_integration`, `test_pipeline` (pytest).
- One deep-dependent test (`test_consistency_sweep_method_name_is_supported`)
  needs torch; README documents the degraded-without-torch behaviour.
- `stash@{0}` adds 6 more hermetic hybrid tests.

### Gaps / risks (explicit)

1. **No local GPU** → all local verification must be CPU smoke tests, clearly
   labelled; no training, no full-dataset runs, no large downloads.
2. **Missing checkpoints:** `SGNet_X4_R.pth` (the primary x4 robust checkpoint)
   and `runs/e7_6epoch/best.pth` are absent → deep/hybrid quality evaluation
   cannot run locally; use `--random-init` for latency/architecture-only smoke, or
   evaluate on Kaggle/Colab with an uploaded checkpoint.
3. **Classical branch currently loses to bicubic on RMSE** while winning on edge
   F1 — the main classical improvement target.
4. **Hybrid fixed-α gain over deep is marginal** (0.01228 vs 0.0123) and adaptive
   fusion is *worse* (0.0338) — the confidence fusion needs the research-driven
   rework (hole band + regime awareness from the stashes/`origin/hybrid`).
5. **Deep is worse than bicubic under 10 % holes and under noise** — robustness
   is an open, already-measured failure mode.
6. `origin/hybrid.ablations()` NameError; `origin/kumarbro` unrelated history.
7. Deployment image ships without torch → live URL would serve classical only.
8. KITTI unused; Middlebury partially used; **team USNs, deadline, live URL and
   scoring weights beyond the rubric: UNKNOWN.**

---

*End of Phase 1. Next: Phase 2 — study of the six reference repositories under
`tmp/` → `RESEARCH_REFERENCE_ANALYSIS.md`.*
