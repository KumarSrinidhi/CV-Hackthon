# PS4 — Edge-Preserving Depth Map Upsampling & Refinement · Team Plan & Task Board

> **Claim a task:** put your name/handle in the **Owner** column, comment on the repo, and work on the matching feature branch. Push small commits often. Open a PR when your task is done (PR title = task ID).

**ProbID P4 (from the official PDF):** build an intelligent, edge-aware depth reconstruction system that turns **sparse / low-resolution depth maps** into **dense, high-resolution depth**, preserving object boundaries and depth discontinuities. RGB guidance available. Must compare **interpolation vs classical refinement vs deep learning vs hybrid**, support **2×/4×/8×**, handle **missing + noisy depth**, and report **RMSE, MAE, PSNR, SSIM, edge preservation, boundary accuracy, latency, memory**. Deliverable: a **deployed app** + evaluation + ablations + failure analysis.

**Timeline:** ~1 week · **DL hardware:** Colab GPUs · **Dataset:** NYU Depth v2 (labeled)

---

## 0. Quick Start (everyone, Day 0)

```bash
git clone https://github.com/KumarSrinidhi/CV-Hackthon.git
cd CV-Hackthon
uv sync                      # creates .venv from uv.lock (Python 3.12)
uv run python -c "import cv2, numpy, torch; print('env OK')"
uv run pytest tests/         # should pass once your task's tests exist
```

- **Env manager: UV only.** Add deps with `uv add <pkg>` (updates `uv.lock`) — never pip/conda. Commit the lock change with your task.
- Scratch files, datasets, big downloads → **`tmp/` only** (gitignored).

## 1. Repo Layout (target)

```
CV-Hackthon/
├── pyproject.toml  uv.lock  .python-version     # UV project at ROOT (already pinned)
├── .gitignore                # tmp/, data/, *.mat, weights/, outputs/, __pycache__/
├── TEAM_PLAN.md              # ← this file
├── README.md                 # final report: results tables, ablations, failure analysis
├── src/ps4/
│   ├── data/     loader.py  preprocess.py  degrade.py      # Track B
│   ├── classical/ interp.py jbf.py guided.py edges.py holes.py  # Track A
│   ├── deep/     model.py losses.py train.py infer.py      # Track C
│   ├── hybrid/   fuse.py                                    # Track D
│   ├── eval/     metrics.py benchmark.py viz.py             # Track B (+D ablations)
│   ├── app/      gradio_app.py                              # Track C
│   └── cli.py
├── configs/      benchmark.yaml  train.yaml
├── notebooks/    colab_train.ipynb                          # Track C (GPU entry point)
├── tests/        test_classical.py  test_data.py  test_metrics.py
└── tmp/          # GITIGNORED — data cache, scratch, local experiments
```

## 2. Git Workflow

- `master` = stable, always-runnable. Do **not** commit to master directly (except small doc fixes).
- Branch per track: `feat/classical`, `feat/data-eval`, `feat/deep`, `feat/hybrid-app`
- **Commit small & push often** (multiple times a day). PR per task; one review, then merge.
- Commit messages: `P4-<task-id>: <what>` e.g. `P4-A3: implement own guided filter with box filters`
- **Never commit:** `.mat`, weights >50MB, datasets, `.venv/`, `tmp/`. Weights go to Google Drive/Release link → paste link in README weights table.

---

## 3. Task Board

### 🔧 Day 0 — Foundation (blocking everything else)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **F1** | Move `pyproject.toml`, `uv.lock`, `.python-version` from `tmp/` to repo root; write `.gitignore` (tmp/, data/, *.mat, weights/, outputs/, __pycache__/); create full `src/ps4/` skeleton with empty modules + `uv sync` works at root | root files, `src/ps4/**` | _unclaimed_ | `uv sync && uv run pytest` runs green on empty skeleton; skeleton pushed | — |
| **F2** | Download `nyu_depth_v2_labeled.mat` (~2.8GB) → `tmp/data/`. Primary: official NYU page (cs.nyu.edu/~silberman). Fallbacks: Kaggle mirror, Hugging Face mirror. Write the exact working URL + command into `src/ps4/data/README.md` | `tmp/data/`, notes | _unclaimed_ | File verified loadable (h5py/scipy.io), URL documented | F1 |
| **F3** | Preprocess: parse `.mat` → per-image `.npz` cache in `tmp/cache/` (RGB uint8 640×480, depth meters float16, label map). Eigen split 795 train / 654 test; carve 50-image val from train. Save split lists to `configs/splits/{train,val,test}.txt` | `src/ps4/data/loader.py`, `preprocess.py` | _unclaimed_ | `uv run python -m ps4.data.preprocess` produces 1499 npz + split files in <10 min | F2 |

### 📐 Track A — Classical CV (CPU, pure numpy/OpenCV)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **A1** | Interpolation baselines: nearest, bilinear, bicubic (wrap `cv2.resize`); expose uniform API `upsample(lr_depth, rgb, factor, method)` | `classical/interp.py` | _unclaimed_ | Round-trip test: downsample→upsample at 1× ≈ identity; all 3 factors work | F1 |
| **A2** | **Own guided filter** (He et al. 2010, box-filter based, O(N)) for depth guided by RGB; cross-check against `cv2.ximgproc.guidedFilter` (max abs diff < 1e-2 on synthetic) | `classical/guided.py`, `tests/test_classical.py` | _unclaimed_ | Test green incl. cv2 cross-check | A1 |
| **A3** | Joint bilateral filter for depth upsampling: brute-force at small radius + fast approx (bilateral grid or separable) for 8×; auto params from image stats | `classical/jbf.py` | _unclaimed_ | Visual: edges crisper than bicubic on 3 sample pairs | A1 |
| **A4** | Edge & discontinuity detection: RGB gradient (Sobel/Canny) map, depth-gradient discontinuity mask (thresh + morphological cleanup), confidence map (1 = reliable) | `classical/edges.py` | _unclaimed_ | Overlay viz on 3 samples looks correct (edges on object boundaries) | A1 |
| **A5** | Hole filling & denoising: morphological dilate-fill → JBF-guided inpaint for holes; median/bilateral denoise with edge preservation; `cv2.inpaint` cross-check | `classical/holes.py` | _unclaimed_ | Fills 30% random holes; PSNR vs GT improves over raw fill | A2, A3 |
| **A6** | Boundary-aware classical refinement: fusion policy — classical edge-aware output near detected boundaries, smoother estimate in flat regions | `classical/refine.py` | _unclaimed_ | Boundary RMSE improves vs plain JBF | A2–A4 |

### 📊 Track B — Data + Evaluation

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **B1** | Degradation simulator: box-filter+subsample downsampling (2/4/8×); sparse sampling (500 pts + gradient-dense variant); Gaussian noise σ∈{1,2,3}cm; random hole masks 10–30%; boundary erosion. Deterministic (seeded) | `data/degrade.py`, `tests/test_data.py` | _unclaimed_ | Round-trip + determinism tests green; covers all advanced requirements | F3 |
| **B2** | Metrics: RMSE, MAE (meters, valid-GT mask), PSNR/SSIM (normalized depth), **boundary RMSE** (RMSE in dilated GT-discontinuity band), **edge F1** (depth-Sobel edges vs GT), gradient-preservation score | `eval/metrics.py`, `tests/test_metrics.py` | _unclaimed_ | Sanity tests: identical maps → 0 error; known-degraded map → expected worse score | F3 |
| **B3** | Benchmark runner: `method × factor(2/4/8) × noise-level` grid over val split → pandas DataFrame → CSV + Markdown table; per-run config JSON saved (reproducibility) | `eval/benchmark.py`, `configs/benchmark.yaml` | _unclaimed_ | Table fills every cell with classical methods; runs on 50-image val in <30 min | A1–A3, B2 |
| **B4** | Visualization: error heatmaps, GT vs pred side-by-sides, boundary overlay (GT edges in red, pred edges in cyan), zoomed discontinuity crops | `eval/viz.py` | _unclaimed_ | 3+ sample figure sets generated from benchmark output | B3 |
| **B5** | Latency + memory harness: wall-clock per method (CPU) + peak RAM (`tracemalloc`/psutil); GPU latency note for DL from Colab; params count | `eval/benchmark.py` extension | _unclaimed_ | Latency/memory columns appear in benchmark table | B3 |

### 🧠 Track C — Deep Learning (Colab GPU) + App

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **C1** | Unified U-Net (~2M params): inputs = [depth, validity mask, RGB] → dense depth. Handles SR (LR dense, mask≈1) AND completion (sparse) in one model. Log-depth parametrization, metrics in meters | `deep/model.py` | _unclaimed_ | Forward pass test: 256² input → 256² output; param count ≤ 3M | F1 |
| **C2** | Losses: masked L1 + multi-scale gradient loss (Sobel) + optional SILog; edge-weighted variant (weight near GT boundaries) | `deep/losses.py` | _unclaimed_ | Unit test: loss decreases on toy overfit (10 patches) | C1 |
| **C3** | Colab training notebook: patch sampling (256², flips, RGB jitter), AMP, AdamW, cosine schedule, checkpoint to Drive, CSV logs, val RMSE each epoch | `notebooks/colab_train.ipynb`, `deep/train.py` | _unclaimed_ | Run #1 completes: val RMSE < bicubic baseline on val | C1, C2, B1, F3 |
| **C4** | Fine-tuning phase: randomized degradation mix (sparsity, noise, factor) as domain randomization; early stop on val; export best weights + inference script | `deep/train.py`, `deep/infer.py` | _unclaimed_ | Fine-tuned weights beat run #1 on all 3 factors; weights link in README | C3 |
| **C5** | Gradio app: pick sample or upload RGB+depth → method selector (all classical + DL + hybrid) × factor → output image, metrics panel, boundary overlay. CPU fallback for DL inference | `app/gradio_app.py`, `cli.py` | _unclaimed_ | `uv run python -m ps4.app.gradio_app` serves demo with all methods | C4, A1–A6, D1 |

### 🧬 Track D — Hybrid + Report

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **D1** | Hybrid v1 — guided-filter residual correction: GF(residual = known−upsampled) anchored at valid pixels, applied to DL *or* classical output; depth-domain residual ⇒ no texture-copying | `hybrid/fuse.py` | _unclaimed_ | Boundary RMSE improves over uncorrected DL output | A2, C4 |
| **D2** | Hybrid v2 — confidence fusion: per-pixel blend of DL vs classical using RGB-gradient confidence + local prediction error at valid pixels | `hybrid/fuse.py` | _unclaimed_ | Beats both parents on ≥2 of {RMSE, boundary RMSE, edge F1} | D1, A6 |
| **D3** | Ablations: no-RGB-guidance, no-edge-loss, no-hybrid, single classical components removed | `eval/ablation.py` | _unclaimed_ | Ablation table in README with deltas | B3, C4, D2 |
| **D4** | Robustness + unseen eval: full 654-image test split, noise/hole sweep curves, failure analysis (texture-copy cases, extreme sparsity, transparent/reflective surfaces) with figures | README section | _unclaimed_ | Sweep plots + ≥3 documented failure cases | B3, C4 |
| **D5** | Final README report: method descriptions, full comparison tables, ablations, latency/memory, failure analysis, how-to-run, weights links | `README.md` | _unclaimed_ | README self-contained; a stranger can run everything | all |

---

## 4. Phase Timeline (suggested order of execution)

| Phase | Day | Gates (PR merged to master) |
|-------|-----|------------------------------|
| **P0 Foundation** | 0 | F1–F3 merged; env green for everyone |
| **P1 Build** | 1–2 | A1–A4, B1–B2, C1–C2 merged; first classical benchmark table posted |
| **P2 Results** | 2–3 | C3 first Colab run done; B3–B4 merged; full classical matrix on val |
| **P3 Fusion** | 3–4 | C4 fine-tuned weights; A5–A6, D1–D2 merged; 4-way comparison complete |
| **P4 Ablate** | 5 | D3–D4 merged; sweep + ablation tables done |
| **P5 Ship** | 6 | C5 app + D5 README merged; tag `v1.0` |
| Buffer | 7 | bugfixes, PR cleanup |

## 5. Acceptance Criteria (whole project)

1. Benchmark table: every `method × factor(2/4/8)` cell filled with RMSE/MAE/PSNR/SSIM/boundary-RMSE/edge-F1/latency
2. **Hybrid ≥ DL** on boundary RMSE & edge F1; **DL ≥ all classical** on RMSE at 8×; **hybrid ≥ classical** on RMSE
3. Works on held-out test split (654 unseen images) + handles noise/holes
4. `uv run pytest tests/` green; pipeline end-to-end smoke on 5 images < 1 min
5. App runs locally; README reproducible from a clean clone

## 6. Task Board Sync

- When you **claim** a task: replace `_unclaimed_` with your handle, commit `P4-<id>: claim`, push.
- When **done**: tick nothing here — the merged PR is the record. If scope changed, edit this row in the same PR.
- Blockers: open a GitHub issue titled `P4-<id>: blocker — <short>`.
