# PS4 — Edge-Preserving Depth Map Upsampling & Refinement · Team Plan & Task Board

> **Claim a task:** put your name/handle in the **Owner** column, comment on the repo, and work on the matching feature branch. Push small commits often. Open a PR when your task is done (PR title = task ID).

**ProbID P4:** build an intelligent, edge-aware depth reconstruction system that turns **sparse / low-resolution depth maps** into **dense, high-resolution depth**, preserving object boundaries and depth discontinuities. RGB guidance available. Must compare **interpolation vs classical refinement vs deep learning vs hybrid**, support **2×/4×/8×**, handle **missing + noisy depth**, and report **RMSE, MAE, PSNR, SSIM, edge preservation, boundary accuracy, latency, memory**.

**Timeline:** ~1 week · **DL hardware:** Colab GPUs · **Dataset:** NYU Depth v2 (labeled, primary)

---

## 0. Mandatory Deliverables (from the PDF Instruction Sheet — do not skip)

These are what actually get graded. Missing any of these caps a rubric category hard.

| # | Deliverable | Owner | Due |
|---|---|---|---|
| **D1** | **Live hosted application URL** (public, works on phone + laptop) | P3 | Day 5 |
| **D2** | **Complete source code** in GitHub | all | ongoing |
| **D3** | **`docs/architecture.md`** — system architecture diagram matching the code | P4 | Day 1 |
| **D4** | **`docs/DESIGN.md`** — design document (every template section; use organizer template if provided) | P4 | Day 3 |
| **D5** | **`eval_P4_T<teamID>.md`** — metrics in PDF order, **first line = all USNs comma-separated** | P2 | Day 5 |
| **D6** | **Ablation study + failure-case analysis** (in README + design doc) | P1/P4 | Day 5 |
| **D7** | **`Dockerfile` or one-command run; no secrets committed** | P4 | Day 1 |

**Rubric weights:** Demo/viva 25 · Evaluation metrics 25 · Implementation 20 · UI/UX + deployment 15 · Documentation 15.

---

## 1. Quick Start (everyone, Day 0)

```bash
git clone https://github.com/KumarSrinidhi/CV-Hackthon.git
cd CV-Hackthon
uv sync                      # creates .venv from uv.lock (Python 3.12)
uv run python -c "import cv2, numpy, torch; print('env OK')"
uv run pytest tests/         # should pass once your task's tests exist
uv run python -m ps4.cli --help   # one-command entry point
```

- **Env manager: UV only.** Add deps with `uv add <pkg>` (updates `uv.lock`) — never pip/conda. Commit the lock change with your task.
- Scratch files, datasets, big downloads → **`tmp/` only** (gitignored).
- **Never commit:** `.mat`, weights >50MB, datasets, `.venv/`, `tmp/`, secrets/keys. Weights go to Google Drive/Release link → paste link in README weights table.

## 2. Repo Layout (target)

```
CV-Hackthon/
├── pyproject.toml  uv.lock  .python-version     # UV project at ROOT
├── .gitignore                # tmp/, data/, *.mat, weights/, outputs/, __pycache__/
├── Dockerfile
├── .env.example
├── TEAM_PLAN.md              # ← this file
├── README.md                 # final report: results, ablations, failure analysis, ethics
├── eval_P4_T<teamID>.md      # required metric file (USNs on line 1)
├── docs/
│   ├── architecture.md       # mermaid diagram matching code
│   └── DESIGN.md             # design doc
├── src/ps4/
│   ├── data/     loader.py  preprocess.py  degrade.py
│   ├── classical/ interp.py jbf.py guided.py edges.py holes.py refine.py
│   ├── deep/     model.py losses.py train.py infer.py
│   ├── hybrid/   fuse.py
│   ├── eval/     metrics.py benchmark.py viz.py ablation.py
│   ├── app/      gradio_app.py
│   └── cli.py
├── configs/      benchmark.yaml  train.yaml
├── notebooks/    colab_train.ipynb
├── tests/        test_classical.py  test_data.py  test_metrics.py
└── tmp/          # GITIGNORED — data cache, scratch, local experiments
```

## 3. Git Workflow

- `master` = stable, always-runnable. Do **not** commit to master directly (except small doc fixes).
- Branch per track: `feat/classical`, `feat/data-eval`, `feat/deep`, `feat/hybrid-app`, `feat/docs`
- **Commit small & push often.** PR per task; one review, then merge.
- Commit messages: `P4-<task-id>: <what>` e.g. `P4-A3: implement own guided filter with box filters`

---

## 4. Ownership (3–4 people)

| Person | Role | Owns |
|---|---|---|
| **P1** | Classical + Hybrid | Track A (A1–A7) + Track D (D1–D2) |
| **P2** | Data + Evaluation | Track B (B1–B6) |
| **P3** | Deep Learning + App + Hosting | Track C (C1–C7) |
| **P4** | Docs + Delivery (shared/flex) | F4–F5, D3–D7; writes own sections into P1–P3 work |

With only 3 people, merge P4 into P1/P2/P3 and have P3 also own the design doc + diagram.

## 5. Task Board

### 🔧 Day 0 — Foundation (blocking everything else)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **F1** | Move `pyproject.toml`, `uv.lock`, `.python-version` from `tmp/` to root; write `.gitignore`; create `src/ps4/` skeleton + `cli.py` stub | root, `src/ps4/**` | _unclaimed_ | `uv sync && uv run pytest` green; `python -m ps4.cli --help` works | — |
| **F2** | Download `nyu_depth_v2_labeled.mat` (~2.8GB) → `tmp/data/`. Primary: NYU official page; fallbacks Kaggle/HF. Document exact URL in `src/ps4/data/README.md` | `tmp/data/` | _unclaimed_ | File loadable (scipy.io/h5py), URL documented | F1 |
| **F3** | Preprocess `.mat` → per-image `.npz` cache (RGB, depth meters, labels). Eigen split 795 train / 654 test, hold 50 val. Write `configs/splits/*.txt` | `data/loader.py`, `data/preprocess.py` | _unclaimed_ | `python -m ps4.data.preprocess` → 1499 npz + splits <10 min | F2 |
| **F4** | `docs/DESIGN.md` skeleton + `docs/architecture.md` (mermaid diagram v1); confirm organizer design-doc template | `docs/*` | _unclaimed_ | Diagram v1 + design skeleton merged | F1 |
| **F5** | `Dockerfile` + `.env.example`; verify one-command run; confirm no secrets in repo | root | _unclaimed_ | `docker build`/`uv run` works; `git ls-files` has no keys | F1 |
| **F6** | Record Team ID + USN list; choose hosting (recommend Hugging Face Spaces) | `configs/team.yaml` | _unclaimed_ | `TEAM_ID` + `USNS` filled in config | — |

### 📐 Track A — Classical CV (P1, CPU, numpy/OpenCV)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **A1** | Interpolation baselines NN/bilinear/bicubic; uniform `upsample(lr, rgb, factor, method)` | `classical/interp.py` | _unclaimed_ | 1× round-trip ≈ identity; 2/4/8× all work | F1 |
| **A2** | **Own guided filter** (He et al., box-filter O(N)) + `cv2.ximgproc.guidedFilter` cross-check (<1e-2 abs diff) | `classical/guided.py`, `tests/test_classical.py` | _unclaimed_ | Cross-check test green | A1 |
| **A3** | Joint bilateral filter (brute force small radius + fast approx for 8×); auto params | `classical/jbf.py` | _unclaimed_ | Edges crisper than bicubic on 3 samples | A1 |
| **A4** | Edge + depth-discontinuity detection (Sobel/Canny, gradient thresh + morph), confidence maps | `classical/edges.py` | _unclaimed_ | Boundary overlay correct on 3 samples | A1 |
| **A5** | Hole filling + denoising (morph dilate → JBF inpaint; median/bilateral; `cv2.inpaint` check) | `classical/holes.py` | _unclaimed_ | 30% holes filled; PSNR improves over raw fill | A2, A3 |
| **A6** | Boundary-aware classical refinement (edge-aware near boundaries, smooth in flat regions) | `classical/refine.py` | _unclaimed_ | Boundary RMSE improves vs plain JBF | A2–A4 |
| **A7** | Bad-input handling: missing/empty depth, mismatched RGB/depth size, all-zero depth, non-square → graceful fallback + error msg | `classical/*`, `cli.py` | _unclaimed_ | Bad inputs raise clear errors, never crash silently | A1 |

### 📊 Track B — Data + Evaluation (P2)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **B1** | Degradation sim: 2/4/8× (box+subsample), 500-pt sparse (+gradient-dense), σ 1–3cm noise, 10–30% holes, boundary erosion; seeded | `data/degrade.py`, `tests/test_data.py` | _unclaimed_ | Determinism + round-trip tests green | F3 |
| **B2** | Metrics: RMSE, MAE (meters, valid-GT), PSNR, SSIM (norm), boundary-RMSE (dilated GT-discontinuity band), edge-F1 (depth-Sobel), grad-preservation | `eval/metrics.py`, `tests/test_metrics.py` | _unclaimed_ | Sanity tests green (identity→0, degraded→worse) | F3 |
| **B3** | Benchmark runner `method × factor(2/4/8) × noise` → pandas → CSV + Markdown; per-run config JSON | `eval/benchmark.py`, `configs/benchmark.yaml` | _unclaimed_ | Every cell filled; val 50 in <30 min | A1–A3, B2 |
| **B4** | Visualization: error heatmaps, GT vs pred, boundary overlay (GT red / pred cyan), zoomed crops | `eval/viz.py` | _unclaimed_ | 3+ figure sets from benchmark | B3 |
| **B5** | Latency + memory harness (CPU classical + **DL CPU inference path used by app**); params count | `eval/benchmark.py` | _unclaimed_ | Latency/memory columns in table | B3 |
| **B6** | Generate `eval_P4_T<teamID>.md`: metrics in PDF order (RMSE/MAE → PSNR/SSIM → edge/boundary → per-factor → latency/memory), **line 1 = comma-separated USNs** | `eval_P4_*.md` | _unclaimed_ | File named correctly, USN header, metric order matches PDF | B3, F6 |

### 🧠 Track C — Deep Learning + App + Hosting (P3, Colab GPU)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **C1** | Unified U-Net (~2M params): [depth, validity mask, RGB] → dense depth; SR + completion in one model | `deep/model.py` | _unclaimed_ | 256²→256² forward passes; ≤3M params | F1 |
| **C2** | Losses: masked L1 + multi-scale Sobel gradient + optional SILog; edge-weighted variant | `deep/losses.py` | _unclaimed_ | Loss decreases on 10-patch overfit | C1 |
| **C3** | Colab training notebook: 256² patches, flips, RGB jitter, AMP, AdamW, cosine, checkpoints, val RMSE | `notebooks/colab_train.ipynb`, `deep/train.py` | _unclaimed_ | Run #1 val RMSE < bicubic | C1, C2, B1, F3 |
| **C4** | Fine-tune with randomized degradation (sparsity/noise/factor); early stop; export weights + `infer.py` | `deep/train.py`, `deep/infer.py` | _unclaimed_ | Fine-tuned beats run #1 on all factors; weights link in README | C3 |
| **C5** | **Gradio app scaffold Day 2 with classical methods (runs before DL exists)** | `app/gradio_app.py`, `cli.py` | _unclaimed_ | Local app serves classical methods on Day 2 | A1–A6 |
| **C6** | Wire DL + hybrid into app; add loading spinner, error states, phone+laptop responsive | `app/gradio_app.py` | _unclaimed_ | All methods selectable; error/loading states present | C4, D2 |
| **C7** | **Deploy to HF Spaces (public URL) Day 5**; verify <10s on CPU; confirm no secrets | deployment | _unclaimed_ | Public URL works from phone + laptop | C6, F5 |

### 🧬 Track D — Hybrid + Report (P1/P4)

| ID | Task | Files | Owner | Done when | Depends |
|----|------|-------|-------|-----------|---------|
| **D1** | Hybrid v1: guided-filter residual correction GF(residual = known − upsampled) anchored at valid pixels; depth-domain ⇒ no texture-copy | `hybrid/fuse.py` | _unclaimed_ | Boundary RMSE improves over uncorrected DL | A2, C4 |
| **D2** | Hybrid v2: confidence fusion (RGB-gradient confidence + local error at valid pixels) blends DL vs classical | `hybrid/fuse.py` | _unclaimed_ | Beats both parents on ≥2 of {RMSE, boundary-RMSE, edge-F1} | D1, A6 |
| **D3** | Ablations: no-RGB, no-edge-loss, no-hybrid, each classical component removed | `eval/ablation.py` | _unclaimed_ | Ablation table with deltas in README | B3, C4, D2 |
| **D4** | Robustness + unseen eval (654 test), noise/hole sweeps, failure analysis incl. **texture-copy case check** | README section | _unclaimed_ | Sweep plots + ≥3 failure cases (incl. texture-copy) | B3, C4 |
| **D5** | Final README: methods, tables, ablations, latency/memory, failure analysis, run steps, weights links | `README.md` | _unclaimed_ | Stranger reproduces results from clean clone | all |
| **D6** | Limitations + ethics section (depth privacy, deployment honesty, bias) | `README.md`, `docs/DESIGN.md` | _unclaimed_ | Ethics/limitations stated honestly | all |
| **D7** | 3-min demo script + viva prep: each member explains their part, metrics, a failure case | demo notes | _unclaimed_ | Dry-run demo; everyone answers their part | D5, D6 |

---

## 6. Phase Timeline

| Phase | Day | Gate (merged to master) |
|-------|-----|--------------------------|
| **P0 Foundation** | 0 | F1–F6 merged; design doc + diagram v1 + Dockerfile + hosting chosen |
| **P1 Build** | 1–2 | A1–A7, B1–B2, C1–C2 merged; **C5 app live locally (classical-only)** |
| **P2 Results** | 2–3 | C3 first Colab run; B3–B4 merged; full classical benchmark |
| **P3 Fusion** | 3–4 | C4 fine-tuned; D1–D2 merged; 4-way comparison |
| **P4 Ablate + Host** | 5 | D3–D4, **B6 eval file**, **C7 hosted URL live**, D6 ethics |
| **P5 Ship** | 6 | C6 final app, D5 README, D7 demo rehearsal, tag `v1.0` |
| Buffer | 7 | bugfix, URL re-verify, backup run |

## 7. Acceptance Criteria (rubric-mapped)

1. **Public URL** live, phone+laptop responsive, <10s, loading/error states (UI/UX 15).
2. **`eval_P4_T<teamID>.md`** with USN first line + metrics in PDF order (Evaluation 25).
3. **`docs/architecture.md`** matches code; **`docs/DESIGN.md`** all template sections; README reproduces results; limitations + ethics stated (Documentation 15).
4. Core features on **unseen** test split; proper train/val split; modular code; **bad input handled** (Implementation 20).
5. `uv run pytest` green; repo runs via **one command or Dockerfile**; **no secrets committed**.
6. Every `method × factor` cell filled; hybrid ≥ DL on boundary metrics; DL ≥ all classical on RMSE at 8× (PS4 body).
7. Demo rehearsal: each member explains their part + metrics + a failure case (Demo/Viva 25).

## 8. Open Items (fill in before Day 0 ends)

1. **Team ID** (e.g. `T01`) and **full USN list** → `configs/team.yaml` (F6).
2. **Official design-document template** — use if organizer provides one, else standard sections.
3. **Hosting platform** — recommended Hugging Face Spaces (free, Gradio-native, CPU-friendly).
4. **Dataset scope** — NYU primary (confirmed); add Middlebury Stereo for external SR eval? (recommended, tiny download).

## 9. Task Board Sync

- **Claim:** replace `_unclaimed_` with your handle, commit `P4-<id>: claim`, push.
- **Done:** the merged PR is the record. If scope changed, edit the row in the same PR.
- **Blockers:** open a GitHub issue titled `P4-<id>: blocker — <short>`.
