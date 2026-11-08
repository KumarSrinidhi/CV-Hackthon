# RESEARCH_REFERENCE_ANALYSIS — Six reference repositories

> Phase-2 deliverable. All six repos were cloned (shallow, `--depth 1`) into
> `tmp/` — **`tmp/` is gitignored and must not be committed**. Findings below are
> cited to file paths *inside those clones*. Where a claim comes from the linked
> paper rather than repo code (only the SpAgNet/WACV-2023 paper), it is labelled
> **[paper, external]**.

---

## 1. Scope and method

| # | Clone | Origin | Type |
|---|---|---|---|
| 1 | `tmp/sparse-to-dense` | fangchangma/sparse-to-dense (ICRA 2018) | training-pipeline reference (Torch) |
| 2 | `tmp/Sparse-Depth-Completion` | wvangansbeke/Sparse-Depth-Completion (MVA 2019) | two-branch + learned confidence |
| 3 | `tmp/PENet_ICRA2021` | JUGGHM/PENet_ICRA2021 (ICRA 2021) | SOTA KITTI, CSPN++ refinement |
| 4 | `tmp/CFCNet` | choyingw/CFCNet (NeurIPS 2019, "Deep RGB-D CCA") | mask-aware encoders + CCA loss |
| 5 | `tmp/sparsity-agnostic-depth-completion` | andreaconti/… (WACV 2023, SpAgNet) | **eval-only** companion + hole-pattern masks |
| 6 | `tmp/awesome-state-of-depth-completion` | alexklwong/… | curated leaderboard/resource list |

Each repo was read at README + core model/loss/data/metrics level by direct
inspection; every claim below carries the file path it came from.

**Important corrections to common assumptions about these repos:**

- `sparse-to-dense` and `Sparse-Depth-Completion` contain **no** guided filter,
  bilateral filter, sparsity-invariant convolution or post-processing at all
  (grep-verified; `Sparse-Depth-Completion/README.md:28` states "without …
  postprocessing"). SICN/Uhrig-style masked convs are *not* in these clones.
- `sparsity-agnostic-depth-completion` has **no model code** — it ships only
  evaluation scripts, metric harness and reusable hole-pattern masks
  (`patterns/*.npy`), plus the linked WACV 2023 paper **[paper, external]**.
- `CFCNet` is **not** a colorization repo: it is *Deep RGB-D Canonical
  Correlation Analysis for Sparse Depth Completion* (NeurIPS 2019,
  `CFCNet/README.md:1-4,73-79`).
- The awesome list ranks on **all four** metrics (MAE/RMSE/iMAE/iRMSE), never
  RMSE alone (`awesome-…/README.md:28`), and has no edge/SSIM/latency columns.

---

## 2. Repo-by-repo findings

### 2.1 `tmp/sparse-to-dense` — variable-density training done right

- **What:** monolithic FCN (ResNet-18/50 + up-projection decoder) that takes
  RGB + sparse depth as **early-fused channels** and predicts dense depth from
  an arbitrary number of random samples (`README.md:10-13`; `models/resnet-nyudepthv2.lua:120-184`).
- **Missing data is handled in the data pipeline, not the network:** Bernoulli
  thinning of GT re-drawn on **every** `get()` at `prob = nSample/nValid`
  (`datasets/kitti.lua:27-38`), so one model covers 20–500 points. Input holes
  are zeros; **target** holes become NaN and are excluded from loss and metrics
  with renormalization `sum · nElement/nValidElement`
  (`criteria/myAbsCriterion.lua:31-42`; `utils.lua:41-63`).
- **Loss:** default L1; **berHu** available with `c = 0.2·max|err|`
  (`criteria/BerHuCriterion.lua:33-66`) — its quadratic tail tracks RMSE ranking
  better than L1.
- **Augmentation discipline:** one shared random draw (rotation angle, crop,
  flip) for RGB/depth/GT (`datasets/kitti.lua:53-80`); rotation of sparse depth
  deliberately uses nearest-ish interpolation because *"bilinear interpolation
  creates a problem when the input depth image is sparse"*
  (`datasets/transforms.lua:201-210`).
- **Eval discipline:** test batch = 1; RMSE pooled over a batch >1 is *overestinated*
  (`utils.lua:162-164`); model selection = lowest val RMSE (`main.lua:99-110`).
- **`ChannelDropout(p=0.5)`** on input channels (`modules/ChannelDropout.lua:28-52`,
  wired in `models/encoders.lua:26-31`) forces an RGB-only fallback path.

### 2.2 `tmp/Sparse-Depth-Completion` — the confidence-fusion template

- **What:** two-branch (global ERFNet + local hourglass) with **learned,
  softmax-competitive per-pixel confidence** late fusion
  (`Models/model.py:44-75`): `out = w_g·precise_depth + w_l·lidar_to_depth`,
  `w_g + w_l = 1` — exactly the structure our hybrid's fixed α is a degenerate
  special case of.
- **Confidence trained without labels** via heteroscedastic loss
  `0.5·e²·exp(−|c|) + 0.5·|c|` (`Loss/loss.py:57-67`).
- **Deep supervision on every branch:**
  `loss = 1.0·L_fused + 0.1·L_local + 0.1·L_global_depth + 0.1·L_proxy`
  (`main.py:90-93,297-301`) — the direct counter to "only the fused output gets
  gradient, so the weak branch never improves".
- **Multi-stage fusion:** the local branch consumes the global branch's
  intermediate decoder features at two scales (`Models/model.py:65-67`), not
  just the final concat.
- **Metrics:** per-image RMSE/MAE over `gt>0`, predictions clamped to
  `[0, max_depth=85]` first (`Loss/benchmark_metrics.py:17-35`).
- **Latency protocol:** `torch.cuda.synchronize()` bracket + drop first 100
  timings (`Test/test.py:136-141,159`).

### 2.3 `tmp/PENet_ICRA2021` — measurement re-injection and guided propagation

- **What:** KITTI leaderboard winner at submission; two-branch ENet where the
  RGB branch's **predicted depth** feeds the depth branch, plus a
  **dilated CSPN++** refinement head that predicts per-pixel 3×5×7 propagation
  kernels from RGB guidance (`README.md:39-57`; `model.py:3-234,236-740`).
- **Hard measurement preservation:** after *every* propagation step,
  `depth = mask·d + (1−mask)·depth` with `mask = sigmoid(pred)·valid`
  (`model.py:297-300,338-344,502-506`) — i.e. measurement consistency, but
  spatially varying and learned rather than a global damping constant.
- **Never average zeros into a pyramid:** `SparseDownSampleClose` encodes
  invalid as `−600` before max-pool so coarse levels keep the nearest valid
  measurement and carry the mask (`basic.py:141-153`, used `model.py:108-112`).
- **Learned joint-bilateral filter:** `CSPNGenerateAccelerate` emits
  sum-normalized kernels (center = 1−sum) applied in one `unfold`+`einsum`
  (`basic.py:220-268`) — a neural guided filter, ~40 lines, CPU-viable.
- **Confidence softmax fusion** of the two branches' (depth, conf) heads
  (`model.py:175-177,220-224`).
- **Honest latency numbers with `torch.cuda.synchronize()`**: PENet 0.161 s,
  ENet-only 0.064 s (`README.md:1-18`) — the paper's own numbers were off by 5×.
- **Losses:** masked MSE/L1 (`criteria.py:6-29`); deep supervision with
  decaying weights 0.2→0.05→0 over the first 3 epochs (`main.py:224-245`).

### 2.4 `tmp/CFCNet` — explicit hole encoding and mask-normalized convs

- **What:** dual RGB/depth encoders + a third **missing-data encoder**
  (`rgb·(1−mask)` through shared RGB weights) so the network explicitly sees
  *where the holes are*; RGB and depth bottlenecks aligned by a 2-D CCA loss
  (`models/DCCA_sparse_networks.py:184-317,365-428`; input packing
  `dataloaders/dataloader.py:100-106,132-133` = `[rgb, d, rgb·mask, mask]`).
- **SAConv — mask-normalized convolution:** `conv(x·m)` rescaled by
  `9/sum(m)` over a fixed 3×3 ones kernel, mask max-pooled alongside
  (`DCCA_sparse_networks.py:84-124`) so sparse depth is not diluted by zeros.
- **Loss:** `MaskedMSE + DCCA + TransformLoss + 0.1·SmoothLoss`
  (`DCCA_sparse_model.py:73-78`); SmoothLoss = second-order gradient penalty
  (`DCCA_sparse_networks.py:341-363`).
- **Sparsity simulators worth stealing:** `UniformSampling`,
  **`SimulatedStereo`** (blur→Sobel→percentile threshold→dilate — an
  *edge-biased* hole pattern) and ORB-keypoint sampling
  (`dataloaders/dense_to_sparse.py:19-122`).
- All depth-augmentation interpolation is **nearest** so zeros stay zeros
  (`dataloaders/transforms.py:308-309,336-340`).
- Known code quirk: stage-3 pooling uses `pool4` indices
  (`DCCA_sparse_networks.py:252,297`) — do not copy blindly.

### 2.5 `tmp/sparsity-agnostic-depth-completion` — the eval harness + the anti-lesson

- **What:** eval-only companion to SpAgNet (WACV 2023). Ships `evaluate.py`,
  `hubconf.py`, a Streamlit viewer and **reusable boolean hole patterns**
  (`patterns/grid_mask_{320,640}.npy`, `patterns/livox_mask_*.npy` incl. a
  45-frame rotating variant) — drop-in hole generators for our synthetic
  pipeline; metric harness computes RMSE/MAE/REL under `gt>0` and
  `gt<=max_depth` (`evaluate.py:38-49,64-76`).
- **[paper, external] SpAgNet's core idea:** keep convolutions **sparsity-blind**
  (RGB-only ResNeXt encoder); inject measurements only via iterative
  **Scale-and-Place** modules — closed-form confidence-weighted linear
  regression `p→α+βp` fit on valid hints, then hard overwrite (`C=1`) exactly
  at hint pixels; sparsity-aware pooling averages **valid pixels only**; loss =
  per-scale confidence-weighted L1+L2 with exponential decay `γ=0.4` minus
  `η·ln C` (η=0.1 NYU / 20 KITTI).
- **The anti-lesson (§4.2 of the paper):** kernels that *consume* the sparsity
  pattern (Minkowski-style sparse convs) overfit one density and fail at unseen
  ones. The paper's own ablation: removing confidence+scale degrades 5-point
  NYU RMSE 0.467 → 0.770 m.
- **Evaluation across a density sweep**, not one operating point: NYU densities
  `5|50|100|200|500|livox|grid-shift`, KITTI `lines4…lines64`
  (`hubconf.py:51,61`).

### 2.6 `tmp/awesome-state-of-depth-completion` — the map of the field

- Curated VOID + KITTI leaderboards (supervised/unsupervised), ranked on all
  four MAE/RMSE/iMAE/iRMSE metrics (`README.md:26-28`).
- Most relevant entries for us (verbatim from `README.md`):
  - **ip_basic** — *In Defense of Classical Image Processing: Fast Depth
    Completion on the CPU* (CRV 2018, line 70) → the classical baseline family
    we compete with.
  - **Semantically Guided Depth Upsampling** (GCPR 2016, line 74) → nearest
    published analogue of our JBU/guided branch.
  - **NConv / PnCnn** (PAMI 2019 / CVPR 2020, lines 109/114) →
    confidence propagation through networks — the confidence machinery our
    hybrid fusion needs.
  - **Sparse-Depth-Completion** (MVA 2019, line 101) = repo 2 above.
  - **BP-Net** (CVPR 2024, line 88, best supervised-KITTI RMSE 684.90) →
    bilateral (edge-aware) propagation = the classical bilateral idea learned.
  - **DFU** (CVPR 2024, line 81) → feature-level rather than depth-level
    upsampling.
  - **CompletionFormer** (CVPR 2023, lines 80/92, best KITTI MAE) →
    conv+transformer hybrid.
  - **Sparsity Invariant CNNs** (3DV 2017, line 118) → origin of KITTI's
    hole-encoding baseline (not present in any of our clones).

---

## 3. Cross-cutting findings

1. **Every modern method replaces fixed fusion weights with per-pixel,
   softmax-normalized, learned confidence** (repo 2 `model.py:72-75`,
   repo 3 `model.py:220-224`, SpAgNet Place). Our frozen α=0.02 is the
   known-degenerate case.
2. **Invalid pixels must never enter any aggregation** — losses mask+renormalize
   (repos 1, 2, 3, 4), pyramids use nearest-valid/max-pool with a carried mask
   (repo 3 `basic.py:141-153`), depth resampling uses nearest so zeros stay
   zeros (repos 1, 4), metrics clamp to `[0, max_depth]` and mask `gt>0`
   (repos 2, 3, 5).
3. **Measurement values are re-injected/hard-overwritten at measured pixels**
   (repo 3 every CSPN step; SpAgNet Place `C=1`) — a *derived* rule, not a
   hand-tuned global damping factor like our current consistency strength 0.25.
4. **Train across the density/noise distribution, never at one fixed hole
   ratio** (repo 1 Bernoulli-per-sample; repo 4 SimulatedStereo/UniformSampling;
   repo 5 density sweeps). This is precisely our measured failure: deep < bicubic
   at 10 % holes.
5. **Edge preservation comes from RGB-guided, spatially varying propagation**
   (repo 3 CSPN++, repo 6's BP-Net/GuideNet entries), not from a fixed-sigma
   bilateral — though a fixed JBU is our CPU baseline.
6. **Eval hygiene is part of the method:** per-image metrics then average,
   batch-1 at test, `torch.cuda.synchronize()` + warm-up discard for latency,
   clamp before scoring, selection on a fixed val subset's RMSE (repos 1, 2, 3).
7. **None of the six contains PSNR/SSIM/edge-F1 or 2×/4×/8× scaling** — those
   stay our own implementation, per the official P4 requirements.

---

## 4. Synthesis — what we adopt (prioritized)

| # | Adoption | Source (clone path) | Target in our code | Addresses |
|---|---|---|---|---|
| A1 | **Per-pixel softmax-competitive confidence** replacing fixed α; two 1-channel heads, weights forced to sum to 1 | `Sparse-Depth-Completion/Models/model.py:72-75`; `PENet_ICRA2021/model.py:220-224` | `hybrid/confidence.py`, `hybrid/fusion.py` | hybrid gain marginal |
| A2 | **Hole-band confidence**: classical gains weight in the dilated band around missing LR pixels (already prototyped in `stash@{0}`) + SpAgNet-style hard overwrite `C=1` at measured pixels | `stash@{0}`; [paper, external] Place | `hybrid/confidence.py` (merge stash) | deep corrupted by nearest-fill; consistency derived not tuned |
| A3 | **Mask-normalized input preprocessing for the deep branch**: `conv(d,ones3×3)/conv(mask,ones3×3)` + explicit mask channel, instead of raw nearest-fill | `CFCNet/models/DCCA_sparse_networks.py:84-124`; input packing `dataloader.py:100-106` | `deep/inference.py` (pre-post) or finetune input packing | deep < bicubic at 10 % holes |
| A4 | **Never average zeros**: nearest-valid downsample with carried mask for every pyramid/GT construction; nearest interpolation for any mask resampling | `PENet_ICRA2021/basic.py:141-153`; `sparse-to-dense/datasets/transforms.lua:201-210` | `data/degradation.py`, `data/preprocessing.py` | holes/noise robustness, edge smearing |
| A5 | **Metrics clamp to `[0, max_depth]` and mask `gt>0`, per-image then average** | `Sparse-Depth-Completion/Loss/benchmark_metrics.py:17-35` | `eval/metrics.py` | classical RMSE gap may be partly unclamped blow-up |
| A6 | **berHu (or L1+quadratic tail) for RMSE-aligned losses**; NaN/0-aware mask+renormalize in any loss we write | `sparse-to-dense/criteria/BerHuCriterion.lua:33-66`; `myAbsCriterion.lua:31-42` | future finetune losses (Kaggle) | RMSE optimization |
| A7 | **Density-jittered training + edge-biased hole simulation** (`SimulatedStereo`) and Bernoulli per-sample thinning | `CFCNet/dataloaders/dense_to_sparse.py:19-122`; `sparse-to-dense/datasets/kitti.lua:27-38` | `data/degradation.py` (add SimulatedStereo-style pattern), Kaggle finetune | deep brittleness at unseen densities |
| A8 | **Learned guided propagation (CSPN-style 3×3 kernels, 1–3 steps, optionally dilated for 8×)** as a classical/hybrid CPU refinement | `PENet_ICRA2021/basic.py:220-268` | new `classical/guided_propagation.py` | boundary/edge-F1, classical RMSE |
| A9 | **Re-inject measurements per-pixel with a reliability gate** instead of global damping | `PENet_ICRA2021/model.py:338-344` | `hybrid/consistency.py` | consistency step correctness |
| A10 | **Eval harness discipline**: batch-1 per-image metrics, latency with sync+warm-up, clamp-then-score, density sweep with `patterns/*.npy` masks | `sparse-to-dense/utils.lua:162-164`; `Sparse-Depth-Completion/Test/test.py:136-141`; `sparsity-agnostic-…/evaluate.py:64-76` + `patterns/` | `eval/evaluate.py`, Kaggle notebook | metric comparability, required density sweep |
| A11 | **iMAE/iRMSE (inverse-depth) as extra metrics** | `awesome-…/README.md` tables | `eval/metrics.py` | far-field/boundary errors RMSE hides |
| A12 | **`stash@{0}` hole-band fusion + tests** (our own prior WIP, corroborated by A1/A2) | `stash@{0}` on `feat/combined-hybrid` | `hybrid/*`, `tests/test_hybrid.py` | hybrid improvement |

**Explicitly NOT adopted**

- Sparse/adaptive convolutions that consume the sparsity pattern (SpAgNet §4.2
  evidence they overfit one density; also absent from our codebase's design).
- Full DCCA loss (O(m) `symeig`, `.cuda()`-only — `CFCNet/models/DCCA_sparse_networks.py:387-389`).
- Repo-1-style monolithic RGBD early-fusion retrain (we keep SGNet; input-side
  changes only, A3/A7).
- Any post-processing claim from repos 1/2 — they have none; our classical branch
  remains the only place edge-aware filtering lives (plus A8's new propagation).

---

## 5. Gaps the references do NOT solve (our own work)

1. **2×/4×/8× multi-scale protocol** — none of the six does scale-factor SR;
   only CSPN dilation and nearest-valid pyramids transfer (A4, A8).
2. **Classical-vs-bicubic RMSE deficit** — references contain no classical
   upsample method to copy (verified by grep); the fix is A5 (clamping/metric
   hygiene) + A8 (guided propagation) + re-tuning, measured by us.
3. **PSNR/SSIM/edge-F1/latency/memory** per the P4 rubric — only RMSE/MAE/δ/REL
   exist in the references; ours already implements the required set (A10/A11
   extend it).
4. **CPU-only constraint** — everything here is GPU-trained; our local
   verification remains smoke tests only (no training, no full-dataset runs).
5. **Confidence calibration honesty** — references call their weights
   "confidence" but none are calibrated probabilities either; we keep the
   README's existing caveat.

---

*End of Phase 2. Next: Phase 3 — branch strategy + research-driven
implementation on `feat/research-driven-depth-completion`.*
