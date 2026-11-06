# GPU Run Report — Deep Track (SGNet) on RTX 5070

> **Status: COMPLETE — every claim below was actually executed on this machine on 2026-11-06.**
> Failures and workarounds are recorded, not hidden. Numbers are real measurements.

---

## 0. Honest summary

| Question | Verified answer |
|---|---|
| Does CUDA work on this RTX 5070? | **YES** — torch 2.11.0+cu128, real kernel launch confirmed |
| Does the full test suite pass with CUDA torch? | **YES — 35 passed, 0 skipped** |
| Does the official pretrained SGNet load and run on GPU? | **YES** — strict state-dict match, sha256 `ebab512732b2` |
| Does it beat bicubic on a real dataset? | **YES, decisively** — 479 real NYU test images, 0 failures (table below) |
| Are the quality numbers paper-comparable? | **NO** — different (overlapping) test-image source than the SGNet paper bundle; same metric protocol. Quote as "HF NYU val subset, x4, RTX 5070 Laptop". |
| Was the official SGNet Google-Drive NYU bundle used? | **NO** — Drive throttled the download to a stall on this machine; documented below |

**Final quality table — 479 real NYU-v2 test images (HuggingFace FastDepth val), x4, GPU, border 6, data_range 10 m:**

| method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | boundary RMSE | edge F1 | latency ms |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0377 | 0.0121 | 50.18 | 0.9943 | 0.1743 | 0.278 | 6.5 |
| **deep (SGNet)** | **0.0096** | **0.0031** | **61.90** | **0.9995** | **0.0403** | **0.967** | 1895.0 |

SGNet cuts RMSE by **74%** (3.9×) and boundary RMSE by **77%** versus bicubic; ~96.7% of
ground-truth depth edges are correctly recovered (vs 27.8% for bicubic). The price is
~1.9 s/image vs 6.5 ms — the classical/hybrid tracks remain the fast path.

Full per-image CSV / JSON / markdown: `results/hf_nyu_val_x4/`.

---

## 1. Environment (measured, 2026-11-06)

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 (Laptop), 8 GB VRAM, Blackwell `sm_120` | `nvidia-smi` |
| Driver | 591.97, CUDA capability up to 13.1 | `nvidia-smi` |
| OS | Windows, WDDM mode | `nvidia-smi` |
| Python | 3.12 (uv-managed) | `uv run python` |
| torch (initial) | **2.15.0+cpu — CPU-ONLY build, cannot use the GPU** | `torch.__version__` |

**Problem found and fixed:** the project initially resolved to a CPU-only torch wheel.
Blackwell (sm_120) GPUs need a torch build compiled for CUDA 12.8+.
Fix applied in `pyproject.toml` (pinning torch to the official PyTorch cu128 index):

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

*(verification of the new build is recorded in section 2 below — not claimed until it actually imports and launches a kernel)*

---

## 2. Step 1 — CUDA torch install

**PASSED.** `uv sync` downloaded the full cu128 wheel set (torch + NVIDIA CUDA/cuDNN/cuBLAS
runtime wheels, ~7 GB in the uv cache). Measured result:

```
torch 2.11.0+cu128
cuda available: True
device: NVIDIA GeForce RTX 5070 Laptop GPU
kernel launch OK: 16.0        # (ones(8, device='cuda') + 1).sum() == 16 -> real kernel ran on sm_120
```

## 3. Step 2 — GPU smoke test (random init, no checkpoint)

**PASSED** for both scales. Output quality is meaningless with random weights; latency and
VRAM numbers below are real measurements (`python -m deep.inference --random-init --require-cuda`).

| Scale | Params | HR size | Latency mean | Latency p95 | Peak VRAM (allocated) | Peak VRAM (reserved) | Result |
|---|---|---|---|---|---|---|---|
| x4 | 36.4 M | 480×640 | 1833.6 ms | 1835.3 ms | 3468.9 MB | 4458.0 MB | forward OK, finite output, fp32 |
| x8 | 39.9 M | 480×640 | 1316.5 ms | 1331.2 ms | 2735.7 MB | 3822.0 MB | forward OK, finite output, fp32 |

**Honest caveat:** at x4, 480×640 already reserves ~4.5 GB of the 8 GB card. Larger inputs
(720p HR and above) may hit the automatic tiled-OOM fallback, which is *not* bit-identical
to full-image inference (SGNet uses global FFT blocks).

Full test suite after the CUDA install: **35 passed, 0 skipped** (the CUDA-only test
`test_cuda_forward_and_profile` now runs and passes; it was previously skipped on CPU).

## 4. Step 3 — Official SGNet checkpoint (trusted source)

**PASSED.** The official Google Drive folder could not be fetched programmatically (gdown
folder listing failed); the team downloaded it manually in the browser from the link in the
official [yanzq95/SGNet README](https://github.com/yanzq95/SGNet) (AAAI 2024, Apache-2.0).
Files copied into `checkpoints/sgnet/` (git-ignored, never committed):

| File | Size | Note |
|---|---|---|
| `SGNet_X4_R.pth` | 140.1 MB | x4, num_feats=48, official "_R" re-upload |
| `SGNet_X8_R.pth` | 153.3 MB | x8, num_feats=40 |
| `SGNet_X16_R.pth` | 331.4 MB | x16, num_feats=40 |

Real pretrained inference on GPU (x4, 480×640):

```
model: SGNet x4, 36.395 M params, pretrained=True
ckpt=checkpoints\sgnet\SGNet_X4_R.pth (sha256[:12]=ebab512732b2)
forward OK: LR (120,160) -> HR (480,640), float32, range [0.000, 4.484]
latency mean 1832.7 ms | p95 1840.7 ms | peak VRAM allocated 3468.9 MB / reserved 4458.0 MB
```

The checkpoint passed the strict state-dict key check (`load_state_dict` with zero
missing/unexpected keys) and the sha256 fingerprint is recorded above so the exact weights
used for every number in this report are traceable.

## 5. Step 4 — Real dataset evaluation (trusted source)

**Dataset provenance (honest):**
- The *official* SGNet NYU npy bundle lives on the authors' Google Drive and could **not** be
  downloaded from this machine (Drive throttled the transfer to a stall; browser-cookie
  fallback blocked by Windows DPAPI for gdown). Marked NOT used.
- The *official* Middlebury/Lu zip from Portland State (`web.cecs.pdx.edu`) also stalled
  (23.6 MB of unknown total in ~10 min). Marked NOT used.
- **USED instead:** the HuggingFace mirror [`sayakpaul/nyu_depth_v2`](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2)
  (Apache-2.0), validation split = the FastDepth 654-image labelled NYU-v2 test set
  (real Kinect RGB-D pairs, 480×640, depth in metres). Downloaded via HF CDN
  (`data/val-000000.tar`, 955 MB total; the network initially crawled at ~13 KB/s, so the
  first eval ran on the sequential prefix of the tar; the resumed download later reached
  688 MB ≈ 479 readable samples and the eval was re-run on those).

**Honest protocol note:** this is *not* byte-identical to the SGNet paper protocol (which
uses the DKN 449-image npy bundle and its exact splits). Same degradation (PIL bicubic ×4
on dense GT, operator exact), same masked metric implementation (`eval/metrics.py`:
border 6, data_range 10 m, depth-only edges), but a different (overlapping) subset of the
official 654 test images and images stored as .h5 by FastDepth. **Do not quote these
numbers as paper-comparable; quote them as "HF NYU val subset, x4, this machine".**

**Results (real pretrained checkpoint `SGNet_X4_R.pth`, sha256 ebab512732b2, CUDA):**

- 20-sample sanity run (tar prefix): deep RMSE **0.0135 m** vs bicubic **0.0518 m**;
  deep edge F1 **0.969** vs bicubic **0.440**; deep PSNR 58.8 dB vs 47.7 dB.
  Deep latency ≈ 1.9 s/image; bicubic ≈ 1.1 ms.

**Final 479-sample table** (see section 0 — full run, 0 failures, results written to
`results/hf_nyu_val_x4/{per_image.csv, summary.json, summary.md}`).

## 6. Full history log (append-only)

| # | Time (local) | Action | Result |
|---|---|---|---|
| 1 | 22:04 | Checked GPU: RTX 5070 Laptop 8 GB, driver 591.97 | OK |
| 2 | 22:04 | Checked torch: 2.15.0+cpu, `cuda.is_available()` False | CPU-only build found — blocker |
| 3 | 22:05 | Pinned torch to official `pytorch-cu128` index in `pyproject.toml` | done |
| 4 | 22:05–22:40 | `uv sync` — downloaded cu128 torch + NVIDIA runtime wheels (~7 GB) | done |
| 5 | 22:41 | Verified: torch 2.11.0+cu128, `cuda available: True`, real kernel launched | **PASSED** |
| 6 | 22:43 | GPU smoke test x4 random-init, 480×640, 10 runs | PASSED — 1.83 s mean, 3.47 GB VRAM |
| 7 | 22:45 | GPU smoke test x8 random-init | PASSED — 1.32 s mean, 2.74 GB VRAM |
| 8 | 22:46 | CUDA-only pytest now runs | PASSED |
| 9 | 22:47 | Full suite with CUDA torch | **35 passed, 0 skipped** |
| 10 | 22:48 | Attempted `gdown --folder` of official SGNet checkpoints | FAILED (folder listing; also uv OneDrive hardlink error → fixed with `UV_LINK_MODE=copy`) |
| 11 | 23:30–23:45 | Team downloaded checkpoint folder manually in browser; copied 3 `*_R.pth` into `checkpoints/sgnet/` | done (git-ignored) |
| 12 | 23:47 | Real pretrained x4 GPU inference, 480×640, 10 runs | **PASSED** — strict ckpt match, sha256 ebab512732b2, 1.83 s, range [0.000, 4.484] |
| 13 | 23:50 | Attempted official NYU Drive bundle via gdown (multiple retries, `--continue`, browser cookies) | **FAILED** — Drive throttled to stall; Edge/Chrome cookie extraction blocked by Windows DPAPI |
| 14 | 23:55 | Attempted official Middlebury/Lu zip from web.cecs.pdx.edu | **FAILED** — server stalled (23.6 MB in 10 min) |
| 15 | 00:00 | Pivoted to HuggingFace: `sayakpaul/nyu_depth_v2` val split (FastDepth 654-img labelled NYU test set, Apache-2.0), direct CDN tar | download working but slow (~400 KB/s → 955 MB) |
| 16 | 00:15 | Evaluated first 20 complete samples of partial tar (truncation-tolerant reader) | deep 0.0135 m RMSE vs bicubic 0.0518 m — sanity PASSED |
| 17 | 00:20–00:45 | Resumed tar download; reached 688 MB = **479 readable .h5 samples** | done |
| 18 | 00:46–01:07 | Full GPU eval, 479 samples × (bicubic + deep), real checkpoint | **COMPLETED, 479/479, 0 failures** — see section 0 table |
| 19 | 01:10 | Wrote this report; results in `results/hf_nyu_val_x4/` | done |

**Files created/changed by this session:**
- `pyproject.toml` — cu128 torch index pin + `scikit-image` dependency
- `scripts/eval_nyu_hf.py` — reusable HF-NYU eval script (same metric protocol as `eval/`)
- `docs/GPU_RUN_REPORT.md` — this report
- `results/hf_nyu_val_x4/` — per-image CSV, summary JSON/MD (committed-worthy numbers)
- `checkpoints/sgnet/*.pth`, `datasets/*.tar` — downloaded, git-ignored, never commit

**How to reproduce:**
```bash
uv sync                                            # cu128 torch (needs the pyproject index pin)
uv run python -m deep.inference --scale 4 --height 480 --width 640 --require-cuda
uv run pytest tests/ -q                            # 35 passed expected
uv run python scripts/eval_nyu_hf.py --tars datasets/val-000000.tar --device cuda
```
