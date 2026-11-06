# Deep branch, data pipeline, evaluation (Rakshitha)

Files: `deep/` (SGNet wrapper), `data/` (loaders, degradation, preprocessing), `eval/` (metrics, evaluate, benchmark), `tests/test_deep.py`.

## 1. One-time setup (RTX 5050)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # needs CUDA 12.8+ build for sm_120
pip install -r deep/requirements_deep.txt
python -m deep.inference --scale 4 --random-init --require-cuda          # GPU + architecture check, no weights needed
```
SGNet's README pins torch 2.1; that build has no sm_120 kernels. Use a cu128+ build instead (SGNet runs fine on it).

## 2. Files you must download yourself (not in the repo)
* Checkpoints: the official `*_R.pth` files from the SGNet README (Google Drive) -> `checkpoints/sgnet/` (x4, x8, x16). Never commit them.
* NYU-v2 npy bundle -> `datasets/nyu_data/` (`train_depth_split.npy, train_images_split.npy, test_depth.npy, test_images_v2.npy, test_minmax.npy`).
* Middlebury / Lu (optional) -> `datasets/Middlebury/`, `datasets/Lu/` (`output_color_*.png`, `output_depth_*.png`).
Override locations with `SGNET_CKPT_DIR`, `SGNET_CKPT_X4/X8/X16`, `P4_DATA_ROOT`, `NYU_ROOT`, `MIDDLEBURY_ROOT`, `LU_ROOT`.

## 3. Run order
```bash
python -m deep.inference --scale 4 --height 480 --width 640     # real forward pass + latency + peak VRAM (record these)
python -m deep.inference --scale 8 --height 480 --width 640
python -m pytest tests/test_deep.py -q                           # CPU, no downloads needed
python -m eval.evaluate  --dataset nyu --split val --scale 4 --max-samples 30
python -m eval.benchmark --dataset nyu --split val --scales 4 8 --methods bicubic deep classical hybrid
python -m eval.benchmark --dataset nyu --split test --scales 4 8 --final      # ONCE, after everyone froze configs
```

## 4. Frozen contract (all branches)
`f(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32 ndarray`
* `hr_rgb` (H,W,3) RGB, float [0,1] (uint8 accepted) · `lr_depth` (h,w) float32, native unit, invalid = 0 · `valid_mask` (h,w) bool **at LR resolution**, True = valid · `H = h*scale`.
* Units: NYU = metres, Middlebury/Lu = 0-255 grey levels. Never compare numbers across datasets.
* Originals are never modified; filled/normalised arrays are copies. Dataset samples keep `depth_original`, `valid_mask_original`, `lr_depth_clean` (= S(GT)); `operator_exact` says when `lr_depth == S(GT)` (no noise/holes), i.e. when Rishita's consistency step is allowed.
* Errors: `UnsupportedScaleError` (ValueError) for scales without a checkpoint, `DeepModelUnavailableError` (RuntimeError) for missing/mismatched checkpoint, CUDA problems, OOM. Kumar's UI should catch both.

## 5. Verified facts about SGNet (read from the official repo, Apache-2.0)
* Native call `net((rgb, lr_depth))`: rgb (N,3,H,W) [0,1]; depth (N,1,H/s,W/s) **not pre-upsampled**, per-image min-max normalised; output normalised HR depth. No mask channel -> invalid LR pixels are nearest-filled in a working copy before the model; the output is the raw prediction everywhere.
* One checkpoint per scale: x4 (num_feats 48, 36.4 M params), x8 (40, 39.9 M), x16 (40, 86.6 M). **No x2 checkpoint**: deep supports 4/8/16 only, x2 raises `UnsupportedScaleError`.
* Trained on synthetic NYU-v2 with bicubic LR. `deep/sgnet/` is a vendored copy; the only change is replacing hard-coded `.cuda()` (crashes on CPU / `.to()`) by buffers; checkpoint keys are unchanged.
* Normalisation uses valid LR min/max (deployable). The paper normalises with GT min/max, so expect small differences from paper RMSE. Compare to the paper on the full 449 images (`--split full --final`), border 6, and convert metres to cm.

## 6. Splits and rules enforced in code
* NYU `val` = first third of the 449 official test images, `test` = last block after an 8-image gap (no scene ids exist, so adjacent frames are separated). The SGNet checkpoints were trained on the NYU *train* split, so tuning on train-derived data would favour the deep branch: **tune everything on `val`**.
* `--split test/full` refuses to run without `--final`. Middlebury/Lu have no val split: tune on NYU val, report there as test only.
* One shared evaluation mask (GT validity, 6 px border), fixed `data_range` (NYU 10 m, Middlebury 255). Boundary metrics use **depth** discontinuities only (rel. threshold 0.03 of data_range), never RGB edges.
* Methods that are unavailable (stub, missing checkpoint, unsupported scale) are written as "not evaluated" with the reason.
* KITTI: not supported (`get_dataset('kitti')` raises). Sparse LiDAR completion is a different task and no SGNet-on-KITTI claim is made.

## 7. Known limits
* Tiled inference (`tile_lr=` or automatic on CUDA OOM) is a fallback and is not identical to full-image inference (SGNet has global FFT blocks). Large Middlebury images at x4 may need it on 8 GB.
* fp32 only (cuFFT half precision supports only power-of-two sizes).
* No fine-tuning script is included; released checkpoints are used as is.
