# SGNet fine-tuning add-on (Rakshitha)

Copy `deep/finetune.py` into the existing project's `deep` folder. No existing
files or teammates' modules need replacement. Use the already activated uv
virtual environment. No extra packages are needed beyond requirements_deep.txt.

## 1. GPU training smoke check

From the repository root:

```powershell
python -m deep.finetune --scope head --epochs 1 --steps-per-epoch 5 --val-samples 3 --out runs/x4_smoke
```

This performs actual gradient updates on the final reconstruction convolutions,
starting from your released x4 checkpoint. All other weights stay frozen.
128x128 HR patches, batch size 1, FP32 and a learning rate of 1e-5 are defaults.
This is a deliberately small adaptation check, NOT a final accuracy experiment.
It first measures pretrained validation RMSE, then trains, then validates again.
The train arrays are memory mapped; entire datasets are not loaded into RAM.
A CPU test cannot establish whether this fits your particular GPU; run this first.

## 2. Initial adaptation experiment

```powershell
python -m deep.finetune --scope head --epochs 3 --steps-per-epoch 200 --val-samples 30 --out runs/x4_head_adapt
```

Here an epoch means 200 sampled patch batches, NOT a full pass through NYU.
Record this fact in your report. Training patches come only from the NYU train
files, validation only from the existing val subset; test is never opened.
Normalized training L1 and validation RMSE in metres are different quantities.
The existing loader has no train_minmax.npy in the standard bundle: training
operates in normalized units, with target and input normalized identically from
valid LR extrema. Do not describe train loss as metres.

## 3. Optional full-model fine-tuning

After the smoke check, if time/memory allow:

```powershell
python -m deep.finetune --scope all --epochs 3 --steps-per-epoch 200 --val-samples 30 --out runs/x4_full_adapt
```

This unfreezes all model weights and consumes more VRAM. It is not required to
call the head-only experiment partial model fine-tuning. If OOM occurs, use
--patch 64 --batch-size 1, --scope head and a NEW output folder. The source
checkpoint is never overwritten. Fresh output directories are required; resume
is not implemented. FP16 is not enabled because of the Fourier blocks.

## 4. Evaluate adapted versus original weights

Run both commands on the SAME validation subset with separate output folders:

```powershell
$env:SGNET_CKPT_X4 = (Resolve-Path checkpoints/sgnet/SGNet_X4_R.pth).Path
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic deep --max-samples 30 --out results/pretrained_x4
$env:SGNET_CKPT_X4 = (Resolve-Path runs/x4_head_adapt/last.pth).Path
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic deep --max-samples 30 --out results/adapted_x4
Remove-Item Env:SGNET_CKPT_X4
```

Use your actual experiment folder if different. `last.pth` always contains the
adapted weights after the last epoch. `best.pth` is the best validation model
INCLUDING the original pretrained baseline: if no epoch improves, it retains
the baseline. `summary.json` records `best_epoch=0` in that case. Do not claim
best.pth is fine-tuned when best_epoch is zero. Report degradation honestly.
Use best.pth for integration only after checking summary.json.

Outputs: config.json, history.csv, summary.json, best.pth and last.pth.
Do not commit datasets, run checkpoints or downloaded weights to Git.

## Limits and reporting

- This is a project-specific adaptation script, not a reproduction of the
  authors' complete training procedure. It uses masked reconstruction L1 only;
  the auxiliary gradient output is not separately supervised.
- A checkpoint trained previously on NYU may not improve with more NYU training.
- The existing val/test split is image-disjoint, not proven scene-disjoint.
- Your 30 validation images have already been used for development. They are
  not an untouched final test. Freeze settings before final held-out evaluation.
- The initial implementation was checked with synthetic fixtures on CPU;
  no official checkpoint/dataset or RTX 5050 was available in the authoring
  environment. Your smoke run supplies the actual GPU feasibility evidence.
