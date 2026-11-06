# Missing-depth adaptation

Add finetune_robust.py to your existing deep folder. Requires your original
DL/data/eval package. Does not replace finetune.py or any teammates' files.

Smoke run (actual gradient updates, synthetic missing pixels on real NYU train):

```powershell
python -m deep.finetune_robust --ckpt runs/x4_head_adapt/best.pth --epochs 1 --steps-per-epoch 5 --val-samples 3 --out runs/x4_robust_smoke
```

Then, if the smoke run fits your GPU:

```powershell
python -m deep.finetune_robust --ckpt runs/x4_head_adapt/best.pth --epochs 3 --steps-per-epoch 200 --val-samples 30 --out runs/x4_robust
```

Starts from your clean fine-tuned x4 checkpoint. Half the sampled training
examples are clean, half have random 10% LR pixel dropout. Masked HR ground
truth is never corrupted. Dropout pattern varies across draws. Frozen backbone,
trainable final convolutions, 128x128 HR patches, batch 1, FP32, LR 1e-5.
An epoch is a fixed number of sampled patch updates, not a full dataset pass.

Each validation runs both clean and fixed-seed dropout conditions on the SAME
validation images. best.pth minimizes their equally weighted mean RMSE in metres,
including the starting checkpoint. history.csv records both component metrics.
This selection can trade clean accuracy for hole accuracy: inspect both rather
than claiming universal improvement. summary.json best_epoch=0 means no update
beat the starting weights; in that case best.pth is the starting checkpoint.
last.pth always contains the last adapted epoch. Config and logs record the run.
Training does not access test images; scene separation in existing image-based
splits is not verified. This does not establish sparse LiDAR completion support.

Evaluate the chosen checkpoint on both conditions with separate result folders:

```powershell
$env:SGNET_CKPT_X4 = (Resolve-Path runs/x4_robust/best.pth).Path
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic deep --max-samples 30 --out results/robust_clean
python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic deep --max-samples 30 --hole-ratio 0.1 --out results/robust_holes
Remove-Item Env:SGNET_CKPT_X4
```

Compare with existing clean and missing-depth runs. Do not tune on the final
test subset. No improvement is guaranteed. If head-only adaptation is ineffective,
full-model adaptation is available with --scope all, but first smoke-test VRAM.
If OOM, reduce --patch to 64 and choose a NEW --out directory. No resume support.
CPU synthetic training/validation checked during authoring; real data/GPU run
must be verified on your laptop. Baseline validation is done twice (clean and
holes), so this is slower than the previous clean-only experiment.
