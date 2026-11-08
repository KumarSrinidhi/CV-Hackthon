"""Drop into deep/. Fine-tune SGNet on a RANDOM MIX of realistic input corruptions.

    python -m deep.finetune_realistic --ckpt checkpoints/sgnet/SGNet_X4_R.pth --scope all \
        --lr 2e-5 --epochs 6 --steps-per-epoch 400 --val-samples 30 --out runs/x4_realistic

Reuses deep.finetune_robust unchanged (same loop, validation, saving); only swaps the
training-data corruption: noise + random holes + block holes + varying downsample operator.
"""
from __future__ import annotations

import numpy as np

import deep.finetune_robust as fr
from data.degradation import DegradationConfig, degrade


class RealisticTrainingData(fr.MixedTrainingData):
    P_CLEAN = 0.2
    MODES = ("bicubic", "bilinear", "nearest")
    MODE_P = (0.6, 0.2, 0.2)

    def __getitem__(self, index):
        sample = self.base[index]
        r = self.rng
        if r.random() < self.P_CLEAN:
            return sample
        cfg = DegradationConfig(
            mode=str(r.choice(self.MODES, p=self.MODE_P)),
            noise_std=float(r.uniform(0.0, 0.03)) if r.random() < 0.6 else 0.0,  # metres
            hole_ratio=float(r.uniform(0.0, 0.3)) if r.random() < 0.7 else 0.0,
            n_block_holes=int(r.integers(0, 3)) if r.random() < 0.4 else 0,
            block_frac=float(r.uniform(0.05, 0.2)),
            seed=int(r.integers(0, 2**30)),
        )
        deg = degrade(sample["depth_original"], sample["scale"], cfg,
                      valid_hr=sample["valid_mask_original"], sample_index=index)
        sample = dict(sample)
        sample["lr_depth"] = deg["lr_depth"]
        sample["lr_valid_mask"] = deg["lr_valid_mask"]
        return sample


if __name__ == "__main__":
    fr.MixedTrainingData = RealisticTrainingData   # main() looks the name up at call time
    fr.main()
