"""Compare original vs robust vs routed on NYU val, per condition, with route counts.

  $env:SGNET_CKPT_X4        = <original .pth>
  $env:SGNET_CKPT_X4_ROBUST = <runs/x4_realistic/best.pth>
  python -m scripts.eval_routed --root "C:\\Users\\raksh\\Downloads\\nyu_data\\nyu_data" --start 75 --n 75
"""
import argparse
import os
from collections import Counter

import numpy as np

from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from deep.routing import assess_input, deep_hr_depth_routed
from deep.inference import deep_hr_depth
from eval.metrics import compute_all_metrics

CONDS = {
    "clean": {},
    "holes0.1": {"hole_ratio": 0.1},
    "noise0.01": {"noise_std": 0.01},
    "noise0.005": {"noise_std": 0.005},
    "blocks2": {"n_block_holes": 2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--start", type=int, default=75)
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    orig = os.environ[f"SGNET_CKPT_X{a.scale}"]
    rob = os.environ[f"SGNET_CKPT_X{a.scale}_ROBUST"]
    print(f"{'condition':12s} {'original':>9s} {'robust':>9s} {'routed':>9s}  routed->robust/original")
    for name, kw in CONDS.items():
        ds = NYUv2Dataset(a.root, split="val", scale=a.scale, degradation=DegradationConfig(seed=0, **kw))
        r = {"original": [], "robust": [], "routed": []}
        routes = Counter()
        for i in range(a.start, min(a.start + a.n, len(ds))):
            s = ds[i]
            lr, rgb, m = s["lr_depth"], s["hr_rgb"], s["lr_valid_mask"]   # ORIGINAL mask, pre-fill
            info = assess_input(lr, m)
            routes[info["route"]] += 1
            preds = {
                "original": deep_hr_depth(lr, rgb, m, a.scale, ckpt_path=orig),
                "robust": deep_hr_depth(lr, rgb, m, a.scale, ckpt_path=rob),
            }
            # Use deep_hr_depth_routed for the routed prediction to get per-call metadata
            # (both env-vars must be set; _ckpt_for will fail clearly if either is missing)
            routed_pred, route_info = deep_hr_depth_routed(lr, rgb, m, a.scale)
            preds["routed"] = routed_pred
            for k, p in preds.items():
                r[k].append(compute_all_metrics(p, s["depth_original"], data_range=s["data_range"],
                                                valid=s["valid_mask_original"])["rmse"])
        print(f"{name:12s} {np.mean(r['original'])*100:8.2f}c {np.mean(r['robust'])*100:8.2f}c "
              f"{np.mean(r['routed'])*100:8.2f}c  {routes['robust']}/{routes['original']}")


if __name__ == "__main__":
    main()