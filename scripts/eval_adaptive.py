# Team Owner: Kumar
# Workstream: Deep Learning / Evaluation
# Module: scripts/eval_adaptive.py
# Description: Evaluate the regime-aware adaptive deep method (deep/adaptive.py) with the
# official harness: clean HF benchmark and/or NYU val degraded regimes.
"""Examples:
    python scripts/eval_adaptive.py --tars datasets/val-000000.tar --max-samples 100 --out results/e8_hf_clean100
    python scripts/eval_adaptive.py --nyu-val --out results/e8_val150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from deep.adaptive import deep_hr_depth_adaptive, detect_regime
from eval.evaluate import evaluate_methods, save_results
from scripts.eval_all3_hf import HFNYUValDataset

METHODS = ["nearest", "bilinear", "bicubic", "deep", "deep_adaptive", "classical", "hybrid"]


def run(dataset, out, tag, methods):
    def adaptive(lr, rgb, mask, scale):
        return deep_hr_depth_adaptive(lr, rgb, mask, scale)

    res = evaluate_methods(dataset, methods, methods={"deep_adaptive": adaptive})
    paths = save_results(res, out, tag)
    from eval.evaluate import summary_markdown
    print(summary_markdown(res))
    print("saved:", ", ".join(str(p) for p in paths.values()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="*", default=None)
    ap.add_argument("--nyu-val", action="store_true")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--methods", nargs="+",
                    default=["bicubic", "deep", "deep_adaptive"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.tars:
        ds = HFNYUValDataset(a.scale, DegradationConfig(), tar_paths=a.tars)
        run(ds, a.out, f"hf_nyu_val_x{a.scale}", a.methods)
    if a.nyu_val:
        root = "nyu_data/nyu_data"
        for name, extra, tag in [
            ("clean", {}, "nyu_val_x4"),
            ("holes", {"hole_ratio": 0.1}, "nyu_val_x4_holes"),
            ("noise", {"noise_std": 0.02}, "nyu_val_x4_noisy"),
            ("noise_strong", {"noise_std": 0.05}, "nyu_val_x4_noisy"),
        ]:
            ds = NYUv2Dataset(root=root, split="val", scale=a.scale,
                              degradation=DegradationConfig(**extra))
            run(ds, f"{a.out}_{name}", tag, a.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
