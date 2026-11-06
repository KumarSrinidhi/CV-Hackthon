"""Compare Deep with small classical contributions and post-processing stages.

The first group disables refinement and consistency and evaluates fixed blends
with 0%, 2%, 5%, and 10% classical contribution. The best blend is selected
by lowest validation RMSE, then compared with refinement and consistency added
one at a time (and both together).

Example:
    python scripts/eval_hybrid_ablation.py --split val --max-samples 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classical import classical_hr_depth
from data.dataset import get_dataset
from data.degradation import DegradationConfig
from deep.inference import deep_hr_depth
from eval.metrics import compute_all_metrics
from hybrid.boundary_refinement import refine_hybrid_depth
from hybrid.consistency import enforce_consistency
from hybrid.fusion import fuse_fixed


ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.20)
METRICS = ("rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_f1")


def _variants(
    classical, deep, rgb, lr, valid, scale, alpha, refinement, consistency,
    selective_refinement=False,
):
    fused = fuse_fixed(classical, deep, alpha)
    if refinement:
        fused = refine_hybrid_depth(fused, rgb, selective=selective_refinement)
    if consistency:
        fused = enforce_consistency(fused, lr, valid, scale, mode="bicubic")
    return fused


def _mean_metrics(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nyu")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/hybrid_ablation")
    args = parser.parse_args(argv)

    dataset = get_dataset(args.dataset, args.split, args.scale, args.root, DegradationConfig())
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    stage_names = ["deep"] + [f"classical_{int(alpha * 100):02d}pct" for alpha in ALPHAS]
    best_name = None
    per_variant = {name: [] for name in stage_names}
    samples = []
    predictions = []

    for index in range(n):
        sample = dataset[index]
        samples.append(sample)
        lr, rgb, valid, scale = (
            sample["lr_depth"].copy(),
            sample["hr_rgb"].copy(),
            sample["lr_valid_mask"].copy(),
            sample["scale"],
        )
        deep = deep_hr_depth(lr, rgb, valid, scale, device=args.device)
        classical = classical_hr_depth(lr, rgb, valid, scale)
        variants = {"deep": deep}
        variants.update(
            {
                f"classical_{int(alpha * 100):02d}pct": _variants(
                    classical, deep, rgb, lr, valid, scale, alpha, False, False
                )
                for alpha in ALPHAS
            }
        )
        predictions.append((classical, deep, variants))
        for name, prediction in variants.items():
            per_variant[name].append(
                compute_all_metrics(
                    prediction,
                    sample["depth_original"],
                    sample["data_range"],
                    sample["valid_mask_original"],
                )
            )
        print(f"  evaluated {index + 1}/{n}")

    summary = {name: _mean_metrics(rows) for name, rows in per_variant.items()}
    blend_names = [f"classical_{int(alpha * 100):02d}pct" for alpha in ALPHAS]
    best_name = min(blend_names, key=lambda name: summary[name]["rmse"])
    best_alpha = float(best_name.split("_")[1].removesuffix("pct")) / 100.0

    nonzero_names = [name for name in blend_names if name != "classical_00pct"]
    best_nonzero_name = min(nonzero_names, key=lambda name: summary[name]["rmse"])
    best_nonzero_alpha = float(best_nonzero_name.split("_")[1].removesuffix("pct")) / 100.0
    stages = {
        "deep": (0.0, False, False, False),
        "best_nonzero_fusion": (best_nonzero_alpha, False, False, False),
        "best_nonzero_refinement": (best_nonzero_alpha, True, False, False),
        "best_nonzero_selective_refinement": (best_nonzero_alpha, True, False, True),
        "best_nonzero_consistency": (best_nonzero_alpha, False, True, False),
        "best_nonzero_refinement_consistency": (best_nonzero_alpha, True, True, False),
    }
    stage_summary = {}
    for name, (alpha, refinement, consistency, selective_refinement) in stages.items():
        rows = []
        for index, sample in enumerate(samples):
            lr, rgb, valid, scale = (
                sample["lr_depth"],
                sample["hr_rgb"],
                sample["lr_valid_mask"],
                sample["scale"],
            )
            classical, deep, variants = predictions[index]
            if name == "deep":
                prediction = variants["deep"]
            else:
                prediction = _variants(
                    classical, deep, rgb, lr, valid, scale, alpha, refinement, consistency,
                    selective_refinement,
                )
            rows.append(
                compute_all_metrics(
                    prediction,
                    sample["depth_original"],
                    sample["data_range"],
                    sample["valid_mask_original"],
                )
            )
        stage_summary[name] = _mean_metrics(rows)

    result = {
        "meta": {"dataset": dataset.name, "split": args.split, "scale": args.scale, "n_samples": n},
        "fixed_blends_no_postprocessing": summary,
        "selected_blend": {"name": best_name, "classical_fraction": best_alpha},
        "selected_nonzero_blend": {
            "name": best_nonzero_name,
            "classical_fraction": best_nonzero_alpha,
        },
        "selected_blend_stages": stage_summary,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    with open(out / "fixed_blends_per_image.csv", "w", newline="") as handle:
        fields = ["sample", "method", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, sample in enumerate(samples):
            for name, rows in per_variant.items():
                writer.writerow({"sample": sample["name"], "method": name, **rows[index]})
    print(json.dumps(result, indent=2))
    print(f"saved: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
