# Team Owner: Kumar
# Workstream: Deep Learning / Statistics
# Module: scripts/statistical_comparison.py
# Description: Paired per-image statistical comparison of two evaluation runs (plan.txt §9).
"""Paired per-image comparison with bootstrap confidence intervals; the scene is the
experimental unit (pixels are never treated as independent).

    python scripts/statistical_comparison.py \
        --baseline results/baseline_deep_repro --candidate results/e4b_hf_clean100 \
        --metrics rmse boundary_rmse edge_f1

Interpretation fields: mean/median delta (candidate - baseline), bootstrap 95% CI of the
mean delta, n improved / worsened, worst-case regression, and a practical-significance note.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_METRICS = ["rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_f1"]


def load_per_image(result_dir: str) -> dict:
    path = next(Path(result_dir).glob("*_per_image.csv"))
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["sample"]] = {k: float(v) for k, v in r.items()
                                 if k not in ("sample", "method") and v not in ("", "nan")}
    return rows


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args(argv)

    base = load_per_image(a.baseline)
    cand = load_per_image(a.candidate)
    common = sorted(set(base) & set(cand))
    print(f"paired samples: {len(common)} ({a.baseline} vs {a.candidate})")

    report = {"n_paired": len(common), "baseline": a.baseline, "candidate": a.candidate, "metrics": {}}
    for metric in a.metrics:
        pairs = [(base[s][metric], cand[s][metric]) for s in common
                 if metric in base[s] and metric in cand[s]
                 and np.isfinite(base[s][metric]) and np.isfinite(cand[s][metric])]
        b = np.array([p[0] for p in pairs])
        c = np.array([p[1] for p in pairs])
        n = len(pairs)
        if n < 10:
            print(f"  {metric}: skipped (only {n} finite paired values)")
            continue
        diffs = c - b
        lower, upper = paired_bootstrap(diffs)
        # for rmse/mae/boundary_rmse lower is better; for psnr/ssim/edge_f1 higher is better
        lower_better = metric in ("rmse", "mae", "boundary_rmse")
        improved = int((diffs < 0).sum()) if lower_better else int((diffs > 0).sum())
        entry = {
            "n_finite_pairs": n,
            "mean_delta": float(diffs.mean()),
            "median_delta": float(np.median(diffs)),
            "std_delta": float(diffs.std()),
            "ci95_of_mean_delta": [lower, upper],
            "n_improved": improved,
            "n_worsened": len(diffs) - improved,
            "worst_case_regression": float(diffs.max()) if lower_better else float(diffs.min()),
            "baseline_mean": float(b.mean()),
            "candidate_mean": float(c.mean()),
        }
        sig = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
        entry["statistically_signed"] = bool(sig)
        entry["note"] = ("delta is candidate - baseline; negative = candidate better for "
                         "error metrics, positive = candidate better for PSNR/SSIM/F1. "
                         "'statistically_signed' = 95% CI excludes zero.")
        report["metrics"][metric] = entry
        print(f"  {metric} (n={n}): mean_delta={entry['mean_delta']:+.5f} "
              f"CI95=[{lower:+.5f}, {upper:+.5f}] improved/worsened="
              f"{improved}/{n - improved} signed={sig}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
