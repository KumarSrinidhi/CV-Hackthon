"""Reproduce the classical branch's synthetic validation and held-out evaluation."""

import argparse
import csv
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from ..classical_hr_depth import reconstruct
from ..interpolation import interpolate_depth
from .metrics import quality
from .synthetic_data import KINDS, SCALES, make_scene


BASELINES = ("nearest", "bilinear", "bicubic")
COMPARISONS = BASELINES + ("jbu", "guided", "depth_only", "no_refinement", "classical")
JUMPS = (0.05, 0.08, 0.12)


def predict(lr, rgb, valid, scale, method, jump):
    if method in BASELINES:
        return interpolate_depth(lr, valid, scale, method)
    return reconstruct(lr, rgb, valid, scale, method=method, relative_jump=jump)


def evaluate(split, methods, repeats, measure_memory):
    seeds = range(100, 103) if split == "validation" else range(200, 203)
    for kind in KINDS:
        for seed in seeds:
            for scale in SCALES:
                lr, rgb, valid, target = make_scene(kind, seed, scale)
                for method, jump in methods:
                    prediction = predict(lr, rgb, valid, scale, method, jump)
                    timings = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        predict(lr, rgb, valid, scale, method, jump)
                        timings.append((time.perf_counter() - start) * 1e3)
                    peak = None
                    if measure_memory:
                        tracemalloc.start()
                        try:
                            predict(lr, rgb, valid, scale, method, jump)
                            peak = tracemalloc.get_traced_memory()[1] / 1024
                        finally:
                            tracemalloc.stop()
                    yield dict(split=split, kind=kind, seed=seed, scale=scale,
                               method=method, jump=jump, **quality(prediction, target),
                               latency_ms=statistics.median(timings), peak_traced_kib=peak)


def mean(rows, key):
    return statistics.mean(row[key] for row in rows if row[key] is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "synthetic" / "benchmark.csv")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    validation = list(evaluate(
        "validation", [(method, 0.08) for method in COMPARISONS if method != "classical"] +
        [("classical", jump) for jump in JUMPS], repeats=1, measure_memory=False))
    candidates = [(method, 0.08) for method in ("jbu", "guided", "depth_only", "no_refinement")]
    candidates += [("classical", jump) for jump in JUMPS]
    scores = {}
    for method, jump in candidates:
        rows = [row for row in validation if (row["method"], row["jump"]) == (method, jump)]
        scores[method, jump] = mean(rows, "rmse")
        print(f"validation {method:14} jump={jump:.2f} RMSE={scores[method, jump]:.5f}")
    winner = min(scores, key=scores.get)
    print(f"Validation winner: {winner[0]}, relative_jump={winner[1]:.2f}")

    # Keep default selection separate from the held-out comparison. All classical
    # jump settings were scored above, and only the best of those enters the test.
    best_jump = min(JUMPS, key=lambda jump: scores["classical", jump])
    test = list(evaluate("test", [(method, best_jump if method == "classical" else 0.08)
                                  for method in COMPARISONS], repeats=5, measure_memory=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test[0]))
        writer.writeheader()
        writer.writerows(validation + test)

    print(f"Python {platform.python_version()} | NumPy {np.__version__} | OpenCV {cv2.__version__}")
    print("Method             RMSE     MAE    PSNR    SSIM   EdgeF1  BndRMSE   Cover    ms     KiB")
    for scale in (*SCALES, None):
        print(f"Scale {scale if scale is not None else 'all'}")
        for method in COMPARISONS:
            rows = [r for r in test if r["method"] == method and (scale is None or r["scale"] == scale)]
            metrics = ("rmse", "mae", "psnr", "ssim", "edge_f1", "boundary_rmse",
                       "coverage", "latency_ms", "peak_traced_kib")
            print(f"{method:14} " + " ".join(f"{mean(rows, key):7.3f}" for key in metrics))
    print("Validation winner by held-out scenario")
    for kind in KINDS:
        rows = [r for r in test if r["method"] == winner[0] and r["kind"] == kind]
        print(f"{kind:12} RMSE={mean(rows, 'rmse'):.3f} "
              f"boundary_RMSE={mean(rows, 'boundary_rmse'):.3f} coverage={mean(rows, 'coverage'):.3f}")


if __name__ == "__main__":
    main()

