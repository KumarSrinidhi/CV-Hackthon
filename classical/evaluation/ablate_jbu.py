"""Tune JBU on synthetic scenes and evaluate frozen settings on held-out inputs."""

import argparse
import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ..classical_hr_depth import reconstruct
from ..methods.tuned_jbu import DEFAULT_SETTINGS
from .metrics import ablation_quality
from .synthetic_data import KINDS, SCALES, SPLIT_SEEDS, make_scene
from .visualization import write_comparison


STAGES = ("jbu", "masked_jbu", "tuned_jbu", "adaptive_jbu", "refined_jbu")
GRID = tuple(itertools.product((0.55, 0.85, 1.1), (0.08, 0.12, 0.22), (1, 2)))
METRICS = ("rmse", "mae", "boundary_rmse", "edge_f1", "thin_rmse", "coverage", "latency_ms")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def synthetic_samples(split, scale):
    for kind in KINDS:
        for seed in SPLIT_SEEDS[split]:
            yield kind, seed, *make_scene(kind, seed, scale)


def redwood_pairs(root):
    pairs = [(depth, root / "color" / (depth.stem + ".jpg"))
             for depth in sorted((root / "depth").glob("*.png"))]
    if not pairs:
        raise ValueError(f"No depth PNGs found in {root / 'depth'}")
    for _, color in pairs:
        if not color.is_file():
            raise ValueError(f"Missing aligned RGB image: {color}")
    return pairs


def redwood_samples(pairs, scale):
    for depth_path, rgb_path in pairs:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or depth.ndim != 2 or rgb is None or rgb.shape[:2] != depth.shape:
            raise ValueError(f"Expected aligned RGB and single-channel depth for {depth_path.stem}")
        if any(size % scale for size in depth.shape) or not np.any(depth > 0):
            raise ValueError(f"Invalid depth dimensions or empty reference for {depth_path.stem} at {scale}x")
        depth = depth.astype(np.float32)
        offset = scale // 2
        lr = depth[offset::scale, offset::scale]
        yield "redwood", depth_path.stem, lr, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), lr > 0, depth


def evaluate(split, scale, stages, samples, *, repeats=3, image_dir=None):
    for kind, scene, lr, rgb, valid, target in samples:
        predictions = {}
        for method, settings in stages:
            prediction = reconstruct(lr, rgb, valid, scale, method=method, jbu_settings=settings)
            timings = []
            for _ in range(repeats):
                start = perf_counter()
                reconstruct(lr, rgb, valid, scale, method=method, jbu_settings=settings)
                timings.append((perf_counter() - start) * 1000)
            predictions[method] = prediction
            yield dict(split=split, scale=scale, kind=kind, scene=scene, method=method,
                       sigma_space=settings[0], sigma_color=settings[1], radius=settings[2],
                       **ablation_quality(prediction, target, kind),
                       latency_ms=statistics.median(timings) if timings else None)
        if image_dir is not None:
            write_comparison(image_dir / f"{scene}_{scale}x.png", rgb, target, predictions)


def summarize(rows):
    groups = defaultdict(list)
    by_seed = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        split, scale, kind, scene, method = (row[key] for key in ("split", "scale", "kind", "scene", "method"))
        for metric in METRICS:
            value = row[metric]
            if value is not None:
                groups[split, scale, kind, method, metric].append(value)
                if split != "redwood":
                    by_seed[split, scale, scene, method, metric].append(value)
        if method in ("jbu", "tuned_jbu"):
            pairs[split, scale, kind, scene][method] = row["rmse"]

    for (split, scale, _, method, metric), values in by_seed.items():
        groups[split, scale, "all", method, metric].append(statistics.mean(values))

    seed_deltas = defaultdict(list)
    for (split, scale, kind, scene), pair in pairs.items():
        delta = pair["tuned_jbu"] - pair["jbu"]
        groups[split, scale, kind, "tuned_jbu_minus_jbu", "rmse"].append(delta)
        if split != "redwood":
            seed_deltas[split, scale, scene].append(delta)
    for (split, scale, _), deltas in seed_deltas.items():
        groups[split, scale, "all", "tuned_jbu_minus_jbu", "rmse"].append(statistics.mean(deltas))

    rng = np.random.default_rng(2026)
    for (split, scale, kind, method, metric), values in sorted(groups.items()):
        n = len(values)
        low = high = None
        if split != "redwood" and n >= 10:
            indices = rng.integers(n, size=(10000, n))
            means = np.asarray(values)[indices].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
        yield dict(split=split, scale=scale, kind=kind, method=method, metric=metric, n=n,
                   mean=statistics.mean(values), sd=statistics.stdev(values) if n > 1 else None,
                   ci95_low=low, ci95_high=high)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "synthetic" / "ablation.csv")
    parser.add_argument("--redwood", type=Path, help="Folder containing aligned depth/*.png and color/*.jpg.")
    parser.add_argument("--redwood-out", type=Path, default=RESULTS / "redwood")
    args = parser.parse_args()
    try:
        pairs = redwood_pairs(args.redwood) if args.redwood else []
    except ValueError as error:
        parser.error(str(error))

    cv2.setNumThreads(1)
    tuning, held_out = [], []
    frozen = {}
    for scale in SCALES:
        scores = []
        for settings in GRID:
            cases = list(evaluate("tuning", scale, [("tuned_jbu", settings)],
                                  synthetic_samples("tuning", scale), repeats=0))
            scores.append((statistics.mean(row["rmse"] for row in cases), settings))
            tuning.extend(cases)
        _, settings = min(scores)
        frozen[scale] = [(method, DEFAULT_SETTINGS if method in STAGES[:2] else settings)
                         for method in STAGES]
        print(f"{scale}x frozen settings {settings}", flush=True)

    for split in ("validation", "test", "test_extension"):
        for scale, stages in frozen.items():
            held_out.extend(evaluate(split, scale, stages, synthetic_samples(split, scale)))
    write_csv(args.out, tuning + held_out)
    write_csv(args.out.with_name(f"{args.out.stem}_summary.csv"), list(summarize(held_out)))

    if pairs:
        image_dir = args.redwood_out / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for scale, stages in frozen.items():
            rows.extend(evaluate("redwood", scale, stages, redwood_samples(pairs, scale),
                                 image_dir=image_dir))
        write_csv(args.redwood_out / "ablation.csv", rows)
        write_csv(args.redwood_out / "ablation_summary.csv", list(summarize(rows)))


if __name__ == "__main__":
    main()
