"""Stress-test frozen JBU methods on reproducible randomized RGB-D inputs."""

import argparse
import platform
import statistics
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .. import classical_hr_depth
from ..classical_hr_depth import reconstruct
from ..interpolation import interpolate_depth
from .ablate_jbu import RESULTS, STAGES, summarize, write_csv
from .metrics import ablation_quality
from .synthetic_data import SCALES
from .visualization import write_comparison


SEEDS = range(1000, 1020)
KINDS = ("aligned", "unrelated_rgb", "sparse", "iid_depth")
BASELINES = ("nearest", "bilinear")
METHODS = BASELINES + STAGES


def random_scene(seed, kind, scale):
    rng = np.random.default_rng(seed)
    h, w = (int(size) * 8 for size in rng.integers(4, 9, size=2))
    y, x = np.indices((h, w), dtype=np.float32)
    y /= h
    x /= w
    depth = rng.uniform(2, 4) + rng.uniform(-0.7, 0.7) * x + rng.uniform(-0.7, 0.7) * y
    colors = np.broadcast_to(rng.uniform(0.12, 0.88, 3), (h, w, 3)).copy()
    for _ in range(rng.integers(3, 8)):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        rx, ry = rng.uniform(0.04, 0.37, 2)
        distance = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        region = distance < 1
        depth[region] = rng.uniform(0.8, 7) + rng.uniform(-0.4, 0.4) * x[region]
        colors[region] = rng.uniform(0.12, 0.88, 3)
    colors += rng.normal(0, 0.035, colors.shape)
    if kind in {"unrelated_rgb", "sparse", "iid_depth"}:
        colors = rng.random(colors.shape)
    if kind == "iid_depth":
        depth = rng.uniform(0.8, 7, (h, w))
    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    target = depth.astype(np.float32)

    offset = scale // 2
    lr = target[offset::scale, offset::scale].copy()
    degraded = np.random.default_rng([seed, scale])
    lr += degraded.normal(0, 0.20 if kind == "sparse" else 0.05, lr.shape).astype(np.float32)
    lr = np.maximum(lr, np.float32(0.01))
    valid = degraded.random(lr.shape) >= (0.85 if kind == "sparse" else 0.10)
    if kind == "sparse":
        rows, cols = lr.shape
        valid[rows // 4:3 * rows // 4, cols // 4:3 * cols // 4] = False
    # Keep empty-input behavior in the separate contract checks.
    valid.flat[np.argmax(lr)] = True
    lr[~valid] = 0
    return lr, rgb, valid, target


def predict(method, depth, rgb, valid, scale):
    if method in BASELINES:
        return interpolate_depth(depth, valid, scale, method)
    if method == "jbu":
        return classical_hr_depth(depth, rgb, valid, scale)
    return reconstruct(depth, rgb, valid, scale, method=method)


def check_output(output, shape, depth, valid):
    assert output.shape == shape and output.dtype == np.float32
    assert np.isfinite(output).all() and np.all(output >= 0)
    supported = output[output > 0]
    if supported.size:
        assert valid.any()
        np.testing.assert_array_less(depth[valid].min() - 1e-4, supported)
        np.testing.assert_array_less(supported, depth[valid].max() + 1e-4)
    if not valid.any():
        assert not supported.size


def evaluate(image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    for scale in SCALES:
        for kind in KINDS:
            for seed in SEEDS:
                lr, rgb, valid, target = random_scene(seed, kind, scale)
                snapshots = [array.copy() for array in (lr, rgb, valid)]
                predictions = {}
                for method in METHODS:
                    output = predict(method, lr, rgb, valid, scale)
                    check_output(output, target.shape, lr, valid)
                    timings = []
                    for _ in range(3):
                        start = perf_counter()
                        repeat = predict(method, lr, rgb, valid, scale)
                        timings.append((perf_counter() - start) * 1000)
                        np.testing.assert_array_equal(repeat, output)
                    for actual, original in zip((lr, rgb, valid), snapshots):
                        np.testing.assert_array_equal(actual, original)
                    predictions[method] = output
                    yield dict(split="random_stress", scale=scale, kind=kind, scene=seed,
                               height=target.shape[0], width=target.shape[1],
                               input_coverage=float(valid.mean()), method=method,
                               **ablation_quality(output, target, kind),
                               latency_ms=statistics.median(timings))
                if seed == SEEDS.start:
                    write_comparison(image_dir / f"{kind}_{seed}_{scale}x.png",
                                     rgb, target, predictions)
        print(f"Completed {scale}x quality and timing cases", flush=True)


def constant_result(method, scale, rgb, shape, case):
    depth = np.full(shape, 3.25, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    output = predict(method, depth, rgb, valid, scale)
    check_output(output, rgb.shape[:2], depth, valid)
    error = np.abs(output - depth[0, 0])
    return dict(case=case, method=method, scale=scale, lr_height=shape[0], lr_width=shape[1],
                constant_bad_pixels=int(np.count_nonzero(error > 1e-5)),
                constant_zero_pixels=int(np.count_nonzero(output == 0)),
                constant_max_error=float(error.max()), hr_pixels=output.size)


def contract_checks():
    rng = np.random.default_rng(91234)
    for scale in (1, 2, 3, 4, 8):
        for shape in ((1, 1), (1, 7), (8, 1), (17, 9), (129, 3)):
            size = tuple(n * scale for n in shape)
            rgb = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
            float_rgb = rgb.astype(np.float32) / 255
            depth = rng.uniform(1, 6, shape).astype(np.float32)
            valid = rng.random(shape) > 0.4
            valid.flat[-1] = True
            poisoned = depth.copy()
            poisoned[~valid] = np.resize([np.nan, np.inf, -1, 2000], (~valid).sum())
            for method in METHODS:
                output = predict(method, depth, rgb, valid, scale)
                check_output(output, size, depth, valid)
                np.testing.assert_array_equal(predict(method, poisoned, rgb, valid, scale), output)
                np.testing.assert_array_equal(predict(method, depth, float_rgb, valid, scale), output)
                empty = np.zeros(shape, dtype=bool)
                check_output(predict(method, depth, rgb, empty, scale), size, depth, empty)
                yield constant_result(method, scale, rgb, shape, "random_rgb")
    contrast = np.zeros((2, 2, 3), dtype=np.uint8)
    contrast[0, 0] = 255
    for method in METHODS:
        yield constant_result(method, 2, contrast, (1, 1), "single_white_pixel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "random")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | OpenCV {cv2.__version__}")
    checks = list(contract_checks())
    write_csv(args.out / "contracts.csv", checks)
    constant_failures = sum(row["constant_bad_pixels"] > 0 for row in checks)
    print(f"Checked {len(checks)} method/scale/shape combinations. "
          f"Constant-depth failures in {constant_failures}.", flush=True)
    rows = list(evaluate(args.out / "images"))
    write_csv(args.out / "scenes.csv", rows)
    write_csv(args.out / "summary.csv", list(summarize(rows)))
    print(f"Passed {len(rows)} method/scene cases. Results in {args.out}")


if __name__ == "__main__":
    main()
