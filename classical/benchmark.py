"""Self-contained classical benchmark on synthetic depth scenes.

Runs the classical reconstruction methods against ground truth and reports the
shared evaluation metrics (eval.metrics.compute_all_metrics). This replaces the
earlier compatibility import that referenced a lost ``classical.evaluation``
subpackage.

    python -m classical.benchmark --scale 4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from classical import classical_hr_depth
from classical.pipeline import reconstruct
from eval.metrics import compute_all_metrics

_METHODS = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical",
            "adaptive_jbu", "refined_jbu"]


def make_scene(height=120, width=160):
    """Piecewise-depth scene with a foreground block and a smooth background."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = 2.0 + 2.0 * (xx / width) + 0.2 * np.sin(yy / 13.0)
    depth[height // 4 : height // 2, width // 4 : width // 2] = 5.0
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[:] = (60, 60, 120)
    rgb[height // 4 : height // 2, width // 4 : width // 2] = (200, 160, 90)
    return depth.astype(np.float32), rgb


def benchmark(scale=4, height=120, width=160):
    gt, rgb = make_scene(height, width)
    h, w = height // scale, width // scale
    lr = cv2.resize(gt, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    valid = np.isfinite(lr) & (lr > 0)

    rows = []
    for method in _METHODS:
        pred = reconstruct(lr, rgb, valid, scale, method)
        m = compute_all_metrics(pred, gt, data_range=10.0, valid=np.isfinite(gt) & (gt > 0))
        rows.append((method, m))
    return {"lr_depth": lr, "gt": gt, "rgb": rgb, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--width", type=int, default=160)
    a = ap.parse_args(argv)

    res = benchmark(a.scale, a.height, a.width)
    print(f"classical synthetic benchmark x{a.scale} ({a.height}x{a.width})")
    print(f"{'method':<12} {'rmse':>8} {'mae':>8} {'psnr':>8} {'ssim':>8} {'bnd_rmse':>9} {'edge_f1':>8}")
    for method, m in res["rows"]:
        print(f"{method:<12} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['psnr']:>8.2f} "
              f"{m['ssim']:>8.4f} {m['boundary_rmse']:>9.4f} {m['edge_f1']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
