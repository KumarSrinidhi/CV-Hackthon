"""Command-line entry point for depth reconstruction.

    python main.py --rgb RGB.png --depth depth.npy --scale 4 --method hybrid \
                   --output results/out

Supported methods: nearest, bilinear, bicubic, jbu, guided, classical,
adaptive_jbu, refined_jbu, deep, hybrid.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from app.visualization import depth_colormap, export_npy, export_png
from classical import classical_hr_depth
from classical.interpolation import interpolate_depth
from classical.pipeline import reconstruct as classical_reconstruct

_INTERP = {"nearest", "bilinear", "bicubic"}
_CLASSICAL = {"jbu", "guided", "classical", "depth_only", "no_refinement",
              "masked_jbu", "tuned_jbu", "adaptive_jbu", "refined_jbu"}


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    if path.lower().endswith(".npy"):
        return np.load(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read depth: {path}")
    return img.astype(np.float32)


def reconstruct(rgb, lr_depth, valid_mask=None, scale=4, method="hybrid"):
    """Dispatch to the requested reconstruction branch. Returns a dict."""
    rgb = np.asarray(rgb)
    lr = np.asarray(lr_depth, dtype=np.float32)
    if lr.ndim == 3 and lr.shape[2] == 1:
        lr = lr[..., 0]

    if method in _INTERP:
        hr = interpolate_depth(lr, valid_mask, scale, method)
        return {"depth": hr, "method": method, "intermediates": {}}

    if method in _CLASSICAL:
        hr = classical_reconstruct(lr, rgb, valid_mask, scale, method=method)
        return {"depth": hr, "method": method, "intermediates": {}}

    if method == "deep":
        from deep.inference import deep_hr_depth

        hr = deep_hr_depth(lr, rgb, valid_mask, scale)
        return {"depth": hr, "method": method, "intermediates": {}}

    if method == "hybrid":
        from hybrid import hybrid_hr_depth

        try:
            hr, intermediates = hybrid_hr_depth(
                lr, rgb, valid_mask, scale, return_intermediates=True
            )
        except NotImplementedError:
            # Deep branch unavailable: degrade to classical and say so explicitly.
            print("WARNING: deep branch unavailable; hybrid fell back to classical.", file=sys.stderr)
            hr = classical_reconstruct(lr, rgb, valid_mask, scale, method="classical")
            return {"depth": hr, "method": "classical", "intermediates": {}}
        return {"depth": hr, "method": method, "intermediates": intermediates}

    raise ValueError(f"Unknown method: {method!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P4 edge-preserving depth reconstruction")
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--output", default="tmp/app_output")
    ap.add_argument("--gt", default=None)
    args = ap.parse_args(argv)

    try:
        rgb = load_image(args.rgb)
        lr = load_depth(args.depth)
        mask = load_depth(args.mask) if args.mask else None
        result = reconstruct(rgb, lr, mask, args.scale, args.method)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    base = os.path.join(args.output, f"{args.method}_x{args.scale}")
    export_npy(result["depth"], base + ".npy")
    export_png(result["depth"], base + ".png")
    cv2.imwrite(base + "_colormap.png", depth_colormap(result["depth"]))

    print(f"method={result['method']} scale=x{args.scale}")
    print(f"depth: {result['depth'].shape} dtype={result['depth'].dtype} "
          f"range=[{result['depth'].min():.3f}, {result['depth'].max():.3f}]")
    print(f"saved: {base}.npy, {base}.png, {base}_colormap.png")

    if args.gt:
        gt = load_depth(args.gt)
        from eval.metrics import compute_all_metrics

        pred = result["depth"]
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        metrics = compute_all_metrics(pred, gt, data_range=10.0)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
