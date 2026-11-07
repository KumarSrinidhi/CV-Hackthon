"""Command-line entry point for depth reconstruction.

    python main.py --rgb RGB.png --depth depth.npy --scale 4 --output results/out

The pipeline always runs the full hybrid workflow:
  Classical branch (A1 hole-fill → A2 bicubic → A3 JBU → A4 boundary refinement)
  + Deep branch (SGNet guided SR) → Hybrid fusion (alpha=0.02) → Measurement consistency.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from app.config import hybrid_kwargs, load_config
from app.validation import validate_inputs
from app.visualization import (
    depth_colormap,
    error_map,
    export_npy,
    export_png,
    shared_depth_colormap,
    side_by_side,
    to_rgb_uint8,
)
from classical.pipeline import reconstruct as classical_reconstruct


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path, depth_scale=1.0):
    """Load a depth map from .npy or an image file (e.g. 16-bit PNG in mm).

    ``depth_scale`` multiplies the stored values, e.g. ``0.001`` converts a
    millimetre PNG to metres.
    """
    if not os.path.isfile(str(path)):
        raise ValueError(f"File not found: {path}")
    if str(path).lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read depth: {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
    arr = arr.astype(np.float32)
    return arr * np.float32(depth_scale) if depth_scale != 1.0 else arr


def _fallback(lr, rgb, mask, scale, reason):
    print(f"WARNING: {reason}; fell back to classical.", file=sys.stderr)
    hr = classical_reconstruct(lr, rgb, mask, scale, method="classical")
    return {
        "depth": hr,
        "method": "classical",
        "fallback": True,
        "notes": [f"{reason}. Showing the classical result instead."],
        "intermediates": {},
    }


def reconstruct(rgb, lr_depth, valid_mask=None, scale=4, config=None):
    """Validate inputs and run the hybrid pipeline. Returns a result dict.

    Keys: ``depth`` (H, W) float32, ``method``, ``fallback`` (bool),
    ``notes`` (list of str), ``intermediates`` (branch outputs + confidence maps).
    """
    cfg = config or load_config()
    rgb, lr, mask, scale = validate_inputs(rgb, lr_depth, valid_mask, scale,
                                           deep_scales=tuple(cfg["app"]["deep_scales"]))
    out = {"method": "hybrid", "fallback": False, "notes": [], "intermediates": {}}

    if scale not in cfg["app"]["deep_scales"]:
        return _fallback(lr, rgb, mask, scale,
                         f"hybrid needs the deep branch, which does not support x{scale}")
    from hybrid import hybrid_hr_depth

    hr, inter = hybrid_hr_depth(lr, rgb, mask, scale, return_intermediates=True,
                                **hybrid_kwargs(cfg))
    out["depth"] = hr
    out["intermediates"] = inter
    if not inter.get("deep_available", True):
        out["notes"].append("Deep branch unavailable (no checkpoint) — running classical pipeline.")
    return out


def comparison_image(rgb, result, gt=None, lr_depth=None):
    """Side-by-side panel: RGB, LR depth, classical branch, deep branch, hybrid output, optional GT."""
    inter = result.get("intermediates", {})
    panels = []
    H, W = result["depth"].shape[:2]

    if lr_depth is not None:
        panels.append(("LR depth", cv2.resize(np.asarray(lr_depth, np.float32), (W, H),
                                               interpolation=cv2.INTER_NEAREST)))
    if inter.get("classical") is not None:
        panels.append(("classical", inter["classical"]))
    if inter.get("deep") is not None:  # None when deep branch was unavailable
        panels.append(("deep", inter["deep"]))
    panels.append(("hybrid", result["depth"]))
    if gt is not None:
        panels.append(("ground truth", gt))

    depths = [d for _, d in panels]
    labels = [l for l, _ in panels]
    colored = shared_depth_colormap(depths)
    return side_by_side([("RGB", to_rgb_uint8(rgb))] + list(zip(labels, colored)))


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="P4 edge-preserving depth reconstruction")
    ap.add_argument("--rgb", required=True, help="HR RGB guidance image")
    ap.add_argument("--depth", required=True, help="LR depth map (.npy or image)")
    ap.add_argument("--mask", default=None, help="optional LR validity mask (non-zero = valid)")
    ap.add_argument("--scale", type=int, default=cfg["app"]["default_scale"])
    ap.add_argument("--output", default=cfg["app"]["output_dir"])
    ap.add_argument("--gt", default=None, help="optional HR ground-truth depth for metrics + error map")
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="multiply loaded depth values (e.g. 0.001 for millimetre PNGs)")
    ap.add_argument("--compare", action="store_true",
                    help="write a side-by-side panel: RGB, LR depth, classical, deep, hybrid, GT")
    args = ap.parse_args(argv)

    try:
        rgb = load_image(args.rgb)
        lr = load_depth(args.depth, args.depth_scale)
        mask = load_depth(args.mask) if args.mask else None
        gt = load_depth(args.gt, args.depth_scale) if args.gt else None
        result = reconstruct(rgb, lr, mask, args.scale, cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    depth = result["depth"]
    os.makedirs(args.output, exist_ok=True)
    base = os.path.join(args.output, f"{result['method']}_x{args.scale}")
    export_npy(depth, base + ".npy")
    export_png(depth, base + ".png")
    cv2.imwrite(base + "_colormap.png", depth_colormap(depth))

    print(f"method={result['method']} scale=x{args.scale}")
    for note in result["notes"]:
        print(f"note: {note}")
    print(f"depth: {depth.shape} dtype={depth.dtype} range=[{depth.min():.3f}, {depth.max():.3f}]")
    print(f"saved: {base}.npy, {base}.png, {base}_colormap.png")

    if args.compare:
        comp = comparison_image(rgb, result, gt, lr)
        path = os.path.join(args.output, f"comparison_x{args.scale}.png")
        cv2.imwrite(path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"saved: {path}")

    if gt is not None:
        from eval.metrics import compute_all_metrics

        pred = depth
        if pred.shape != gt.shape:
            print(f"WARNING: prediction {pred.shape} and GT {gt.shape} differ; "
                  "resizing prediction for metrics.", file=sys.stderr)
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        metrics = compute_all_metrics(pred, gt, data_range=cfg["app"]["data_range"])
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")
        err_path = base + "_error.png"
        cv2.imwrite(err_path, error_map(pred, gt))
        print(f"saved: {err_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
