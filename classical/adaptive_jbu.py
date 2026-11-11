"""Depth-vs-RGB edge agreement for adaptive joint bilateral upsampling.

RGB edges are only *supporting* evidence. A depth discontinuity is always the
primary signal; an RGB edge only "agrees" when it lies inside the band of an
actual depth discontinuity, so RGB-only texture edges never relocate depth.
"""

from __future__ import annotations

import cv2
import numpy as np

from .boundary_refinement import depth_boundaries


def edge_evidence(
    lr_depth,
    hr_rgb,
    *,
    rel_thresh: float = 0.08,
    rgb_thresh: float = 0.035,
):
    """Return ``(band, agreement, rgb_edges)`` at HR resolution.

    band       : (H, W) bool, a band around nearest-upsampled LR depth discontinuities.
    rgb_edges  : (H, W) bool, RGB Sobel edges.
    agreement  : (H, W) bool, RGB edges that lie inside the depth band (agreement
                 is always a subset of ``band``).
    """
    depth = np.asarray(lr_depth, dtype=np.float32)
    rgb = np.asarray(hr_rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    elif rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)

    H, W = rgb.shape[:2]
    h, w = depth.shape
    scale = H // h

    edges_lr = depth_boundaries(depth, relative_jump=rel_thresh)
    band = np.repeat(np.repeat(edges_lr, scale, axis=0), scale, axis=1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
    rgb_edges = cv2.magnitude(gx, gy) > rgb_thresh
    agreement = band & rgb_edges
    return band, agreement, rgb_edges
