"""Pixel-wise branch confidence estimation.

The returned weights are **normalized confidence scores** (``W_c + W_d == 1``), not
calibrated probabilities. They rank the two branches per pixel using three pieces of
evidence:

1. **Structure** -- normalized depth-gradient magnitude of the classical output. The
   classical branch (edge-aware JBU) is preferred where there is structure/edges.
2. **Measurement proximity** -- distance to the nearest original valid LR pixel. The
   classical branch is preferred near measurements (it preserves them); the deep
   branch is preferred far from measurements (it supplies a learned prior).
3. **RGB/depth edge agreement** -- where an RGB edge agrees with a depth
   discontinuity, the classical branch (which explicitly preserves boundaries) is
   preferred.

All evidence is in [0, 1] and combined with a fixed convex combination.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from classical.adaptive_jbu import edge_evidence


def _resolve_valid(depth, valid_mask):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        if mask.shape != depth.shape:
            raise ValueError(f"valid_mask shape {mask.shape} != depth shape {depth.shape}")
        valid &= mask
    return valid


def _normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mx = float(x.max())
    return x if mx <= 1e-12 else (x / mx).astype(np.float32)


def estimate_confidence(
    classical_depth,
    deep_depth,
    lr_depth,
    valid_mask=None,
    rgb=None,
    scale=4,
) -> tuple:
    """Return ``(W_c, W_d)`` float32 arrays in [0, 1] with ``W_c + W_d == 1``."""
    Dc = np.asarray(classical_depth, dtype=np.float32)
    Dd = np.asarray(deep_depth, dtype=np.float32)
    if Dc.shape != Dd.shape:
        raise ValueError(f"classical {Dc.shape} and deep {Dd.shape} shapes differ")
    H, W = Dc.shape[:2]

    # 1. structure: normalized Sobel magnitude of the classical reconstruction.
    gx = cv2.Sobel(Dc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Dc, cv2.CV_32F, 0, 1, ksize=3)
    edge_score = _normalize01(cv2.magnitude(gx, gy))

    # 2. measurement proximity: 0 at a valid LR pixel, 1 far away.
    valid = _resolve_valid(lr_depth, valid_mask)
    if valid.any():
        dist = ndi.distance_transform_edt(~valid)
        dist_hr = cv2.resize(dist.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        dist_score = _normalize01(dist_hr)
    else:
        dist_score = np.ones((H, W), np.float32)

    # 3. RGB/depth edge agreement (0 when rgb is unavailable).
    agreement = np.zeros((H, W), np.float32)
    if rgb is not None:
        try:
            _, agree, _ = edge_evidence(lr_depth, rgb)
            agreement = agree.astype(np.float32)
        except Exception:
            agreement = np.zeros((H, W), np.float32)

    # Convex combination; classical is trusted on structure, near measurements, and
    # on agreed edges. Deep is trusted on flat, measurement-poor regions.
    wc = 0.4 * edge_score + 0.3 * (1.0 - dist_score) + 0.3 * agreement
    wc = np.clip(wc, 0.0, 1.0).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)

    # A branch with invalid/absent output must never be trusted there.
    invalid_deep = ~np.isfinite(Dd) | (Dd <= 0)
    wc[invalid_deep] = 1.0
    wd[invalid_deep] = 0.0

    # Enforce the normalization invariant exactly.
    total = wc + wd
    safe = total > 1e-8
    wc = np.where(safe, wc / np.maximum(total, 1e-8), 0.5).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)
    return wc, wd
