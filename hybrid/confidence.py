"""Conservative, measurement-aware branch confidence estimation.

The deep branch is the primary estimator because validation shows that the
classical branch is substantially less accurate on flat regions. The classical
branch receives a small default contribution and is allowed to contribute more
only where it has boundary support and explains the measured LR depth better.
"""

from __future__ import annotations

import cv2
import numpy as np

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


def _measurement_advantage(classical, deep, lr_depth, valid_mask, operator=None):
    """Return where classical better matches the available LR measurements."""
    h, w = lr_depth.shape
    classical_safe = np.where(np.isfinite(classical), classical, 0.0)
    deep_safe = np.where(np.isfinite(deep), deep, 0.0)
    if operator is not None:
        classical_lr = np.asarray(operator(classical_safe), dtype=np.float32)
        deep_lr = np.asarray(operator(deep_safe), dtype=np.float32)
        if classical_lr.shape != (h, w) or deep_lr.shape != (h, w):
            raise ValueError(f"operator produced {classical_lr.shape}, expected {(h, w)}")
    else:
        classical_lr = cv2.resize(classical_safe, (w, h), interpolation=cv2.INTER_AREA)
        deep_lr = cv2.resize(deep_safe, (w, h), interpolation=cv2.INTER_AREA)
    valid = _resolve_valid(lr_depth, valid_mask)
    err_c = np.abs(classical_lr - lr_depth)
    err_d = np.abs(deep_lr - lr_depth)
    advantage = (err_d - err_c) / (err_d + err_c + 1e-6)
    advantage = np.where(np.isfinite(advantage), advantage, 0.0)
    return np.where(valid, np.clip(advantage, 0.0, 1.0), 0.0).astype(np.float32)


def estimate_confidence(
    classical_depth,
    deep_depth,
    lr_depth,
    valid_mask=None,
    rgb=None,
    scale=4,
    *,
    adaptive=True,
    max_classical_weight=0.35,
    operator=None,
    hole_classical_weight=0.5,
) -> tuple:
    """Return normalized branch weights.

    Adaptive weighting is deliberately conservative: classical support is gated by
    depth/RGB edge agreement and a measurement residual, rather than treating an RGB
    edge or a low residual as proof of higher HR accuracy.  ``adaptive=False`` is the
    zero-classical ablation (apart from invalid deep predictions).
    ``max_classical_weight`` caps the classical contribution in [0, 1] outside the
    hole band; ``hole_classical_weight`` caps it inside the band around missing LR
    measurements, where the deep input is nearest-filled and therefore unreliable
    (and only where the classical output is itself supported).
    """
    if not 0.0 <= max_classical_weight <= 1.0:
        raise ValueError("max_classical_weight must be in [0, 1]")
    if not 0.0 <= hole_classical_weight <= 1.0:
        raise ValueError("hole_classical_weight must be in [0, 1]")
    Dc = np.asarray(classical_depth, dtype=np.float32)
    Dd = np.asarray(deep_depth, dtype=np.float32)
    if Dc.shape != Dd.shape:
        raise ValueError(f"classical {Dc.shape} and deep {Dd.shape} shapes differ")
    H, W = Dc.shape[:2]

    # 1. structure: normalized Sobel magnitude of the classical reconstruction.
    gx = cv2.Sobel(Dc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Dc, cv2.CV_32F, 0, 1, ksize=3)
    edge_score = _normalize01(cv2.magnitude(gx, gy))

    # 2. Compare branch residuals at the available LR measurements. This is
    # safer than assigning a large weight merely because a pixel is nearby.
    valid = _resolve_valid(lr_depth, valid_mask)
    if valid.any():
        advantage_lr = _measurement_advantage(Dc, Dd, lr_depth, valid_mask, operator)
        advantage = cv2.resize(advantage_lr, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        advantage = np.zeros((H, W), np.float32)

    # 3. RGB/depth edge agreement (0 when rgb is unavailable).
    agreement = np.zeros((H, W), np.float32)
    if rgb is not None:
        try:
            _, agree, _ = edge_evidence(lr_depth, rgb)
            agreement = agree.astype(np.float32)
        except Exception:
            agreement = np.zeros((H, W), np.float32)

    if adaptive:
        # Keep deep dominant by default. Extra classical weight requires both boundary
        # support and a meaningful measurement advantage. Strongly supported boundaries
        # may use a larger contribution so a blurred deep edge can actually be repaired.
        cap = float(max_classical_weight)
        boundary_support = 0.35 * edge_score + 0.65 * agreement
        advantage_gate = np.clip((advantage - 0.15) / 0.85, 0.0, 1.0)
        wc = 0.01 + (cap - 0.01) * boundary_support * advantage_gate
        wc = np.clip(wc, 0.0, cap).astype(np.float32)

        # Hole band: the deep input is nearest-filled across missing LR
        # measurements, so its prediction is unreliable exactly there, while the
        # classical masked reconstruction stays anchored to real measurements.
        if valid.any() and not valid.all():
            hole_lr = cv2.dilate(
                (~valid).astype(np.uint8), np.ones((3, 3), np.uint8)
            ).astype(bool)
        else:
            hole_lr = np.zeros_like(valid)
        hole_zone = cv2.resize(
            hole_lr.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
        )
        classical_supported = np.isfinite(Dc) & (Dc > 0)
        wc = np.maximum(
            wc,
            (float(hole_classical_weight) * hole_zone * classical_supported).astype(
                np.float32
            ),
        )
    else:
        wc = np.zeros((H, W), np.float32)

    # A branch with invalid/absent output must never be trusted there.
    invalid_classical = ~np.isfinite(Dc) | (Dc <= 0)
    invalid_deep = ~np.isfinite(Dd) | (Dd <= 0)
    wc[invalid_classical] = 0.0
    wc[invalid_deep] = 1.0
    wd = (1.0 - wc).astype(np.float32)

    # Enforce the normalization invariant exactly.
    total = wc + wd
    safe = total > 1e-8
    wc = np.where(safe, wc / np.maximum(total, 1e-8), 0.5).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)
    return wc, wd
