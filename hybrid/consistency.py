"""Measurement-consistency enforcement.

When the forward degradation operator ``S`` is known (the bicubic downsampling used
to build the synthetic LR measurements), the residual ``r_LR = D_LR - S(D_f)`` at the
valid LR pixels is a direct consistency signal. It is upsampled and added back to the
fused HR depth, so the reconstruction is anchored to the measurements it was derived
from.

For an unknown real sensor formation process this exact residual is not valid; the
correction is therefore applied only when ``operator`` is supplied (the known
synthetic protocol) and only at the original valid measurement locations.
"""

from __future__ import annotations

import cv2
import numpy as np


def enforce_consistency(
    fused_hr,
    lr_depth,
    valid_mask=None,
    scale=4,
    operator=None,
    mode="bicubic",
    strength=1.0,
    iterations=1,
):
    """Project the fused HR depth onto the known measurements and return the corrected map.

    ``strength`` in [0, 1] damps the correction: 0 = no correction, 1 = full correction.
    ``iterations`` re-projects the corrected map (>= 0). The residual is upsampled as
    an implementation approximation, not treated as an exact projection. Invalid LR
    measurements are excluded on every iteration.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    Df = np.asarray(fused_hr, dtype=np.float32)
    lr = np.asarray(lr_depth, dtype=np.float32)
    H, W = Df.shape[:2]

    if operator is None:
        from data.degradation import forward_operator

        operator = forward_operator(scale, mode)

    valid = np.isfinite(lr) & (lr > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        valid &= mask

    interpolation = {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }.get(mode)
    if interpolation is None:
        raise ValueError(f"mode must be one of ('bicubic', 'bilinear', 'nearest'); got '{mode}'")

    corrected = Df.copy()
    for _ in range(iterations):
        projected = np.asarray(operator(corrected), dtype=np.float32)
        if projected.shape != lr.shape:
            raise ValueError(f"operator produced {projected.shape}, expected {lr.shape}")
        residual_lr = np.where(valid, lr - projected, 0.0).astype(np.float32)
        # Backproject with the same smoothness family as the known degradation. Divide
        # by the interpolated validity support so invalid LR holes do not become false
        # zero measurements or bleed into neighboring corrections.
        support_lr = valid.astype(np.float32)
        residual_hr = cv2.resize(residual_lr, (W, H), interpolation=interpolation)
        support_hr = cv2.resize(support_lr, (W, H), interpolation=interpolation)
        residual_hr = np.divide(
            residual_hr,
            np.maximum(support_hr, 1e-6),
            out=np.zeros_like(residual_hr),
            where=support_hr > 1e-6,
        )
        corrected = corrected + np.float32(strength) * residual_hr
    return np.maximum(corrected, 0.0).astype(np.float32)
