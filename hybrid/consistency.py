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


def enforce_consistency(fused_hr, lr_depth, valid_mask=None, scale=4, operator=None, mode="bicubic"):
    """Project the fused HR depth onto the known measurements and return the corrected map."""
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

    s_fused = operator(Df)
    if s_fused.shape != lr.shape:
        raise ValueError(f"operator produced {s_fused.shape}, expected {lr.shape}")

    residual_lr = np.where(valid, lr - s_fused, 0.0).astype(np.float32)
    # Nearest (block-constant) upsampling keeps the correction constant within each
    # LR block, so it anchors measurements without smearing intra-block depth edges.
    residual_hr = cv2.resize(residual_lr, (W, H), interpolation=cv2.INTER_NEAREST)
    corrected = Df + residual_hr
    return np.maximum(corrected, 0.0).astype(np.float32)
