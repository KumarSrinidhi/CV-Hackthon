# Team Owner: Rishita
# Workstream: Hybrid Fusion
# Module: hybrid/consistency.py
# Description: Measurement consistency enforcement: computing residual r_LR = D_LR - S(D_f) and applying consistency correction where forward degradation is known.

from typing import Optional
import cv2
import numpy as np


def enforce_measurement_consistency(
    fused_depth: np.ndarray,
    lr_depth: np.ndarray,
    scale: int = 4,
    valid_mask: Optional[np.ndarray] = None,
    gain: float = 0.5,
) -> np.ndarray:
    """
    Measurement consistency enforcement:
    Projects the high-resolution reconstructed depth back to the low-resolution sensor space
    using the forward sampling operator S, computes the measurement residual:
        r_LR = D_LR - S(D_f)
    and projects the residual back to correct D_f:
        D_cons = D_f + gain * S^dagger(r_LR)

    Args:
        fused_depth: Fused HR depth map of shape (H_hr, W_hr).
        lr_depth: Low-resolution sensor depth map of shape (H_lr, W_lr).
        scale: Downsampling / scale factor.
        valid_mask: Optional boolean validity mask on LR depth.
        gain: Consistency correction damping factor (in (0, 1]).

    Returns:
        Measurement-consistent HR depth map (float32).
    """
    fused_depth = np.asarray(fused_depth, dtype=np.float32)
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    h_hr, w_hr = fused_depth.shape[:2]
    h_lr, w_lr = lr_depth.shape[:2]

    # Forward degradation operator S: downsampling HR depth to LR grid
    s_depth = cv2.resize(fused_depth, (w_lr, h_lr), interpolation=cv2.INTER_AREA)

    # Compute sensor measurement residual
    r_lr = lr_depth - s_depth

    # Mask out invalid or missing measurements
    if valid_mask is not None:
        r_lr = np.where(valid_mask.astype(bool), r_lr, 0.0)
    else:
        invalid = ~np.isfinite(lr_depth) | (lr_depth <= 0)
        r_lr = np.where(~invalid, r_lr, 0.0)

    # Smooth / upsample residual back to high resolution
    r_hr = cv2.resize(r_lr, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    # Apply damped correction
    corrected = fused_depth + gain * r_hr

    # Guarantee numerical sanity
    invalid_out = ~np.isfinite(corrected)
    if np.any(invalid_out):
        corrected = np.where(invalid_out, fused_depth, corrected)

    return corrected.astype(np.float32)
