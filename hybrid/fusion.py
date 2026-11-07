# Team Owner: Rishita
# Workstream: Hybrid Fusion
# Module: hybrid/fusion.py
# Description: Branch fusion combining deep HR depth (D_d) and classical HR depth (D_c) using confidence weights and fixed-weight ablation baselines.

from typing import Union
import numpy as np


def fuse_depth_maps(
    d_c: np.ndarray,
    d_d: np.ndarray,
    w_c: Union[np.ndarray, float],
    w_d: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Fuse classical HR depth map and deep HR depth map using spatial confidence weights:
        D_f = W_c * D_c + W_d * D_d

    Args:
        d_c: High-resolution depth map from classical branch.
        d_d: High-resolution depth map from deep branch.
        w_c: Normalized weight map or scalar for classical branch.
        w_d: Normalized weight map or scalar for deep branch.

    Returns:
        Fused high-resolution depth map (float32).
    """
    d_c = np.asarray(d_c, dtype=np.float32)
    d_d = np.asarray(d_d, dtype=np.float32)
    w_c = np.asarray(w_c, dtype=np.float32)
    w_d = np.asarray(w_d, dtype=np.float32)

    # Branch fusion blending
    fused = w_c * d_c + w_d * d_d

    # Numerical safety check
    invalid = ~np.isfinite(fused)
    if np.any(invalid):
        fused = np.where(invalid, d_d, fused)

    return fused.astype(np.float32)


def fixed_weight_fusion(
    d_c: np.ndarray,
    d_d: np.ndarray,
    alpha: float = 0.02,
) -> np.ndarray:
    """
    Fixed-weight linear fusion baseline:
        D_f = alpha * D_c + (1 - alpha) * D_d

    Args:
        d_c: High-resolution depth map from classical branch.
        d_d: High-resolution depth map from deep branch.
        alpha: Weight assigned to classical branch (calibrated optimal: 0.02).

    Returns:
        Fused high-resolution depth map (float32).
    """
    return fuse_depth_maps(d_c, d_d, w_c=alpha, w_d=1.0 - alpha)
