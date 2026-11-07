# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/degradation.py
# Description: Sensor simulation and downsampling degradation operators (scale factors 2x, 4x, 8x, noise, and holes).

from typing import Tuple
import cv2
import numpy as np


def create_lr_depth(
    hr_depth: np.ndarray,
    scale: int = 4,
    mode: str = "bicubic",
    noise_std: float = 0.0,
    hole_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate low-resolution sensor depth measurement from high-resolution ground truth.

    Args:
        hr_depth: High-resolution depth map (H_hr, W_hr).
        scale: Downsampling scale factor (e.g. 2, 4, 8).
        mode: Downsampling interpolation mode ('bicubic', 'nearest', 'area').
        noise_std: Standard deviation of Gaussian sensor noise to add (meters).
        hole_ratio: Fraction of missing / invalid depth pixels to simulate.

    Returns:
        (lr_depth, valid_mask) where:
            lr_depth: (H_hr // scale, W_hr // scale) float32 array.
            valid_mask: Boolean array of valid sensor measurements.
    """
    hr_depth = np.asarray(hr_depth, dtype=np.float32)
    h_hr, w_hr = hr_depth.shape[:2]
    h_lr, w_lr = max(1, h_hr // scale), max(1, w_hr // scale)

    # Downsampling
    if mode == "bicubic":
        interp = cv2.INTER_CUBIC
    elif mode == "nearest":
        interp = cv2.INTER_NEAREST
    elif mode == "area":
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    lr_depth = cv2.resize(hr_depth, (w_lr, h_lr), interpolation=interp).astype(np.float32)

    # Sensor noise
    if noise_std > 0.0:
        noise = np.random.normal(0.0, noise_std, lr_depth.shape).astype(np.float32)
        lr_depth = np.maximum(0.1, lr_depth + noise)

    # Sensor invalid measurement holes
    valid_mask = np.isfinite(lr_depth) & (lr_depth > 0)
    if hole_ratio > 0.0:
        drop_mask = np.random.rand(*lr_depth.shape) < hole_ratio
        lr_depth[drop_mask] = 0.0
        valid_mask[drop_mask] = False

    return lr_depth, valid_mask
