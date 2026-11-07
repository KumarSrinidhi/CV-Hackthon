# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/jbu.py
# Description: Joint Bilateral Upsampling (JBU) for edge-preserving depth super-resolution using HR RGB guidance.

from typing import Optional, Tuple, Union
import cv2
import numpy as np
from classical.interpolation import bicubic_upsample


def joint_bilateral_upsample(
    lr_depth: np.ndarray,
    hr_rgb: np.ndarray,
    scale: int = 4,
    target_size: Optional[Tuple[int, int]] = None,
    radius: int = 3,
    sigma_spatial: float = 2.5,
    sigma_range: float = 0.1,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Joint Bilateral Upsampling (JBU) following Kopf et al. (2007).
    Upsamples low-resolution depth while preserving high-resolution edges
    using the RGB guidance image.

    Args:
        lr_depth: Low-resolution depth map of shape (H_lr, W_lr).
        hr_rgb: High-resolution RGB guidance image of shape (H_hr, W_hr, 3).
        scale: Upsampling factor (e.g. 2, 4, 8).
        target_size: Optional (H_hr, W_hr) tuple. If None, derived from hr_rgb or scale.
        radius: Half-window size for bilateral filtering.
        sigma_spatial: Gaussian spatial parameter.
        sigma_range: Gaussian color range parameter.
        valid_mask: Optional boolean mask of valid LR depth measurements.

    Returns:
        High-resolution upsampled depth map of shape (H_hr, W_hr).
    """
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    hr_rgb = np.asarray(hr_rgb, dtype=np.float32)

    if hr_rgb.ndim == 2:
        hr_rgb = np.stack([hr_rgb] * 3, axis=-1)

    h_hr, w_hr = hr_rgb.shape[:2]
    if target_size is not None:
        h_hr, w_hr = target_size

    # Initial upsampling of low-resolution depth to high-resolution grid
    init_hr = bicubic_upsample(lr_depth, target_size=(h_hr, w_hr), scale=scale)

    # Normalize RGB guidance to [0, 1]
    if hr_rgb.max() > 1.0:
        guide = hr_rgb / 255.0
    else:
        guide = hr_rgb.copy()

    # Vectorized multi-shift Joint Bilateral Filtering on the HR grid
    weights_sum = np.zeros((h_hr, w_hr), dtype=np.float32)
    result = np.zeros((h_hr, w_hr), dtype=np.float32)

    two_sigma_s_sq = 2.0 * sigma_spatial * sigma_spatial
    two_sigma_r_sq = 2.0 * sigma_range * sigma_range

    pad = radius
    padded_guide = np.pad(guide, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    padded_src = np.pad(init_hr, ((pad, pad), (pad, pad)), mode="reflect")

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dx * dx + dy * dy) / two_sigma_s_sq)
            
            y_start = pad + dy
            x_start = pad + dx
            shifted_src = padded_src[y_start : y_start + h_hr, x_start : x_start + w_hr]
            shifted_guide = padded_guide[y_start : y_start + h_hr, x_start : x_start + w_hr]

            color_diff_sq = np.sum((guide - shifted_guide) ** 2, axis=-1)
            wr = np.exp(-color_diff_sq / two_sigma_r_sq)

            w = ws * wr
            weights_sum += w
            result += w * shifted_src

    filtered_depth = result / np.maximum(weights_sum, 1e-6)

    # Clean up non-finite values if any
    mask_finite = np.isfinite(filtered_depth)
    if not np.all(mask_finite):
        filtered_depth = np.where(mask_finite, filtered_depth, init_hr)

    return filtered_depth.astype(np.float32)
