# Team Owner: Rishita
# Workstream: Hybrid Fusion
# Module: hybrid/confidence.py
# Description: Branch confidence estimation using depth gradients, edge alignment, and measurement residuals.

from typing import Optional, Tuple
import cv2
import numpy as np


def compute_confidence_weights(
    d_c: np.ndarray,
    d_d: np.ndarray,
    hr_rgb: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    base_alpha: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized spatial confidence weights W_c and W_d for fusing
    classical HR depth D_c and deep HR depth D_d.
    
    Guarantees:
        W_c + W_d == 1.0 everywhere.
        0.0 <= W_c <= 1.0, 0.0 <= W_d <= 1.0.

    Args:
        d_c: High-resolution depth map from classical branch (H, W).
        d_d: High-resolution depth map from deep branch (H, W).
        hr_rgb: Optional RGB guidance image (H, W, 3).
        valid_mask: Optional boolean validity mask.
        base_alpha: Baseline blending weight for classical branch (default 0.02).

    Returns:
        (w_c, w_d) tuple of weight arrays matching input shape.
    """
    d_c = np.asarray(d_c, dtype=np.float32)
    d_d = np.asarray(d_d, dtype=np.float32)
    h, w = d_c.shape[:2]

    # Initialize with calibrated baseline alpha
    conf_c = np.full((h, w), base_alpha, dtype=np.float32)

    # Edge-based modulation if RGB guide is provided
    if hr_rgb is not None:
        hr_rgb = np.asarray(hr_rgb)
        if hr_rgb.ndim == 3:
            gray = cv2.cvtColor(hr_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = hr_rgb.astype(np.uint8)

        # Detect RGB edges
        rgb_dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        rgb_dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        rgb_mag = np.sqrt(rgb_dx * rgb_dx + rgb_dy * rgb_dy)
        rgb_mag_norm = rgb_mag / np.maximum(rgb_mag.max(), 1.0)

        # Depth discontinuity from classical output
        d_dx = cv2.Sobel(d_c, cv2.CV_32F, 1, 0, ksize=3)
        d_dy = cv2.Sobel(d_c, cv2.CV_32F, 0, 1, ksize=3)
        d_mag = np.sqrt(d_dx * d_dx + d_dy * d_dy)
        d_mag_norm = d_mag / np.maximum(d_mag.max(), 1e-4)

        # Classical JBU performs well at sharp aligned boundaries;
        # slightly boost classical weight along agreed edges up to 0.05
        edge_agreement = d_mag_norm * rgb_mag_norm
        conf_c += 0.03 * edge_agreement

    # Clamp classical weight to reasonable range [0.0, 0.2] so deep model remains anchor
    conf_c = np.clip(conf_c, 0.0, 0.2)
    conf_d = 1.0 - conf_c

    # Strictly normalize to sum to 1.0
    total = conf_c + conf_d
    w_c = conf_c / total
    w_d = conf_d / total

    return w_c.astype(np.float32), w_d.astype(np.float32)
