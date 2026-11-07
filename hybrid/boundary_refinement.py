# Team Owner: Rishita
# Workstream: Hybrid Fusion
# Module: hybrid/boundary_refinement.py
# Description: Adaptive boundary refinement on fused depth maps using depth discontinuities as primary evidence and RGB as supporting evidence.

from typing import Optional
import cv2
import numpy as np


def adaptive_boundary_refinement(
    fused_depth: np.ndarray,
    hr_rgb: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    depth_thresh: float = 0.04,
    strength: float = 0.25,
) -> np.ndarray:
    """
    Adaptive boundary refinement for fused depth maps.
    Refines depth boundaries by finding true geometric depth steps and aligning them
    with high-contrast RGB edges, strictly avoiding texture copying in smooth depth regions.

    Args:
        fused_depth: High-resolution depth map of shape (H_hr, W_hr).
        hr_rgb: High-resolution RGB guidance image of shape (H_hr, W_hr, 3).
        valid_mask: Optional boolean validity mask.
        depth_thresh: Depth gradient threshold indicating a genuine physical boundary.
        strength: Blending strength for edge sharpening.

    Returns:
        Refined HR depth map (float32).
    """
    fused_depth = np.asarray(fused_depth, dtype=np.float32)
    hr_rgb = np.asarray(hr_rgb)
    h, w = fused_depth.shape[:2]

    # Detect physical depth discontinuities (Sobel gradient)
    dx = cv2.Sobel(fused_depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(fused_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = np.sqrt(dx * dx + dy * dy)
    depth_discontinuity = depth_grad > depth_thresh

    # Detect high-contrast RGB structural edges
    if hr_rgb.ndim == 3:
        gray = cv2.cvtColor(hr_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = hr_rgb.astype(np.uint8)

    rgb_grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad = np.sqrt(rgb_grad_x * rgb_grad_x + rgb_grad_y * rgb_grad_y)
    rgb_edge = rgb_grad > 35.0

    # Only refine in the narrow boundary zone where BOTH depth and RGB edges agree
    boundary_mask = depth_discontinuity & rgb_edge
    if not np.any(boundary_mask):
        return fused_depth

    # Cross bilateral / guided edge sharpening in boundary zone
    refined = cv2.bilateralFilter(fused_depth, d=5, sigmaColor=0.05, sigmaSpace=2.0)
    out = np.where(boundary_mask, (1.0 - strength) * fused_depth + strength * refined, fused_depth)

    return out.astype(np.float32)
