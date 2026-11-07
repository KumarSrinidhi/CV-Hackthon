# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/guided_filter.py
# Description: Edge-preserving guided filtering for depth map refinement using HR RGB guidance.

from typing import Optional
import cv2
import numpy as np


def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Fast box filtering using cv2.boxFilter."""
    ksize = 2 * radius + 1
    return cv2.boxFilter(img, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)


def guided_filter_depth(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Edge-preserving guided filter (He et al.) for depth maps.
    
    Args:
        guide: High-resolution RGB or grayscale image (uint8 or float32 in [0, 1]).
        src: Input depth map to filter (float32, same HxW as guide).
        radius: Radius of the square filter window.
        eps: Regularization parameter penalizing large gradients in a.
        valid_mask: Optional boolean validity mask.
        
    Returns:
        Filtered high-resolution depth map (float32).
    """
    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)

    # Convert RGB guide to grayscale luminance in [0, 1] if needed
    if guide.ndim == 3:
        if guide.shape[2] == 3:
            guide_gray = 0.299 * guide[:, :, 0] + 0.587 * guide[:, :, 1] + 0.114 * guide[:, :, 2]
        else:
            guide_gray = guide[:, :, 0]
    else:
        guide_gray = guide.copy()

    if guide_gray.max() > 1.0:
        guide_gray = guide_gray / 255.0

    # Box filtered statistics
    mean_I = box_filter(guide_gray, radius)
    mean_p = box_filter(src, radius)
    mean_Ip = box_filter(guide_gray * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(guide_gray * guide_gray, radius)
    var_I = mean_II - mean_I * mean_I

    # Linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Smooth coefficients
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    # Reconstruct filtered depth
    q = mean_a * guide_gray + mean_b

    # If validity mask provided, preserve initial depth at reliable pixels
    if valid_mask is not None:
        valid = valid_mask.astype(bool)
        # Avoid overriding valid measurements with distant values
        q = np.where(valid, 0.7 * src + 0.3 * q, q)

    return q.astype(np.float32)
