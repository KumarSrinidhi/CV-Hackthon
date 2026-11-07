# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/preprocessing.py
# Description: RGB and depth normalization, alignment, and coordinate transformations.

from typing import Optional, Tuple
import cv2
import numpy as np


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB image to [0.0, 1.0] float32 array."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0:
        return rgb / 255.0
    return rgb


def normalize_depth(
    depth: np.ndarray,
    min_val: float = 0.5,
    max_val: float = 10.0,
) -> np.ndarray:
    """Normalize depth map to [0, 1] range for visualization or feature extraction."""
    depth = np.asarray(depth, dtype=np.float32)
    clipped = np.clip(depth, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)


def align_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure RGB image and depth map have identical spatial dimensions.
    """
    h_rgb, w_rgb = rgb.shape[:2]
    h_d, w_d = depth.shape[:2]

    if target_size is not None:
        target_h, target_w = target_size
    else:
        target_h, target_w = h_rgb, w_rgb

    if (h_rgb, w_rgb) != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if (h_d, w_d) != (target_h, target_w):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return rgb, depth
