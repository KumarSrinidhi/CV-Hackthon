# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/hole_filling.py
# Description: Conservative depth hole filling and validity mask handling preserving genuine depth measurements.

from typing import Optional, Tuple
import cv2
import numpy as np


def mask_invalid_depth(
    depth: np.ndarray,
    min_val: float = 1e-3,
    max_val: float = 10.0,
) -> np.ndarray:
    """
    Generate boolean mask indicating valid, finite, in-range depth values.
    
    Args:
        depth: Input depth map.
        min_val: Minimum physically plausible depth (meters).
        max_val: Maximum physically plausible depth (meters).
        
    Returns:
        Boolean array of shape depth.shape (True for valid pixels).
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_val) & (depth <= max_val)
    return valid


def conservative_hole_filling(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_hole_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conservatively fill missing or zero depth holes using surrounding valid depth values.
    Preserves all original valid depth values without modification.

    Args:
        depth: Input depth map (H, W).
        valid_mask: Optional boolean validity mask (True = valid). If None, calculated via depth > 0.
        max_hole_radius: Maximum radius for morphological / inpainting hole filling.

    Returns:
        (filled_depth, updated_valid_mask)
    """
    depth = np.asarray(depth, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 1e-4)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 1e-4)

    if np.all(valid_mask):
        return depth, valid_mask

    # Distance transform to find nearest valid depth coordinates
    invalid_mask_u8 = (~valid_mask).astype(np.uint8)
    
    # Inpainting using Navier-Stokes based inpainting for smooth, conservative filling
    depth_zero_filled = np.where(valid_mask, depth, 0.0).astype(np.float32)
    
    # Inpaint only invalid regions
    inpaint_radius = max(1, min(max_hole_radius, 5))
    filled_depth = cv2.inpaint(depth_zero_filled, invalid_mask_u8, inpaint_radius, cv2.INPAINT_NS)
    
    # Strictly ensure original valid measurements are unchanged
    output_depth = np.where(valid_mask, depth, filled_depth)
    
    # Any residual NaN/inf fallback to median of valid pixels
    invalid_remains = ~np.isfinite(output_depth) | (output_depth <= 0)
    if np.any(invalid_remains):
        median_val = float(np.median(depth[valid_mask])) if np.any(valid_mask) else 1.0
        output_depth[invalid_remains] = median_val

    updated_valid_mask = valid_mask | (~invalid_mask_u8.astype(bool))
    return output_depth.astype(np.float32), updated_valid_mask
