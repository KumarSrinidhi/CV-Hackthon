# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/pipeline.py
# Description: End-to-end classical depth upsampling pipeline and classical_hr_depth wrapper.

from typing import Optional
import numpy as np
from classical.interpolation import nearest_upsample, bilinear_upsample, bicubic_upsample
from classical.jbu import joint_bilateral_upsample
from classical.guided_filter import guided_filter_depth
from classical.hole_filling import conservative_hole_filling, mask_invalid_depth
from classical.boundary_refinement import refine_boundaries_classical


def classical_hr_depth(
    lr_depth: np.ndarray,
    hr_rgb: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    scale: int = 4,
    method: str = "jbu",
    refine_edges: bool = False,
) -> np.ndarray:
    """
    Produce a high-resolution depth map from low-resolution depth and RGB guidance
    using classical computer vision techniques.

    Args:
        lr_depth: Low-resolution depth map (H_lr, W_lr) as float32.
        hr_rgb: High-resolution RGB guidance image (H_hr, W_hr, 3) as uint8 or float32.
        valid_mask: Optional boolean mask of valid LR measurements.
        scale: Upsampling factor (default 4).
        method: Upsampling method - 'jbu', 'guided', 'bicubic', 'bilinear', or 'nearest'.
        refine_edges: Whether to apply post-upsampling boundary refinement.

    Returns:
        High-resolution depth map of shape (H_hr, W_hr) as float32.
    """
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    hr_rgb = np.asarray(hr_rgb)
    h_hr, w_hr = hr_rgb.shape[:2]

    # Pre-process LR depth holes if needed
    if valid_mask is not None:
        lr_clean, lr_valid = conservative_hole_filling(lr_depth, valid_mask=valid_mask)
    else:
        # Check for non-positive or non-finite values
        invalid = (~np.isfinite(lr_depth)) | (lr_depth <= 0)
        if np.any(invalid):
            lr_clean, _ = conservative_hole_filling(lr_depth, valid_mask=~invalid)
        else:
            lr_clean = lr_depth

    # Branch method execution
    method_lower = method.lower()
    if method_lower == "jbu":
        hr_depth = joint_bilateral_upsample(
            lr_clean, hr_rgb, scale=scale, target_size=(h_hr, w_hr), radius=3, sigma_spatial=2.5, sigma_range=0.1
        )
    elif method_lower == "guided":
        init_hr = bicubic_upsample(lr_clean, target_size=(h_hr, w_hr), scale=scale)
        hr_depth = guided_filter_depth(hr_rgb, init_hr, radius=6, eps=1e-3)
    elif method_lower == "bicubic":
        hr_depth = bicubic_upsample(lr_clean, target_size=(h_hr, w_hr), scale=scale)
    elif method_lower == "bilinear":
        hr_depth = bilinear_upsample(lr_clean, target_size=(h_hr, w_hr), scale=scale)
    elif method_lower == "nearest":
        hr_depth = nearest_upsample(lr_clean, target_size=(h_hr, w_hr), scale=scale)
    else:
        raise ValueError(f"Unknown classical method '{method}'. Supported: 'jbu', 'guided', 'bicubic', 'bilinear', 'nearest'.")

    # Optional boundary sharpening
    if refine_edges:
        hr_depth = refine_boundaries_classical(hr_depth, hr_rgb)

    # Ensure all values are finite and valid
    if not np.all(np.isfinite(hr_depth)):
        fallback = bicubic_upsample(lr_clean, target_size=(h_hr, w_hr), scale=scale)
        hr_depth = np.where(np.isfinite(hr_depth), hr_depth, fallback)

    return hr_depth.astype(np.float32)
