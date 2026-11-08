"""Public classical pipeline wrapper.

    classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
    reconstruct(lr_depth, hr_rgb, valid_mask=None, scale=4, method="jbu") -> (H, W) float32

The classical branch never modifies the caller's arrays. Invalid pixels
(<= 0, NaN, Inf, or flagged in ``valid_mask``) are excluded as measurements.
"""

from __future__ import annotations

import cv2
import numpy as np

from .boundary_refinement import refine_boundaries
from .edge_upsampling import upsample_blend_depth, upsample_dense_depth, upsample_masked_depth
from .guided_filter import guided_filter
from .hole_filling import prepare_depth
from .interpolation import interpolate_depth
from .jbu import joint_bilateral_upsample, to_float_rgb

_INTERPOLATION_METHODS = {"nearest", "bilinear", "bicubic"}


def _validate(lr_depth, hr_rgb, valid_mask, scale):
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("Scale must be a positive integer.")
    depth = np.asarray(lr_depth, dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("Depth must be a nonempty (H, W) array.")
    rgb = np.asarray(hr_rgb)
    if rgb.ndim != 3 or rgb.shape[2] not in (1, 3, 4):
        raise ValueError(f"RGB shape {rgb.shape} must be (H, W, 3).")
    h, w = depth.shape
    H, W = rgb.shape[:2]
    if (H, W) != (h * scale, w * scale):
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} does not match LR depth shape {depth.shape} "
            f"at scale x{scale}."
        )
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        if mask.shape != depth.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} must match LR depth shape {depth.shape}."
            )
        valid_mask = mask
    return depth, rgb, valid_mask


def reconstruct(lr_depth, hr_rgb, valid_mask=None, scale=4, method="jbu"):
    """Route to a classical reconstruction method.

    Supported methods: nearest, bilinear, bicubic, jbu, guided, classical,
    classical_blend, adaptive_jbu, refined_jbu.
    """
    depth, rgb, valid_mask = _validate(lr_depth, hr_rgb, valid_mask, scale)
    rgb_f = to_float_rgb(rgb)

    if method in _INTERPOLATION_METHODS:
        return interpolate_depth(depth, valid_mask, scale, method)

    if method == "jbu":
        return joint_bilateral_upsample(depth, valid_mask, rgb_f, radius=1)

    if method == "guided":
        bilinear = interpolate_depth(depth, valid_mask, scale, "bilinear")
        gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)
        return guided_filter(bilinear, gray)

    if method in ("adaptive_jbu", "refined_jbu"):
        out = joint_bilateral_upsample(depth, valid_mask, rgb_f, radius=1, use_depth=True)
        if method == "refined_jbu":
            out = refine_boundaries(out, out, rgb_f)
        return out

    if method == "classical":
        depth, valid = prepare_depth(depth, valid_mask)
        if scale == 1:
            return depth
        if valid.all():
            return upsample_dense_depth(depth, rgb_f, scale)
        return upsample_masked_depth(depth, valid, rgb_f, scale)

    if method == "classical_blend":
        if scale == 1:
            d, _ = prepare_depth(depth, valid_mask)
            return d
        return upsample_blend_depth(depth, valid_mask, rgb_f, scale)

    raise ValueError(f"Unknown reconstruction method: {method!r}.")


def classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4):
    """Edge-preserving classical depth upsampling."""
    return reconstruct(lr_depth, hr_rgb, valid_mask, scale, method="classical")
