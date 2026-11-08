"""Public classical pipeline wrapper.

    classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32
    reconstruct(lr_depth, hr_rgb, valid_mask=None, scale=4, method="jbu") -> (H, W) float32

The classical branch never modifies the caller's arrays. Measured depth values are
preserved; invalid pixels (<= 0, NaN, Inf, or flagged in ``valid_mask``) are handled
separately and are never treated as real measurements.
"""

from __future__ import annotations

import cv2
import numpy as np

from .boundary_refinement import refine_boundaries
from .guided_filter import guided_filter
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
    return depth, rgb


def reconstruct(lr_depth, hr_rgb, valid_mask=None, scale=4, method="jbu"):
    """Route to a classical reconstruction method.

    Supported methods: nearest, bilinear, bicubic, jbu, guided, classical,
    adaptive_jbu, refined_jbu.
    """
    depth, rgb = _validate(lr_depth, hr_rgb, valid_mask, scale)
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
        out = joint_bilateral_upsample(depth, valid_mask, rgb_f, radius=1)
        return refine_boundaries(out, out, rgb_f)

    raise ValueError(f"Unknown reconstruction method: {method!r}.")


def classical_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4):
    """Full classical pipeline: A1 hole-fill → A2 bicubic → A3 JBU → A4 boundary refinement."""
    depth, rgb = _validate(lr_depth, hr_rgb, valid_mask, scale)
    rgb_f = to_float_rgb(rgb)

    # A1: validity-aware hole filling (small holes only; original valid mask preserved)
    from .hole_filling import fill_small_holes, fill_holes_nn, prepare_depth
    depth_prep, valid = prepare_depth(depth, valid_mask)
    depth_filled, valid_filled = fill_small_holes(depth_prep, valid)

    # A1b: NN-fill any remaining large holes so A2/A3/A4 never see zeros inside holes.
    # valid_filled is NOT updated — it still reflects only real measurements + small fills.
    depth_seeded = fill_holes_nn(depth_filled, valid_filled) if not valid_filled.all() else depth_filled

    # A2: bicubic initial upsampling on the fully-seeded LR map
    from .interpolation import interpolate_depth
    upsampled = interpolate_depth(depth_seeded, None, scale, "bicubic")

    # A3: JBU RGB-guided reconstruction; seed from upsampled (no zeros), guide with valid_filled
    jbu_out = joint_bilateral_upsample(depth_seeded, valid_filled, rgb_f, radius=1)

    # A4: boundary-aware refinement; start from JBU, use upsampled as a smooth prior for holes
    out = refine_boundaries(jbu_out, jbu_out, rgb_f)
    # Pixels that JBU left at zero (no valid LR neighbor within radius) fall back to bicubic seed.
    out = np.where(out > 0, out, upsampled)
    return out
