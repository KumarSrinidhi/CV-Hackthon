"""Classical branch entry point for the application and hybrid branch."""

import cv2
import numpy as np

from .methods.adaptive_jbu import adaptive_jbu
from .methods.jbu import joint_bilateral_upsample
from .methods.masked_jbu import masked_jbu
from .methods.refined_jbu import refined_jbu
from .methods.tuned_jbu import DEFAULT_SETTINGS, SCALE_SETTINGS, tuned_jbu
from .boundary_refinement import depth_boundaries, refine_boundaries
from .guided_filter import guided_filter
from .hole_filling import fill_small_holes, prepare_depth
from .interpolation import interpolate_depth


JBU_VARIANTS = {"tuned_jbu": tuned_jbu, "adaptive_jbu": adaptive_jbu, "refined_jbu": refined_jbu}


def _normalize_rgb(hr_rgb, shape):
    rgb = np.asarray(hr_rgb)
    if rgb.shape != (*shape, 3):
        raise ValueError("HR RGB must have shape (LR height * scale, LR width * scale, 3).")
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255
    if not np.issubdtype(rgb.dtype, np.floating):
        raise ValueError("RGB must be uint8 [0, 255] or floating point [0, 1].")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Floating-point RGB must be finite and in [0, 1].")
    return rgb.astype(np.float32)


def reconstruct(lr_depth, hr_rgb, valid_mask, scale=4, *, method="classical", relative_jump=0.08, jbu_settings=None):
    """Run a classical method or component variant for comparison."""
    if method not in {"classical", "jbu", "guided", "depth_only", "no_refinement", "masked_jbu", "tuned_jbu", "adaptive_jbu", "refined_jbu"}:
        raise ValueError(f"Unknown reconstruction method: {method!r}.")
    depth, valid = prepare_depth(lr_depth, valid_mask)
    linear = interpolate_depth(depth, valid, scale, "bilinear")
    rgb = _normalize_rgb(hr_rgb, linear.shape)
    if not valid.any():
        return linear

    if method == "masked_jbu":
        return masked_jbu(depth, valid, rgb)
    if method in JBU_VARIANTS:
        settings = jbu_settings or SCALE_SETTINGS.get(scale, DEFAULT_SETTINGS)
        return JBU_VARIANTS[method](depth, valid, rgb, settings)

    depth, support = fill_small_holes(depth, valid)
    if method == "jbu":
        return joint_bilateral_upsample(depth, support, rgb)
    if method == "guided":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return guided_filter(linear, gray)

    linear = interpolate_depth(depth, support, scale, "bilinear")
    nearest = interpolate_depth(depth, support, scale, "nearest")
    edges = depth_boundaries(nearest, relative_jump)
    band = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # Only RGB-guided candidates near depth jumps can replace the depth-only base.
    base = np.where(band & (nearest > 0), nearest, linear)
    if method != "depth_only":
        jbu = joint_bilateral_upsample(depth, support, rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # An RGB edge may move a depth boundary when the nearby JBU samples
        # strongly favor one side. Reject blends across the discontinuity.
        kernel = np.ones((3, 3), np.uint8)
        low = cv2.erode(np.where(nearest > 0, nearest, np.inf), kernel)
        high = cv2.dilate(nearest, kernel)
        low = np.where(np.isfinite(low), low, 0)
        near_low = np.abs(jbu - low) <= 0.15 * (high - low)
        near_high = np.abs(jbu - high) <= 0.15 * (high - low)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
        color_edge = cv2.dilate(cv2.magnitude(gx, gy), kernel) > 0.035
        switch = band & color_edge & (low > 0) & (jbu > 0)
        switch &= (high - low) > relative_jump * high
        base = np.where(switch & near_low, low, base)
        base = np.where(switch & near_high, high, base)
    if method == "no_refinement":
        return base
    if method == "depth_only":
        rgb = np.zeros_like(rgb)
    return refine_boundaries(base, base, rgb, relative_jump=relative_jump)


def classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4):
    """Return an (H*scale, W*scale) float32 depth map in the input units."""
    return reconstruct(lr_depth, hr_rgb, valid_mask, scale, method="jbu")

