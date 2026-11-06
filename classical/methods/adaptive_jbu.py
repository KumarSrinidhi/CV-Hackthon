"""Depth evidence and validation-selected settings for adaptive JBU."""

import cv2
import numpy as np

from ..boundary_refinement import depth_boundaries, refine_boundaries
from .jbu import joint_bilateral_upsample


DEPTH_JUMP = 0.10


def edge_evidence(depth, rgb):
    """Align original LR depth jumps with nearby HR Sobel color edges."""
    height, width = rgb.shape[:2]
    shape = (width, height)
    depth_edges = depth_boundaries(depth, DEPTH_JUMP)
    band = cv2.resize(depth_edges.astype(np.uint8), shape,
                      interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
    rgb_edges = (cv2.magnitude(gx, gy) > 0.035).astype(np.uint8)
    margin = max(1, round(height / depth.shape[0] / 2))
    window = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    agreement = band & cv2.dilate(rgb_edges, window).astype(bool)
    anchor = cv2.resize(depth, shape, interpolation=cv2.INTER_NEAREST_EXACT)
    return band, agreement, anchor


def adaptive_jbu(depth, valid, rgb, settings, *, refine=False):
    """Reduce RGB influence when nearby original depth edges lack agreement."""
    sigma_space, sigma_color, radius = settings
    band, agreement, anchor = edge_evidence(depth, rgb)
    # Use a tighter window near measured boundaries, especially when the RGB
    # edge disagrees. The depth anchor protects narrow measured structures there.
    spatial_factor = np.where(band, np.where(agreement, 0.85, 0.60), 1).astype(np.float32)
    color_strength = np.where(agreement, 1, 0.25).astype(np.float32)
    protected_anchor = np.where(band & ~agreement, anchor, 0)
    result = joint_bilateral_upsample(
        depth, valid, rgb, sigma_space=sigma_space, sigma_color=sigma_color, radius=radius,
        spatial_factor=spatial_factor, color_strength=color_strength,
        protected_anchor=protected_anchor, relative_jump=DEPTH_JUMP,
    )
    if not refine or not band.any():
        return result
    # Adjust only the boundary band. Reject neighbors across depth jumps and
    # retain the depth anchor on ambiguous edges.
    boundary_rgb = np.where(agreement[..., None], rgb, 0)
    filtered = refine_boundaries(result, result, boundary_rgb, relative_jump=DEPTH_JUMP)
    return np.where(agreement, filtered, result)
