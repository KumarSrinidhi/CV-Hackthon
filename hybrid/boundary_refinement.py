"""Boundary-aware refinement of the fused depth map.

Depth discontinuities are the primary evidence; RGB structure is used only as
supporting evidence near those discontinuities. The refinement prevents smoothing
across foreground/background boundaries and does not copy RGB texture into depth.
"""

from __future__ import annotations

import numpy as np
import cv2

from classical.boundary_refinement import refine_boundaries
from classical.boundary_refinement import depth_boundaries


def refine_hybrid_depth(fused_depth, hr_rgb, valid_mask=None, *, selective=False, strength=1.0):
    """Refine fused depth, optionally limiting changes to a boundary band."""
    Df = np.asarray(fused_depth, dtype=np.float32)
    rgb = np.asarray(hr_rgb)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    refined = refine_boundaries(Df, Df, rgb).astype(np.float32)
    if selective:
        edges = depth_boundaries(Df)
        band = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        refined = np.where(band, refined, Df)
    return (Df + strength * (refined - Df)).astype(np.float32)
