"""Boundary-aware refinement of the fused depth map.

Depth discontinuities are the primary evidence; RGB structure is used only as
supporting evidence near those discontinuities. The refinement prevents smoothing
across foreground/background boundaries and does not copy RGB texture into depth.
"""

from __future__ import annotations

import numpy as np

from classical.boundary_refinement import refine_boundaries


def refine_hybrid_depth(fused_depth, hr_rgb, valid_mask=None):
    """Refine ``fused_depth`` using depth-first boundaries and RGB support."""
    Df = np.asarray(fused_depth, dtype=np.float32)
    rgb = np.asarray(hr_rgb)
    return refine_boundaries(Df, Df, rgb).astype(np.float32)
