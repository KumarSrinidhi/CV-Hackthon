# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/boundary_refinement.py
# Description: Edge-preserving boundary refinement using depth discontinuity detection and RGB guidance.

from typing import Optional
import cv2
import numpy as np


def detect_depth_discontinuities(depth: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Detect sharp depth discontinuities using Sobel gradient magnitude."""
    depth = np.asarray(depth, dtype=np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(dx * dx + dy * dy)
    return (grad_mag > threshold).astype(bool)


def detect_rgb_edges(rgb: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """Detect structural edges in RGB guidance using Canny edge detector."""
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb.astype(np.uint8)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return edges > 0


def refine_boundaries_classical(
    depth: np.ndarray,
    hr_rgb: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    depth_edge_thresh: float = 0.05,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Refine object boundaries in depth maps:
    Uses depth discontinuities as primary evidence and RGB edges as supporting evidence
    to sharpen boundaries while avoiding texture-copying artifacts.
    """
    depth = np.asarray(depth, dtype=np.float32).copy()
    h, w = depth.shape[:2]

    depth_edges = detect_depth_discontinuities(depth, threshold=depth_edge_thresh)
    rgb_edges = detect_rgb_edges(hr_rgb)

    # Edge agreement: only refine where depth discontinuity coincides with RGB boundary
    aligned_edges = depth_edges & rgb_edges
    
    # Slight morphological dilation to cover transition band
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blur_ksize, blur_ksize))
    edge_band = cv2.dilate(aligned_edges.astype(np.uint8), kernel).astype(bool)

    if not np.any(edge_band):
        return depth

    # Apply bilateral sharpening / guided filter on edge band
    smoothed = cv2.bilateralFilter(depth, d=5, sigmaColor=0.08, sigmaSpace=3.0)
    output = np.where(edge_band, 0.5 * depth + 0.5 * smoothed, depth)
    return output.astype(np.float32)
