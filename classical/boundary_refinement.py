"""Depth-first boundary detection and joint bilateral refinement."""

import cv2
import numpy as np


def depth_boundaries(depth, relative_jump=0.08):
    """Mark both sides of supported horizontal and vertical depth jumps."""
    edges = np.zeros(depth.shape, dtype=bool)
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        a, b = depth[first], depth[second]
        jump = (a > 0) & (b > 0) & (np.abs(a - b) > relative_jump * np.maximum(a, b))
        edges[first] |= jump
        edges[second] |= jump
    return edges


def refine_boundaries(depth, reference, rgb, *, relative_jump=0.08, sigma_color=0.15):
    """Smooth compatible samples and consult RGB Sobel edges near depth jumps."""
    edges = depth_boundaries(reference, relative_jump)
    near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.125)
    rgb_gate = near_edge & (cv2.magnitude(gx, gy) > 0.035)

    total, weights = depth.copy(), (depth > 0).astype(np.float32)
    height, width = depth.shape
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)):
        center = np.s_[max(0, -dy):min(height, height - dy), max(0, -dx):min(width, width - dx)]
        neighbor = np.s_[max(0, dy):min(height, height + dy), max(0, dx):min(width, width + dx)]
        center_ref, neighbor_ref = reference[center], reference[neighbor]
        a = np.where(center_ref > 0, center_ref, depth[center])
        b = np.where(neighbor_ref > 0, neighbor_ref, depth[neighbor])
        compatible = (np.abs(a - b) <= relative_jump * np.maximum(a, b))
        # Allow zero-depth pixels to receive weight from valid neighbors (one-sided gate).
        compatible &= depth[neighbor] > 0
        distance = np.sum((rgb[center] - rgb[neighbor]) ** 2, axis=2)
        color = np.where(rgb_gate[center] | rgb_gate[neighbor],
                         np.exp(-distance / (2 * sigma_color**2)), 1)
        weight = np.float32(np.exp(-0.5 * (dy**2 + dx**2))) * color * compatible
        total[center] += weight * depth[neighbor]
        weights[center] += weight
    return np.divide(total, weights, out=np.zeros_like(depth), where=weights > 0)
