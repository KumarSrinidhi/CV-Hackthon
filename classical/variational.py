"""RGB-guided reconstruction from masked, antialiased depth measurements."""

from functools import lru_cache

import cv2
import numpy as np
from scipy.sparse import csr_matrix

from data.preprocessing import estimate_relative_noise

from .hole_filling import fill_holes_nearest, fill_small_holes


@lru_cache(maxsize=32)
def _resize_axis(size, scale, mode):
    """Sparse 1D equivalent of Pillow's float resize, including at borders."""
    count = size // scale
    if mode == "nearest":
        rows = np.arange(count)
        return csr_matrix(
            (np.ones(count, np.float32), (rows, rows * scale + scale // 2)),
            shape=(count, size),
        )

    centers = (np.arange(count) + 0.5) * scale - 0.5
    radius = scale if mode == "bilinear" else 2 * scale
    columns = np.floor(centers).astype(int)[:, None] + np.arange(1 - radius, radius + 1)
    x = np.abs(columns - centers[:, None]) / scale
    if mode == "bilinear":
        weights = np.maximum(1 - x, 0)
    else:
        weights = np.where(x < 1, 1 + x**2 * (1.5 * x - 2.5),
                           np.where(x < 2, (x - 2)**2 * (0.5 - 0.5 * x), 0))
    keep = (columns >= 0) & (columns < size) & (weights != 0)
    weights = np.where(keep, weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    rows = np.broadcast_to(np.arange(count)[:, None], columns.shape)
    return csr_matrix(
        (weights[keep].astype(np.float32), (rows[keep], columns[keep])), shape=(count, size)
    )


def _resize(image, rows, cols):
    return (cols @ (rows @ image).T).T


def _dot(x, y):
    return float(np.einsum("ij,ij->", x, y, dtype=np.float64))


def _solve(apply, target, estimate, diagonal, iterations):
    """Jacobi-preconditioned conjugate gradients, warm started from depth."""
    residual = target - apply(estimate)
    direction = residual / diagonal
    energy = _dot(residual, direction)
    tolerance = max(energy * 1e-6, 1e-15)
    for _ in range(iterations):
        if energy <= tolerance:
            break
        update = apply(direction)
        step = energy / _dot(direction, update)
        estimate = estimate + step * direction
        residual -= step * update
        preconditioned = residual / diagonal
        next_energy = _dot(residual, preconditioned)
        direction = preconditioned + (next_energy / energy) * direction
        energy = next_energy
    return estimate


def _guided_hole_seed(filled, support, rgb):
    """Diffuse depth into holes on the LR grid while keeping supported values fixed."""
    guide = cv2.resize(rgb, filled.shape[::-1], interpolation=cv2.INTER_AREA)
    wx = 1 / (1 + (np.linalg.norm(np.diff(guide, axis=1), axis=2) / 0.12)**2)
    wy = 1 / (1 + (np.linalg.norm(np.diff(guide, axis=0), axis=2) / 0.12)**2)
    anchor = 0.025
    diagonal = np.full_like(filled, anchor)
    diagonal[:, :-1] += wx
    diagonal[:, 1:] += wx
    diagonal[:-1, :] += wy
    diagonal[1:, :] += wy
    target = anchor * filled
    for _ in range(32):
        update = target.copy()
        update[:, :-1] += wx * filled[:, 1:]
        update[:, 1:] += wx * filled[:, :-1]
        update[:-1, :] += wy * filled[1:, :]
        update[1:, :] += wy * filled[:-1, :]
        filled = np.where(support, filled, update / diagonal)
    return filled


def restore_depth(depth, valid, rgb, scale, mode):
    """Minimize measurement error and a robust, RGB-guided depth gradient penalty."""
    height, width = rgb.shape[:2]
    values = depth[valid]
    if not values.size:
        return np.zeros((height, width), np.float32)
    if values.min() == values.max():
        return np.full((height, width), values[0], np.float32)
    if scale == 1 and valid.all():
        return depth.copy()

    # Work in relative depth units so the weights transfer between metres,
    # millimetres and disparity. Filled samples initialize the solver only.
    center = np.median(values)
    low, high = np.percentile(values, [5, 95]).astype(np.float32)
    depth_range = max(high - low, 0.05 * center)
    filled, support = fill_small_holes(depth, valid)
    if not support.all():
        filled = _guided_hole_seed(fill_holes_nearest(filled, support), support, rgb)
    seed = cv2.resize((filled - center) / depth_range, (width, height), interpolation=cv2.INTER_CUBIC)
    measured = np.where(valid, (depth - center) / depth_range, 0)

    rows = _resize_axis(height, scale, mode)
    cols = _resize_axis(width, scale, mode)
    rows_t, cols_t = rows.T.tocsr(), cols.T.tocsr()
    data_strength = float(scale**2)
    anchor = 0.0005
    target = data_strength * _resize(measured, rows_t, cols_t) + anchor * seed
    data_diagonal = data_strength * _resize(
        valid.astype(np.float32), rows_t.power(2), cols_t.power(2)
    ) + anchor

    # RGB texture can weaken smoothing only near a discontinuity visible in depth.
    guide = cv2.GaussianBlur(rgb, (3, 3), 0.65)
    dx = np.diff(seed, axis=1, append=seed[:, -1:])
    dy = np.diff(seed, axis=0, append=seed[-1:, :])
    depth_edge = cv2.dilate(np.hypot(dx, dy), np.ones((2 * scale + 1, 2 * scale + 1), np.uint8))
    gate = np.clip((depth_edge * scale - 0.025) / 0.10, 0, 1)
    contrast_x = np.linalg.norm(np.diff(guide, axis=1), axis=2)
    contrast_y = np.linalg.norm(np.diff(guide, axis=0), axis=2)
    noise = estimate_relative_noise(depth, valid)
    missing = 1.0 - float(valid.mean())
    clean = np.clip(1.0 - max((noise - 0.003) / 0.006, (missing - 0.04) / 0.08), 0, 1)
    color_bandwidth = 0.055 + 0.015 * clean
    guide_x = 1 / (1 + gate[:, :-1] * (contrast_x / color_bandwidth)**2)
    guide_y = 1 / (1 + gate[:-1, :] * (contrast_y / color_bandwidth)**2)

    # Reweight the gradient penalty after each solve. Small gradients are
    # smoothed strongly, while persistent depth jumps become cheaper to keep.
    threshold = 0.025 / scale
    smoothness = (0.25 - 0.13 * clean) * scale**2
    estimate = seed
    for _ in range(3):
        wx = smoothness * guide_x / np.hypot(1, np.diff(estimate, axis=1) / threshold)
        wy = smoothness * guide_y / np.hypot(1, np.diff(estimate, axis=0) / threshold)
        diagonal = data_diagonal.copy()
        diagonal[:, :-1] += wx
        diagonal[:, 1:] += wx
        diagonal[:-1, :] += wy
        diagonal[1:, :] += wy

        def apply(image):
            projected = valid * _resize(image, rows, cols)
            out = data_strength * _resize(projected, rows_t, cols_t) + anchor * image
            flux_x = wx * np.diff(image, axis=1)
            flux_y = wy * np.diff(image, axis=0)
            out[:, :-1] -= flux_x
            out[:, 1:] += flux_x
            out[:-1, :] -= flux_y
            out[1:, :] += flux_y
            return out

        estimate = _solve(apply, target, estimate, diagonal, iterations=24)

    # The inverse sampling model can ring at sharp transitions. Bound its
    # extrapolation by the range of actual measurements.
    result = np.clip(estimate * depth_range + center, values.min(), values.max())
    if scale == 1:
        result[valid] = values
    return result
