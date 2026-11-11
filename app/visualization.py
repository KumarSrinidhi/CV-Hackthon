"""Depth visualization, boundary overlay, error maps, and PNG/NPY export."""

from __future__ import annotations

import os

import cv2
import numpy as np


def _valid(depth, valid_mask=None):
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[..., 0]
        if m.shape != d.shape:
            raise ValueError(f"valid_mask shape {m.shape} != depth shape {d.shape}")
        valid &= m
    return d, valid


def normalize_depth(depth, valid_mask=None):
    d, valid = _valid(depth, valid_mask)
    if not valid.any():
        return np.zeros_like(d)
    lo, hi = float(d[valid].min()), float(d[valid].max())
    if hi - lo < 1e-6:
        return np.where(valid, 0.5, 0.0).astype(np.float32)
    return np.where(valid, (d - lo) / (hi - lo), 0.0).astype(np.float32)


def depth_colormap(depth, valid_mask=None, colormap=cv2.COLORMAP_TURBO):
    norm = normalize_depth(depth, valid_mask)
    return cv2.applyColorMap((np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8), colormap)


def shared_depth_colormap(depths, colormap=cv2.COLORMAP_TURBO):
    """Color-map depth arrays using one range shared by all valid pixels."""
    if not depths:
        raise ValueError("At least one depth map is required.")
    arrays = [np.asarray(d, dtype=np.float32) for d in depths]
    valid_values = [d[np.isfinite(d) & (d > 0)] for d in arrays]
    values = [v for v in valid_values if v.size]
    if not values:
        return [depth_colormap(d, colormap=colormap) for d in arrays]
    lo, hi = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    span = max(hi - lo, 1e-6)
    result = []
    for d in arrays:
        valid = np.isfinite(d) & (d > 0)
        norm = np.where(valid, (d - lo) / span, 0.0)
        result.append(cv2.applyColorMap((np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8), colormap))
    return result


def to_rgb_uint8(image):
    """Return an RGB uint8 image, converting grayscale inputs when needed."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Image must have shape (H, W, 3) or (H, W).")
    if img.dtype == np.float32 and img.max(initial=0) <= 1.0:
        img = img * 255.0
    return np.clip(img[..., :3], 0, 255).astype(np.uint8)


def side_by_side(images):
    """Arrange labeled BGR/RGB panels horizontally with a 22-pixel label bar."""
    if not images:
        raise ValueError("At least one image is required.")
    panels = []
    height = max(np.asarray(image).shape[0] for _, image in images)
    for label, image in images:
        panel = to_rgb_uint8(image)
        if panel.shape[0] != height:
            panel = cv2.resize(panel, (panel.shape[1], height), interpolation=cv2.INTER_NEAREST)
        bar = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, str(label), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack((bar, panel)))
    return np.hstack(panels)


def depth_boundaries_mask(depth, valid_mask=None, rel_thresh=0.08):
    d, valid = _valid(depth, valid_mask)
    edges = np.zeros(d.shape, dtype=bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        x, y = d[a], d[b]
        jump = valid[a] & valid[b] & (np.abs(x - y) > rel_thresh * np.maximum(x, y))
        edges[a] |= jump
        edges[b] |= jump
    return edges


def boundaries_overlay(rgb, depth, valid_mask=None, color=(0, 0, 255), thickness=1):
    base = to_rgb_uint8(rgb).copy()
    base[depth_boundaries_mask(depth, valid_mask)] = color
    return base


def error_map(pred, gt, valid_mask=None, max_error=None, colormap=cv2.COLORMAP_HOT):
    p, pv = _valid(pred, valid_mask)
    g, gv = _valid(gt, valid_mask)
    both = pv & gv
    err = np.abs(p - g)
    if max_error is None:
        max_error = float(err[both].max()) if both.any() else 1.0
    img = np.where(both, err / max(max_error, 1e-6), 0.0)
    return cv2.applyColorMap((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), colormap)


def confidence_overlay(rgb, confidence, colormap=cv2.COLORMAP_JET):
    heat = cv2.applyColorMap((np.clip(np.asarray(confidence), 0, 1) * 255).astype(np.uint8), colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(to_rgb_uint8(rgb), 0.4, heat, 0.6, 0.0)


def export_png(depth, path, valid_mask=None):
    d, valid = _valid(depth, valid_mask)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if valid.any():
        lo, hi = float(d[valid].min()), float(d[valid].max())
        img = np.where(valid, (d - lo) / max(hi - lo, 1e-6) * 65535, 0).astype(np.uint16)
    else:
        img = np.zeros(d.shape, np.uint16)
    cv2.imwrite(path, img)
    return path


def export_npy(depth, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32))
    return path
