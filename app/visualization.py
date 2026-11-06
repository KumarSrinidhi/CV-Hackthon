"""Depth visualization, boundary overlay, error maps, and PNG/NPY export.

Pure NumPy / OpenCV / PIL -- no Gradio dependency -- so the core is testable and
reusable by the CLI, the Gradio UI, and the evaluation scripts.
"""

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
    """Normalize valid depth to [0, 1]; invalid pixels become 0."""
    d, valid = _valid(depth, valid_mask)
    if not valid.any():
        return np.zeros_like(d)
    lo = float(d[valid].min())
    hi = float(d[valid].max())
    if hi - lo < 1e-6:
        return np.where(valid, 0.5, 0.0).astype(np.float32)
    out = (d - lo) / (hi - lo)
    return np.where(valid, out, 0.0).astype(np.float32)


def depth_colormap(depth, valid_mask=None, colormap=cv2.COLORMAP_TURBO):
    """Return an (H, W, 3) uint8 BGR colormap of the depth (invalid -> black)."""
    norm = normalize_depth(depth, valid_mask)
    img = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img, colormap)


def depth_boundaries_mask(depth, valid_mask=None, rel_thresh=0.08):
    """Boolean mask marking both sides of horizontal/vertical depth discontinuities."""
    d, valid = _valid(depth, valid_mask)
    edges = np.zeros(d.shape, dtype=bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        x, y = d[a], d[b]
        va, vb = valid[a], valid[b]
        jump = va & vb & (np.abs(x - y) > rel_thresh * np.maximum(x, y))
        edges[a] |= jump
        edges[b] |= jump
    return edges


def boundaries_overlay(rgb, depth, valid_mask=None, color=(0, 0, 255), thickness=1):
    """Overlay depth-discontinuity boundaries on ``rgb`` (channel order preserved).

    ``color`` is applied to the returned array as given: pass (255, 0, 0) for red
    when ``rgb`` is RGB, (0, 0, 255) when it is BGR (e.g. destined for imwrite).
    """
    base = np.asarray(rgb)
    if base.dtype == np.float32 and base.max() <= 1.0:
        base = (np.clip(base, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    base = base[..., :3].copy()
    edges = depth_boundaries_mask(depth, valid_mask)
    base[edges] = color
    return base


def error_map(pred, gt, valid_mask=None, max_error=None, colormap=cv2.COLORMAP_HOT):
    """Colour map of |pred - gt| (BGR); pixels with no GT are black."""
    p, pv = _valid(pred, valid_mask)
    g, gv = _valid(gt, valid_mask)
    both = pv & gv
    err = np.abs(p - g)
    if max_error is None:
        max_error = float(err[both].max()) if both.any() else 1.0
    err = np.where(both, err / max(max_error, 1e-6), 0.0)
    img = (np.clip(err, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img, colormap)


def confidence_overlay(rgb, confidence, colormap=cv2.COLORMAP_JET):
    """Overlay a [0, 1] confidence map on an RGB base image; returns RGB.

    ``rgb`` is treated as an RGB (H, W, 3) array (the Gradio UI convention), and
    the colormap heat is converted from OpenCV's BGR to RGB before blending so
    the displayed colours are correct.
    """
    base = np.asarray(rgb)
    if base.dtype == np.float32 and base.max() <= 1.0:
        base = (np.clip(base, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    c = np.clip(np.asarray(confidence, np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap((c * 255.0).astype(np.uint8), colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base[..., :3], 0.4, heat, 0.6, 0.0)


def export_png(depth, path, valid_mask=None):
    """Export depth as a 16-bit PNG (millimetre-scaled where appropriate) plus a
    8-bit colormap preview sidecar is *not* created; returns the path."""
    d, valid = _valid(depth, valid_mask)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if valid.any():
        lo, hi = float(d[valid].min()), float(d[valid].max())
        if hi - lo > 1e-6:
            scaled = np.where(valid, (d - lo) / (hi - lo), 0.0)
            img = (scaled * 65535.0).astype(np.uint16)
        else:
            img = np.where(valid, 32768, 0).astype(np.uint16)
    else:
        img = np.zeros(d.shape, np.uint16)
    cv2.imwrite(path, img)
    return path


def export_npy(depth, path):
    """Export depth as float32 NPY. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32))
    return path
