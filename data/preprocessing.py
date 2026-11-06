# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/preprocessing.py
# Description: RGB-depth alignment, consistent depth units, normalization, and generation/preservation of original validity masks.
"""Shared preprocessing helpers (pure NumPy/SciPy/OpenCV, no torch).

FROZEN CONVENTIONS (used by every branch)
-----------------------------------------
* hr_rgb     : (H, W, 3) RGB. ``float32`` in [0, 1] after ``to_float_rgb``
               (uint8 [0,255] inputs are accepted and converted).
* lr_depth   : (h, w) ``float32`` in the dataset's native metric unit
               (metres for NYU-v2; 0-255 grey levels for Middlebury/Lu).
               Invalid pixels are stored as 0.0, but ``valid_mask`` is authoritative.
* valid_mask : (h, w) ``bool`` at LOW resolution, True = measured/valid.
               Created from the ORIGINAL data before any filling/interpolation.
* scale      : int, H == h * scale and W == w * scale.
* outputs    : (H, W) ``float32`` NumPy array, same unit as ``lr_depth``.
Original arrays are never modified in place; filled/normalised arrays are copies.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

_UNIT_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


# ----------------------------------------------------------------------------- RGB
def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return float32 (H, W, 3) RGB in [0, 1]. Accepts uint8, float [0,1] or float [0,255]."""
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"hr_rgb must be (H, W, 3); got shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("hr_rgb contains NaN/Inf")
    mx = float(a.max()) if a.size else 0.0
    if mx > 1.0 + 1e-4:
        if mx > 255.0 + 1e-3:
            raise ValueError(f"hr_rgb float values up to {mx:.1f}: expected [0,1] or [0,255]")
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def align_rgb_to_depth(rgb: np.ndarray, depth_shape: Tuple[int, int]) -> np.ndarray:
    """Resize RGB to ``depth_shape`` (H, W) when a dataset ships mismatched sizes."""
    H, W = depth_shape
    if rgb.shape[:2] == (H, W):
        return rgb
    interp = cv2.INTER_AREA if rgb.shape[0] > H else cv2.INTER_CUBIC
    return cv2.resize(rgb, (W, H), interpolation=interp)


# --------------------------------------------------------------------------- depth
def to_depth_float(depth: np.ndarray) -> np.ndarray:
    a = np.asarray(depth)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"depth must be (h, w) or (h, w, 1); got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def to_metres(depth: np.ndarray, unit: str) -> np.ndarray:
    if unit not in _UNIT_TO_METRES:
        raise ValueError(f"unit must be one of {sorted(_UNIT_TO_METRES)}; got '{unit}'")
    return (np.asarray(depth, dtype=np.float32) * np.float32(_UNIT_TO_METRES[unit])).astype(np.float32)


def make_valid_mask(
    depth: np.ndarray, min_valid: Optional[float] = 0.0, max_valid: Optional[float] = None
) -> np.ndarray:
    """Original validity mask: finite and ``depth > min_valid`` (and ``<= max_valid``).

    ``min_valid=None`` disables the lower bound (used for 8-bit Middlebury/Lu, where
    0 is a legitimate grey level and the official protocol evaluates every pixel).
    """
    d = to_depth_float(depth)
    m = np.isfinite(d)
    if min_valid is not None:
        m &= d > min_valid
    if max_valid is not None:
        m &= d <= max_valid
    return m


def resolve_valid_mask(depth: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Mask for the wrapper API. ``None`` -> derived from depth (finite & > 0).
    An explicit mask is honoured but non-finite depth is always invalid."""
    d = to_depth_float(depth)
    if valid_mask is None:
        return make_valid_mask(d)
    m = np.asarray(valid_mask)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.shape != d.shape:
        raise ValueError(f"valid_mask shape {m.shape} != lr_depth shape {d.shape}")
    return m.astype(bool) & np.isfinite(d)


def check_geometry(lr_depth: np.ndarray, hr_rgb: np.ndarray, scale: int) -> None:
    if not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError(f"scale must be a positive int; got {scale!r}")
    h, w = lr_depth.shape[:2]
    H, W = hr_rgb.shape[:2]
    if (H, W) != (h * scale, w * scale):
        raise ValueError(
            f"Geometry mismatch: lr_depth is {h}x{w}, scale x{scale} implies an RGB of "
            f"{h * scale}x{w * scale}, but hr_rgb is {H}x{W}."
        )


def modcrop(arr: np.ndarray, modulo: int) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: h - h % modulo, : w - w % modulo]


def upsample_mask_nearest(mask: np.ndarray, scale: int) -> np.ndarray:
    """LR bool mask -> HR bool mask (each LR pixel covers a scale x scale block)."""
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


# ---------------------------------------------------------------- working copies
def fill_invalid_nearest(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Working copy where invalid pixels take the nearest valid value (conservative:
    no new depth values are invented). The inputs are left untouched."""
    d = to_depth_float(depth)
    if valid_mask.all():
        return d.copy()
    if not valid_mask.any():
        raise ValueError("lr_depth has no valid pixels; nothing to reconstruct from.")
    idx = ndi.distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    return d[idx[0], idx[1]].astype(np.float32)


def minmax_range(
    depth: np.ndarray, valid_mask: np.ndarray, norm_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """(lo, hi) used for normalisation: valid-pixel min/max unless ``norm_range`` given."""
    if norm_range is not None:
        lo, hi = float(norm_range[0]), float(norm_range[1])
    else:
        v = depth[valid_mask]
        lo, hi = float(v.min()), float(v.max())
    if not hi - lo > 1e-6:
        hi = lo + 1.0  # constant depth map: avoid division by ~0
    return lo, hi


def minmax_normalize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((depth.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo)).astype(np.float32)


def minmax_denormalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x.astype(np.float32) * np.float32(hi - lo) + np.float32(lo)).astype(np.float32)
