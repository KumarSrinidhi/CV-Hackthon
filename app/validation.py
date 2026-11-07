"""Input validation for the application layer.

Turns raw user uploads into the frozen branch contract (see README) and raises
``ValueError`` with a clear, user-facing message on anything unusable. The branch
implementations are never edited; they just receive clean inputs.
"""

from __future__ import annotations

import numpy as np

METHODS = ("hybrid", "classical", "deep", "jbu", "guided", "adaptive_jbu",
           "refined_jbu", "bicubic", "bilinear", "nearest")


def validate_method(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}.")
    return method


def validate_scale(scale, method="hybrid", supported=None, deep_scales=(4, 8, 16)):
    """Return ``scale`` as int; reject non-integers and unsupported deep scales."""
    if isinstance(scale, bool) or isinstance(scale, float) and not float(scale).is_integer():
        raise ValueError(f"Scale must be a whole number, got {scale!r}.")
    try:
        scale = int(scale)
    except (TypeError, ValueError):
        raise ValueError(f"Scale must be a whole number, got {scale!r}.") from None
    if scale < 1:
        raise ValueError("Scale must be at least 1.")
    if supported and scale not in supported:
        raise ValueError(f"Scale x{scale} is not offered; choose one of {list(supported)}.")
    # Only the standalone deep method is rejected here. "hybrid" at an unsupported
    # scale is handled by the dispatcher, which degrades to classical *explicitly*.
    if method == "deep" and scale not in deep_scales:
        raise ValueError(
            f"The deep branch supports x{', x'.join(map(str, deep_scales))} only; "
            f"x{scale} is available for the classical methods."
        )
    return scale


def validate_rgb(rgb):
    """Return an (H, W, 3) uint8 or float32 [0, 1] RGB array."""
    if rgb is None:
        raise ValueError("RGB guidance image is required.")
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"RGB image must be (H, W, 3); got shape {a.shape}.")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = a[..., :3]
    if not a.size:
        raise ValueError("RGB image is empty.")
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("RGB image contains NaN/Inf values.")
    if a.max() <= 1.0 and a.min() >= 0.0:
        return np.ascontiguousarray(a)
    return np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))


def validate_depth(lr_depth):
    """Return a (h, w) float32 low-resolution depth map."""
    if lr_depth is None:
        raise ValueError("Low-resolution depth map is required.")
    d = np.asarray(lr_depth)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Depth map must be 2-D (h, w); got shape {d.shape}.")
    if not d.size:
        raise ValueError("Depth map is empty.")
    d = d.astype(np.float32)
    if not (np.isfinite(d) & (d > 0)).any():
        raise ValueError("Depth map has no valid (finite, > 0) pixels.")
    return d


def validate_mask(valid_mask, lr_shape):
    """Return a bool (h, w) mask (non-zero = valid) or ``None``."""
    if valid_mask is None:
        return None
    m = np.asarray(valid_mask)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.shape != tuple(lr_shape):
        raise ValueError(
            f"Validity mask shape {m.shape} must match the LR depth shape {tuple(lr_shape)}."
        )
    m = np.nan_to_num(m.astype(np.float32), nan=0.0) != 0
    if not m.any():
        raise ValueError("Validity mask marks every pixel invalid.")
    return m


def validate_geometry(rgb, lr, scale):
    """The HR RGB must be exactly ``scale`` times the LR depth in each dimension."""
    h, w = lr.shape
    H, W = rgb.shape[:2]
    if (H, W) != (h * scale, w * scale):
        raise ValueError(
            f"RGB is {W}x{H} but LR depth is {w}x{h}; at scale x{scale} the RGB must be "
            f"{w * scale}x{h * scale}. Check the scale or resize your inputs."
        )


def validate_inputs(rgb, lr_depth, valid_mask=None, scale=4,
                    supported=None, deep_scales=(4, 8, 16)):
    """Validate everything and return ``(rgb, lr, mask, scale)``."""
    scale = validate_scale(scale, "hybrid", supported, deep_scales)
    rgb = validate_rgb(rgb)
    lr = validate_depth(lr_depth)
    mask = validate_mask(valid_mask, lr.shape)
    validate_geometry(rgb, lr, scale)
    return rgb, lr, mask, scale
