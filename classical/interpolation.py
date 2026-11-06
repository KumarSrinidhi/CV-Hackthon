"""Nearest, bilinear and bicubic interpolation with missing-depth support."""

import cv2
import numpy as np

from .hole_filling import prepare_depth


_METHODS = {
    "nearest": cv2.INTER_NEAREST_EXACT,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def interpolate_depth(lr_depth, valid_mask=None, scale=4, method="bilinear"):
    """Return float32 depth, using zero when interpolation has no support."""
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, np.integer)) or scale < 1:
        raise ValueError("Scale must be a positive integer.")
    if method not in _METHODS:
        raise ValueError(f"Unknown interpolation method: {method!r}.")
    depth, valid = prepare_depth(lr_depth, valid_mask)
    height, width = depth.shape
    size = (width * int(scale), height * int(scale))
    numerator = cv2.resize(depth, size, interpolation=_METHODS[method])
    if method == "nearest":
        return numerator

    weight = cv2.resize(valid.astype(np.float32), size, interpolation=_METHODS[method])
    threshold = 0.25 if method == "bicubic" else 1e-6
    result = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > threshold)
    if method == "bicubic":
        linear = interpolate_depth(depth, valid, scale, "bilinear")
        result[weight <= threshold] = linear[weight <= threshold]
        if valid.any():
            supported = (weight > threshold) | (linear > 0)
            result[supported] = np.clip(result[supported], depth[valid].min(), depth.max())
    return result
