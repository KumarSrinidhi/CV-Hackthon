"""Prepare independent depth and support arrays for the classical branch."""

import cv2
import numpy as np


def prepare_depth(depth, valid_mask=None):
    """Return positive finite depth and its effective measurement mask."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Depth must be a nonempty (H, W) array.")
    valid = np.isfinite(values) & (values > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != values.shape:
            raise ValueError("Validity mask must match the LR depth shape.")
        valid &= np.isfinite(mask) & (mask != 0)
    return np.where(valid, values, np.float32(0)), valid


def fill_small_holes(depth, valid, relative_span=0.08):
    """Fill isolated holes with at least five agreeing 3x3 neighbors."""
    kernel = np.ones((3, 3), np.uint8)
    kwargs = {"borderType": cv2.BORDER_CONSTANT, "borderValue": 0}
    count = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    total = cv2.filter2D(depth, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    maximum = cv2.dilate(depth, kernel, **kwargs)
    minimum = cv2.erode(np.where(valid, depth, np.inf), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=float("inf"))
    fill = (~valid) & (count >= 5) & ((maximum - minimum) <= relative_span * maximum)
    result = depth.copy()
    result[fill] = total[fill] / count[fill]
    return result, valid | fill

