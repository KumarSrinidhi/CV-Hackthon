"""Validity-aware grayscale guided filter."""

import cv2
import numpy as np


def guided_filter(depth, guide, radius=3, eps=0.0025):
    """Fit local linear depth models over supported pixels only."""
    valid = (depth > 0).astype(np.float32)
    size = (2 * radius + 1,) * 2

    def box(values):
        return cv2.boxFilter(values, -1, size, normalize=False, borderType=cv2.BORDER_REFLECT)

    count = np.maximum(box(valid), 1)
    mean_i = box(guide * valid) / count
    mean_p = box(depth) / count
    variance = np.maximum(box(guide * guide * valid) / count - mean_i**2, 0)
    covariance = box(guide * depth) / count - mean_i * mean_p
    a = covariance / (variance + eps)
    b = mean_p - a * mean_i
    result = (box(a * valid) * guide + box(b * valid)) / count
    return np.where(valid > 0, np.maximum(result, 0), 0)
