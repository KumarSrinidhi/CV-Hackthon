"""Quality measurements for depth reconstruction."""

import math

import cv2
import numpy as np

from ..boundary_refinement import depth_boundaries


DEPTH_RANGE = 8.0


def ssim(prediction, target, *, data_range=DEPTH_RANGE, valid=None):
    def smooth(image):
        return cv2.GaussianBlur(image, (11, 11), 1.5)

    a, b = smooth(prediction), smooth(target)
    var_a = np.maximum(smooth(prediction**2) - a**2, 0)
    var_b = np.maximum(smooth(target**2) - b**2, 0)
    covariance = smooth(prediction * target) - a * b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    scores = ((2 * a * b + c1) * (2 * covariance + c2)) / (
        (a**2 + b**2 + c1) * (var_a + var_b + c2))
    if valid is not None:
        windows = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0)
        scores = scores[windows.astype(bool)]
    return float(np.mean(scores)) if scores.size else None


def quality(prediction, target):
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error**2)))
    boundary = depth_boundaries(target, 0.10)
    predicted_boundary = depth_boundaries(prediction, 0.10)
    matches = np.count_nonzero(boundary & predicted_boundary)
    denominator = np.count_nonzero(boundary) + np.count_nonzero(predicted_boundary)
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "psnr": 20 * math.log10(DEPTH_RANGE / rmse) if rmse else math.inf,
        "ssim": ssim(prediction, target),
        "edge_f1": 2 * matches / denominator if denominator else 1.0,
        "boundary_rmse": float(np.sqrt(np.mean(error[boundary] ** 2))) if boundary.any() else None,
        "coverage": float(np.mean(prediction > 0)),
    }


def ablation_quality(prediction, target, kind):
    """Score only known reference depths; coverage counts all output pixels."""
    valid = target > 0
    error = prediction - target
    edges = depth_boundaries(target, 0.10)
    predicted_edges = depth_boundaries(np.where(valid, prediction, 0), 0.10)
    denominator = edges.sum() + predicted_edges.sum()
    thin = valid & (target < 3) if kind == "thin" else np.zeros_like(valid)
    return dict(
        rmse=float(np.sqrt(np.mean(error[valid] ** 2))),
        mae=float(np.mean(np.abs(error[valid]))),
        boundary_rmse=float(np.sqrt(np.mean(error[edges] ** 2))) if edges.any() else None,
        edge_f1=float(2 * (edges & predicted_edges).sum() / denominator) if denominator else 1.0,
        thin_rmse=float(np.sqrt(np.mean(error[thin] ** 2))) if thin.any() else None,
        coverage=float((prediction > 0).mean()),
    )
