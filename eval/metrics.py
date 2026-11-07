# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/metrics.py
# Description: Official benchmark metrics: RMSE, MAE, PSNR, SSIM, Boundary RMSE, and Edge F1 score.

from typing import Dict, Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def _align_mask(mask: Optional[np.ndarray], target_shape: tuple) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != target_shape:
        h, w = target_shape
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask.astype(bool)


def compute_rmse(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Root Mean Squared Error (RMSE) in meters."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = _align_mask(mask, gt.shape)
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    else:
        mask = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

    if not np.any(mask):
        return 0.0

    diff = pred[mask] - gt[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def compute_mae(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Absolute Error (MAE) in meters."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = _align_mask(mask, gt.shape)
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    else:
        mask = mask & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def compute_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if isinstance(max_val, np.ndarray) and mask is None:
        mask = max_val
        max_val = None

    rmse = compute_rmse(pred, gt, mask=mask)
    if rmse < 1e-8:
        return 100.0
    if max_val is None:
        valid_gt = gt[np.isfinite(gt) & (gt > 0)]
        max_val = float(valid_gt.max()) if len(valid_gt) > 0 else 10.0
    return float(20.0 * np.log10(max_val / rmse))


def compute_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    if isinstance(max_val, np.ndarray) and mask is None:
        mask = max_val
        max_val = None
    pred_clean = np.nan_to_num(pred.copy(), nan=0.0)
    gt_clean = np.nan_to_num(gt.copy(), nan=0.0)
    if max_val is None:
        valid_gt = gt[np.isfinite(gt) & (gt > 0)]
        max_val = float(valid_gt.max()) if len(valid_gt) > 0 else 10.0

    data_range = float(max_val)
    val = ssim_fn(gt_clean, pred_clean, data_range=data_range)
    return float(val)


def extract_depth_edges(depth: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Extract binary edge map of depth discontinuities using Sobel operator."""
    depth = np.asarray(depth, dtype=np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(dx * dx + dy * dy)
    return mag > threshold


def compute_boundary_rmse(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.05,
    dilation_radius: int = 2,
) -> float:
    """Compute RMSE evaluated strictly within the neighborhood of ground-truth depth boundaries."""
    gt_edges = extract_depth_edges(gt, threshold=threshold)
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * dilation_radius + 1, 2 * dilation_radius + 1)
        )
        boundary_zone = cv2.dilate(gt_edges.astype(np.uint8), kernel).astype(bool)
    else:
        boundary_zone = gt_edges

    if mask is not None:
        boundary_zone = boundary_zone & _align_mask(mask, gt.shape)

    return compute_rmse(pred, gt, mask=boundary_zone)


def compute_edge_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05) -> float:
    """
    Compute Edge F1 score evaluating boundary recovery precision and recall.
    """
    pred_edges = extract_depth_edges(pred, threshold=threshold)
    gt_edges = extract_depth_edges(gt, threshold=threshold)

    tp = np.logical_and(pred_edges, gt_edges).sum()
    fp = np.logical_and(pred_edges, np.logical_not(gt_edges)).sum()
    fn = np.logical_and(np.logical_not(pred_edges), gt_edges).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


def compute_all_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    hr_rgb: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all standard benchmark metrics comparing reconstructed depth against ground truth.
    """
    rmse = compute_rmse(pred_depth, gt_depth, mask=valid_mask)
    mae = compute_mae(pred_depth, gt_depth, mask=valid_mask)
    psnr = compute_psnr(pred_depth, gt_depth, mask=valid_mask)
    ssim = compute_ssim(pred_depth, gt_depth, mask=valid_mask)
    b_rmse = compute_boundary_rmse(pred_depth, gt_depth, mask=valid_mask)
    f1 = compute_edge_f1(pred_depth, gt_depth)

    return {
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "boundary_rmse": b_rmse,
        "edge_f1": f1,
    }
