# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/metrics.py
# Description: Common evaluation metrics implementation (RMSE, MAE, PSNR, SSIM, boundary RMSE, edge-preservation).
"""Masked depth metrics (NumPy only).

Rules enforced here
-------------------
* All metrics use ONE evaluation mask = ground-truth validity & finite(pred) & finite(gt),
  optionally with a ``border`` of pixels removed (the official SGNet protocol crops 6 px).
* PSNR / SSIM / edge thresholds use a FIXED, documented ``data_range`` (NYU: 10 m,
  Middlebury/Lu: 255) -- never a per-image range.
* Depth discontinuities are defined from DEPTH only (ground truth for the reference,
  the prediction for its own edges). RGB edges are never used as ground-truth boundaries.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

DEFAULT_BORDER = 6


def eval_mask(
    pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None, border: int = DEFAULT_BORDER
) -> np.ndarray:
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} differ in shape")
    m = np.isfinite(pred) & np.isfinite(gt)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != gt.shape:
            raise ValueError(f"valid mask {valid.shape} != gt {gt.shape}")
        m &= valid
    if border > 0:
        if 2 * border >= min(gt.shape):
            raise ValueError(f"border={border} too large for image {gt.shape}")
        m[:border], m[-border:], m[:, :border], m[:, -border:] = False, False, False, False
    return m


def rmse(pred, gt, valid=None, border: int = DEFAULT_BORDER, mask: Optional[np.ndarray] = None) -> float:
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    if not m.any():
        return float("nan")
    e = pred[m].astype(np.float64) - gt[m].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def mae(pred, gt, valid=None, border: int = DEFAULT_BORDER, mask: Optional[np.ndarray] = None) -> float:
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(pred[m].astype(np.float64) - gt[m].astype(np.float64))))


def psnr(pred, gt, data_range: float, valid=None, border: int = DEFAULT_BORDER,
         mask: Optional[np.ndarray] = None) -> float:
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    if not m.any():
        return float("nan")
    mse = np.mean((pred[m].astype(np.float64) - gt[m].astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range**2 / mse))


def ssim(pred, gt, data_range: float, valid=None, border: int = DEFAULT_BORDER, win_size: int = 7,
         mask: Optional[np.ndarray] = None) -> float:
    """Mean SSIM over the evaluation mask eroded by the window radius (so no window
    includes excluded pixels)."""
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    if min(gt.shape) < win_size:
        return float("nan")
    p = np.where(m, pred, gt).astype(np.float64)  # excluded pixels cannot hurt the score
    _, smap = structural_similarity(gt.astype(np.float64), p, data_range=data_range, win_size=win_size, full=True)
    core = ndi.binary_erosion(m, structure=np.ones((win_size, win_size), bool))
    return float(smap[core].mean()) if core.any() else float("nan")


# ------------------------------------------------------------------- discontinuities
def depth_discontinuity_mask(
    depth: np.ndarray, valid: Optional[np.ndarray], data_range: float, rel_thresh: float = 0.03,
    radius: int = 0,
) -> np.ndarray:
    """True where |depth jump| to a 4-neighbour exceeds ``rel_thresh * data_range``.
    Pairs involving an invalid pixel never create an edge. ``radius`` dilates the result."""
    d = np.asarray(depth, dtype=np.float64)
    v = np.isfinite(d) if valid is None else (np.asarray(valid, bool) & np.isfinite(d))
    thr = rel_thresh * data_range
    e = np.zeros(d.shape, dtype=bool)
    dx = (np.abs(d[:, 1:] - d[:, :-1]) > thr) & v[:, 1:] & v[:, :-1]
    dy = (np.abs(d[1:, :] - d[:-1, :]) > thr) & v[1:, :] & v[:-1, :]
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    if radius > 0:
        e = ndi.binary_dilation(e, structure=ndi.generate_binary_structure(2, 2), iterations=radius)
    return e


def boundary_rmse(pred, gt, data_range: float, valid=None, border: int = DEFAULT_BORDER,
                  rel_thresh: float = 0.03, radius: int = 2, mask: Optional[np.ndarray] = None,
                  min_pixels: int = 50) -> float:
    """RMSE restricted to a band of ``radius`` px around GROUND-TRUTH depth discontinuities."""
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    band = depth_discontinuity_mask(gt, m, data_range, rel_thresh, radius) & m
    if band.sum() < min_pixels:
        return float("nan")
    e = pred[band].astype(np.float64) - gt[band].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def edge_scores(pred, gt, data_range: float, valid=None, border: int = DEFAULT_BORDER,
                rel_thresh: float = 0.03, tol: int = 1, mask: Optional[np.ndarray] = None,
                min_pixels: int = 20) -> Dict[str, float]:
    """Precision / recall / F1 of predicted depth edges vs ground-truth depth edges, with a
    ``tol``-pixel tolerance. Both edge maps come from depth with the same fixed threshold."""
    m = eval_mask(pred, gt, valid, border) if mask is None else mask
    ge = depth_discontinuity_mask(gt, m, data_range, rel_thresh) & m
    pe = depth_discontinuity_mask(pred, m, data_range, rel_thresh) & m
    if ge.sum() < min_pixels:
        return {"edge_precision": float("nan"), "edge_recall": float("nan"), "edge_f1": float("nan")}
    st = ndi.generate_binary_structure(2, 2)
    ge_d = ndi.binary_dilation(ge, st, iterations=tol) if tol > 0 else ge
    pe_d = ndi.binary_dilation(pe, st, iterations=tol) if tol > 0 else pe
    prec = float((pe & ge_d).sum() / pe.sum()) if pe.any() else 0.0
    rec = float((ge & pe_d).sum() / ge.sum())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1}


def compute_all_metrics(
    pred, gt, data_range: float, valid=None, border: int = DEFAULT_BORDER,
    rel_thresh: float = 0.03, boundary_radius: int = 2, edge_tol: int = 1,
    hole_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """All required metrics on one shared mask. Keys: rmse, mae, psnr, ssim,
    boundary_rmse, edge_precision, edge_recall, edge_f1, absrel, delta1,
    rmse_hole.

    Predictions are clamped to ``[0, data_range]`` before scoring (methodology
    A5). ``absrel`` / ``delta1`` are restricted to ``gt > 0`` pixels of the
    shared mask. ``rmse_hole`` restricts the mask to ``hole_mask`` (e.g. HR
    pixels whose source LR pixel was invalid) and is NaN when no hole mask is
    given or the intersection is empty.
    """
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, data_range)
    gt = np.asarray(gt, dtype=np.float32)
    m = eval_mask(pred, gt, valid, border)
    out = {
        "rmse": rmse(pred, gt, mask=m),
        "mae": mae(pred, gt, mask=m),
        "psnr": psnr(pred, gt, data_range, mask=m),
        "ssim": ssim(pred, gt, data_range, mask=m),
        "boundary_rmse": boundary_rmse(pred, gt, data_range, rel_thresh=rel_thresh, radius=boundary_radius, mask=m),
    }
    out.update(edge_scores(pred, gt, data_range, rel_thresh=rel_thresh, tol=edge_tol, mask=m))
    pos = m & (gt > 0)
    if pos.any():
        p = pred[pos].astype(np.float64)
        g = gt[pos].astype(np.float64)
        out["absrel"] = float(np.mean(np.abs(p - g) / g))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p > 0, np.maximum(p / g, g / np.maximum(p, 1e-12)), np.inf)
        out["delta1"] = float(np.mean(ratio < 1.25))
    else:
        out["absrel"] = float("nan")
        out["delta1"] = float("nan")
    if hole_mask is None:
        out["rmse_hole"] = float("nan")
    else:
        hm = np.asarray(hole_mask, dtype=bool)
        if hm.shape != gt.shape:
            raise ValueError(f"hole mask {hm.shape} != gt {gt.shape}")
        out["rmse_hole"] = rmse(pred, gt, mask=m & hm)
    return out
