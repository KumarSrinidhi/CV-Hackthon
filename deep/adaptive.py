# Team Owner: Kumar
# Workstream: Deep Learning
# Module: deep/adaptive.py
# Description: Regime-aware deep inference: deployable LR-only regime detection selects
# between the original (clean-specialist) and robust-curriculum checkpoints.
"""Regime-aware deep inference.

Evidence (docs/PERFORMANCE_ANALYSIS.md): the original SGNet checkpoint is best on
clean LR, while the curriculum fine-tuned checkpoint is best under noise and
missing-depth. The regime is observable from the LR input alone:

* holes   -> any invalid LR pixel (``valid_mask`` not all true)
* noise   -> median absolute Laplacian response over valid LR pixels above a
             threshold calibrated on validation data (clean LR is locally smooth;
             sensor noise is not)

Both detectors use inference-time information only (no GT). Threshold calibration
and per-regime benchmark numbers are in docs/EXPERIMENT_LOG.md (P3-E8).

The default deep_hr_depth behaviour is unchanged; use deep_hr_depth_adaptive for
the adaptive path (same signature contract).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .inference import deep_hr_depth
from .settings import ckpt_dir

_DEFAULT_CLEAN = ckpt_dir() / "SGNet_X4_R.pth"
_AMBIENT_ROBUST = Path("runs/e7_6epoch/best.pth")


def _default_robust() -> Path | None:
    """Ambient robust checkpoint, unless checkpoint isolation is requested.

    SGNET_CKPT_DIR / SGNET_CKPT_X4 pin the checkpoint set (used by the hermetic
    test-suite); in that case the ambient runs/ default is disabled so tests
    never read ambient weights."""
    if os.environ.get("SGNET_CKPT_DIR") or os.environ.get("SGNET_CKPT_X4"):
        return None
    return _AMBIENT_ROBUST if _AMBIENT_ROBUST.is_file() else None

# Calibrated on the NYU val split (50 images, P3-E8): clean |Laplacian| median spans
# 0.0077-0.0245 (max 0.0245); noise sigma=0.02 spans 0.0640-0.0818 (min 0.0640);
# noise sigma=0.05 spans 0.153-0.179. Geometric midpoint of the clean/noise-0.02 gap
# is ~4e-2; the classes are perfectly separated on validation data.
NOISE_THRESHOLD = 4e-2


def laplacian_score(lr: np.ndarray, valid: np.ndarray) -> float:
    """Median |Laplacian| over valid LR pixels (noise detector, LR-only)."""
    lap = cv2.Laplacian(lr.astype(np.float32), cv2.CV_32F, ksize=1)
    vals = np.abs(lap[valid & np.isfinite(lr)])
    if vals.size == 0:
        return float("inf")
    return float(np.median(vals))


def detect_regime(lr_depth: np.ndarray, valid_mask: np.ndarray | None = None,
                  noise_threshold: float = NOISE_THRESHOLD) -> str:
    lr = np.asarray(lr_depth, dtype=np.float32)
    if valid_mask is None:
        valid = np.isfinite(lr) & (lr > 0)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(lr) & (lr > 0)
    if not valid.all():
        return "holes"
    if laplacian_score(lr, valid) > noise_threshold:
        return "noise"
    return "clean"


def deep_hr_depth_adaptive(lr_depth, hr_rgb, valid_mask=None, scale: int = 4, *,
                           clean_ckpt: str | None = None, robust_ckpt: str | None = None,
                           noise_threshold: float = NOISE_THRESHOLD, **kwargs) -> np.ndarray:
    """deep_hr_depth with LR-only regime detection and per-regime checkpoints.

    The robust checkpoint resolves as: explicit argument > ``SGNET_CKPT_ROBUST``
    env > ambient runs/ default (disabled when checkpoint isolation env vars are
    set). If no robust checkpoint is available, the original checkpoint serves
    every regime."""
    regime = detect_regime(lr_depth, valid_mask, noise_threshold)
    robust = Path(robust_ckpt) if robust_ckpt else _default_robust()
    if scale == 4 and regime != "clean" and robust is not None:
        return deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale, ckpt_path=str(robust), **kwargs)
    return deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale, ckpt_path=clean_ckpt, **kwargs)
