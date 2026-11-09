"""Input-quality fusion weights and spatial confidence for ablations."""

from __future__ import annotations

import cv2
import numpy as np

from classical.adaptive_jbu import edge_evidence
from data.preprocessing import estimate_relative_noise


def quality_blend(lr_depth, valid_mask=None):
    """Choose classical weight and consistency damping from observed LR quality."""
    depth = np.asarray(lr_depth, dtype=np.float32)
    valid = _resolve_valid(depth, valid_mask)
    noise = estimate_relative_noise(depth, valid)
    missing = 1.0 - float(valid.mean())
    noise_level = float(np.clip((noise - 0.003) / 0.006, 0, 1))
    hole_level = float(np.clip((missing - 0.04) / 0.19, 0, 1))
    return 0.02 + max(0.63 * noise_level, 0.88 * hole_level), 1.0 - noise_level


def _resolve_valid(depth, valid_mask):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        if mask.shape != depth.shape:
            raise ValueError(f"valid_mask shape {mask.shape} != depth shape {depth.shape}")
        valid &= mask
    return valid


def _normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mx = float(x.max())
    return x if mx <= 1e-12 else (x / mx).astype(np.float32)


def _measurement_advantage(classical, deep, lr_depth, valid_mask):
    """Return where classical better matches the available LR measurements."""
    h, w = lr_depth.shape
    classical_safe = np.where(np.isfinite(classical), classical, 0.0)
    deep_safe = np.where(np.isfinite(deep), deep, 0.0)
    classical_lr = cv2.resize(classical_safe, (w, h), interpolation=cv2.INTER_AREA)
    deep_lr = cv2.resize(deep_safe, (w, h), interpolation=cv2.INTER_AREA)
    valid = _resolve_valid(lr_depth, valid_mask)
    err_c = np.abs(classical_lr - lr_depth)
    err_d = np.abs(deep_lr - lr_depth)
    advantage = (err_d - err_c) / (err_d + err_c + 1e-6)
    advantage = np.where(np.isfinite(advantage), advantage, 0.0)
    return np.where(valid, np.clip(advantage, 0.0, 1.0), 0.0).astype(np.float32)


def estimate_confidence(
    classical_depth,
    deep_depth,
    lr_depth,
    valid_mask=None,
    rgb=None,
    scale=4,
) -> tuple:
    """Return ``(W_c, W_d)`` float32 arrays in [0, 1] with ``W_c + W_d == 1``."""
    Dc = np.asarray(classical_depth, dtype=np.float32)
    Dd = np.asarray(deep_depth, dtype=np.float32)
    if Dc.shape != Dd.shape:
        raise ValueError(f"classical {Dc.shape} and deep {Dd.shape} shapes differ")
    H, W = Dc.shape[:2]

    # 1. structure: normalized Sobel magnitude of the classical reconstruction.
    gx = cv2.Sobel(Dc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Dc, cv2.CV_32F, 0, 1, ksize=3)
    edge_score = _normalize01(cv2.magnitude(gx, gy))

    # 2. Compare branch residuals at the available LR measurements. This is
    # safer than assigning a large weight merely because a pixel is nearby.
    valid = _resolve_valid(lr_depth, valid_mask)
    if valid.any():
        advantage_lr = _measurement_advantage(Dc, Dd, lr_depth, valid_mask)
        advantage = cv2.resize(advantage_lr, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        advantage = np.zeros((H, W), np.float32)

    # 3. RGB/depth edge agreement (0 when rgb is unavailable).
    agreement = np.zeros((H, W), np.float32)
    if rgb is not None:
        try:
            _, agree, _ = edge_evidence(lr_depth, rgb)
            agreement = agree.astype(np.float32)
        except Exception:
            agreement = np.zeros((H, W), np.float32)

    # Keep deep dominant by default. Extra classical weight requires both boundary
    # support and a meaningful measurement advantage. Strongly supported boundaries
    # may use a larger contribution so a blurred deep edge can actually be repaired.
    boundary_support = 0.35 * edge_score + 0.65 * agreement
    advantage_gate = np.clip((advantage - 0.15) / 0.85, 0.0, 1.0)
    wc = 0.01 + 0.34 * boundary_support * advantage_gate
    wc = np.clip(wc, 0.0, 0.35).astype(np.float32)

    # A branch with invalid/absent output must never be trusted there.
    invalid_classical = ~np.isfinite(Dc) | (Dc <= 0)
    invalid_deep = ~np.isfinite(Dd) | (Dd <= 0)
    wc[invalid_classical] = 0.0
    wc[invalid_deep] = 1.0
    wd = (1.0 - wc).astype(np.float32)

    # Enforce the normalization invariant exactly.
    total = wc + wd
    safe = total > 1e-8
    wc = np.where(safe, wc / np.maximum(total, 1e-8), 0.5).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)
    return wc, wd
