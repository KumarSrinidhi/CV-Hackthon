"""Adaptive and fixed-weight branch fusion.

The adaptive fusion is ``D_f = W_c * D_c + W_d * D_d`` with ``W_c + W_d == 1``.
Invalid deep predictions are replaced by the classical value so a bad branch never
corrupts the other.
"""

from __future__ import annotations

import numpy as np


def fuse(classical_depth, deep_depth, classical_weight, deep_weight):
    """Adaptive confidence-weighted fusion. Returns ``(H, W) float32``."""
    Dc = np.asarray(classical_depth, dtype=np.float32)
    Dd = np.asarray(deep_depth, dtype=np.float32)
    Wc = np.asarray(classical_weight, dtype=np.float32)
    Wd = np.asarray(deep_weight, dtype=np.float32)
    if Dc.shape != Dd.shape or Dc.shape != Wc.shape or Dc.shape != Wd.shape:
        raise ValueError(
            f"Shape mismatch: Dc {Dc.shape}, Dd {Dd.shape}, Wc {Wc.shape}, Wd {Wd.shape}"
        )

    # Safety: an invalid branch can never contaminate the fusion. If both are
    # invalid, preserve the existing numeric contract with a zero output.
    valid_c = np.isfinite(Dc) & (Dc > 0)
    valid_d = np.isfinite(Dd) & (Dd > 0)
    Dd = np.where(valid_d, Dd, Dc)
    Dc = np.where(valid_c, Dc, Dd)
    Dc = np.where(np.isfinite(Dc), Dc, 0.0)
    Dd = np.where(np.isfinite(Dd), Dd, 0.0)
    total = Wc + Wd
    total = np.where(total > 1e-8, total, 1.0)
    fused = (Wc * Dc + Wd * Dd) / total
    return fused.astype(np.float32)


def fuse_fixed(classical_depth, deep_depth, alpha=0.5):
    """Fixed-weight ablation baseline: ``alpha * D_c + (1 - alpha) * D_d``."""
    Dc = np.asarray(classical_depth, dtype=np.float32)
    Dd = np.asarray(deep_depth, dtype=np.float32)
    if Dc.shape != Dd.shape:
        raise ValueError(f"classical {Dc.shape} and deep {Dd.shape} shapes differ")
    valid_c = np.isfinite(Dc) & (Dc > 0)
    valid_d = np.isfinite(Dd) & (Dd > 0)
    Dd = np.where(valid_d, Dd, Dc)
    Dc = np.where(valid_c, Dc, Dd)
    Dc = np.where(np.isfinite(Dc), Dc, 0.0)
    Dd = np.where(np.isfinite(Dd), Dd, 0.0)
    return (alpha * Dc + (1.0 - alpha) * Dd).astype(np.float32)
