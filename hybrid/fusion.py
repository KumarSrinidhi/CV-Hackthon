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

    # Safety: a non-finite deep value can never contaminate the fusion.
    Dd = np.where(np.isfinite(Dd), Dd, Dc)
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
    Dd = np.where(np.isfinite(Dd), Dd, Dc)
    return (alpha * Dc + (1.0 - alpha) * Dd).astype(np.float32)
