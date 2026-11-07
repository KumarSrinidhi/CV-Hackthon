"""End-to-end hybrid depth reconstruction pipeline.

    hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32

v2 fusion (default), calibrated on validation data (results/hybrid_v2_sweep):
the deep branch dominates; the classical branch contributes a small constant
weight only in the holes regime, and the fused map is anchored to the LR
measurements by consistency enforcement. The legacy evidence-weighted fusion
(0.4 edge + 0.3 distance + 0.3 agreement) measurably damaged the deep branch
(docs/PERFORMANCE_ANALYSIS.md, probe B2) and is available via ``fusion="legacy"``.

The deep branch is regime-aware (deep/adaptive.py): LR-only detection selects
the clean-specialist or robust-curriculum checkpoint. If the deep branch is
unavailable (no checkpoint, unsupported scale, missing torch, or CUDA-less
machine), the pipeline raises so evaluation can report it honestly.
"""

from __future__ import annotations

import numpy as np

from classical import classical_hr_depth
from classical.jbu import to_float_rgb

from .boundary_refinement import refine_hybrid_depth
from .confidence import estimate_confidence
from .consistency import enforce_consistency
from .fusion import fuse, fuse_fixed

# Validation-calibrated classical weight in the holes regime (P3-E8b sweep):
# 0.05 beats deep-only under holes; on clean/noise any classical weight hurts.
HOLES_CLASSICAL_WEIGHT = 0.05


def _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale):
    """Lazy import + call of the real deep branch. Raises on unavailability."""
    from deep.inference import deep_hr_depth as _deep

    return _deep(lr_depth, hr_rgb, valid_mask, scale)


def _deep_hr_depth_adaptive(lr_depth, hr_rgb, valid_mask, scale):
    """Regime-aware deep branch (falls back to the original checkpoint behaviour)."""
    from deep.adaptive import deep_hr_depth_adaptive

    return deep_hr_depth_adaptive(lr_depth, hr_rgb, valid_mask, scale)


def _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn, adaptive):
    """Return the deep HR depth, or raise NotImplementedError if it is unavailable.

    A genuine hybrid needs a working deep branch; silently substituting the
    classical output and labelling it "hybrid" would misreport results, so the
    unavailability is surfaced instead. The caller (evaluation framework or CLI)
    decides how to degrade.
    """
    if deep_fn is not None:
        return deep_fn(lr_depth, hr_rgb, valid_mask, scale)
    try:
        fn = _deep_hr_depth_adaptive if adaptive else _deep_hr_depth
        return fn(lr_depth, hr_rgb, valid_mask, scale)
    except Exception as exc:  # noqa: BLE001 -- deep branch is optional
        raise NotImplementedError(
            f"hybrid unavailable: deep branch not available: {type(exc).__name__}: {exc}"
        ) from exc


def _v2_weights(Dc, Dd, lr_depth, valid_mask):
    """Regime-aware, validation-calibrated classical weight map (v2)."""
    from deep.adaptive import detect_regime

    regime = detect_regime(np.asarray(lr_depth, dtype=np.float32), valid_mask)
    wc = np.full(np.shape(Dc), HOLES_CLASSICAL_WEIGHT if regime == "holes" else 0.0, dtype=np.float32)
    invalid_deep = ~np.isfinite(Dd) | (Dd <= 0)
    wc = np.where(invalid_deep, 1.0, wc).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)
    return wc, wd, regime


def hybrid_hr_depth(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale=4,
    *,
    use_consistency=True,
    deep_fn=None,
    return_intermediates=False,
    fusion="v2",
):
    """Reconstruct HR depth with the full hybrid pipeline.

    ``deep_fn`` is an injectable replacement for the deep branch (used by tests and
    by users who want to supply their own model). It must match the signature
    ``deep_fn(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32``.

    ``fusion``: "v2" (default) uses validation-calibrated regime-aware weights with
    the adaptive deep branch; "legacy" reproduces the original evidence-weighted
    fusion with the fixed deep branch.

    Raises ``NotImplementedError`` when the deep branch is unavailable (no
    checkpoint, unsupported scale, or missing torch), so evaluation can report
    "not evaluated" honestly. If the deep branch runs but returns invalid pixels,
    those pixels fall back to the classical branch.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))

    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn, adaptive=(fusion == "v2"))

    if fusion == "v2":
        Wc, Wd, regime = _v2_weights(Dc, Dd, lr_depth, valid_mask)
        Df = fuse(Dc, Dd, Wc, Wd)
        if use_consistency:
            Df = enforce_consistency(Df, lr_depth, valid_mask, scale)
        if return_intermediates:
            return Df, {
                "classical": Dc,
                "deep": Dd,
                "deep_available": True,
                "confidence_classical": Wc,
                "confidence_deep": Wd,
                "fused": Df,
                "regime": regime,
                "fusion": fusion,
            }
        return Df

    Wc, Wd = estimate_confidence(Dc, Dd, lr_depth, valid_mask, rgb_f, scale)
    Df = fuse(Dc, Dd, Wc, Wd)
    Df_boundary = refine_hybrid_depth(Df, rgb_f)

    if use_consistency:
        Df = enforce_consistency(Df_boundary, lr_depth, valid_mask, scale)
    else:
        Df = Df_boundary

    if return_intermediates:
        return Df, {
            "classical": Dc,
            "deep": Dd,
            "deep_available": True,
            "confidence_classical": Wc,
            "confidence_deep": Wd,
            "fused": Df_boundary,
            "fusion": fusion,
        }
    return Df


def ablations(lr_depth, hr_rgb, valid_mask=None, scale=4, deep_fn=None):
    """Return every hybrid ablation variant for comparison.

    Keys: classical, deep, fixed_fusion, confidence_fusion, final (legacy chain),
    v2_fusion, v2_final (calibrated chain), plus the deep branch actually used.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn, adaptive=True)

    Wc, Wd = estimate_confidence(Dc, Dd, lr_depth, valid_mask, rgb_f, scale)
    confidence_fusion = fuse(Dc, Dd, Wc, Wd)

    Wc2, Wd2, regime = _v2_weights(Dc, Dd, lr_depth, valid_mask)
    v2_fusion = fuse(Dc, Dd, Wc2, Wd2)

    return {
        "classical": Dc,
        "deep": Dd,
        "deep_available": True,
        "fixed_fusion": fuse_fixed(Dc, Dd, 0.5),
        "confidence_fusion": confidence_fusion,
        "final": enforce_consistency(confidence_fusion, lr_depth, valid_mask, scale),
        "v2_fusion": v2_fusion,
        "v2_final": enforce_consistency(v2_fusion, lr_depth, valid_mask, scale),
        "regime": regime,
    }
