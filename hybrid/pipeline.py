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
machine), the pipeline degrades to the classical branch and reports
``deep_available=False``; evaluation pre-checks the checkpoint so a missing
deep model is reported as "not evaluated" instead of being scored as hybrid.
"""

from __future__ import annotations

import numpy as np

from classical import classical_hr_depth
from classical.classical_hr_depth import reconstruct as classical_reconstruct
from classical.jbu import to_float_rgb

from .boundary_refinement import refine_hybrid_depth
from .confidence import estimate_confidence
from .consistency import enforce_consistency
from .fusion import fuse, fuse_fixed
from data.degradation import forward_operator

# Validation-calibrated classical weight in the holes regime (P3-E8b sweep):
# 0.05 beats deep-only under holes; on clean/noise any classical weight hurts.
HOLES_CLASSICAL_WEIGHT = 0.05


def classical_blend_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4):
    """Classical branch as used by the v2 hybrid: the validation-calibrated blend
    (results/task_b: RMSE 0.0250 vs bicubic 0.0292 on val150 clean)."""
    return classical_reconstruct(lr_depth, hr_rgb, valid_mask, scale, method="classical_blend")


def _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale):
    """Lazy import + call of the original deep branch. Raises on unavailability."""
    from deep.inference import deep_hr_depth as _deep

    return _deep(lr_depth, hr_rgb, valid_mask, scale)


def _deep_hr_depth_adaptive(lr_depth, hr_rgb, valid_mask, scale):
    """Regime-aware deep branch (falls back to the original checkpoint behaviour)."""
    from deep.adaptive import deep_hr_depth_adaptive

    return deep_hr_depth_adaptive(lr_depth, hr_rgb, valid_mask, scale)


def _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn, adaptive=False):
    """Return (depth, available): available=False when the deep branch is unavailable.

    A missing checkpoint, unsupported scale, or missing torch degrades to the
    classical branch rather than failing the whole pipeline; evaluation
    pre-checks the checkpoint so such runs are reported as "not evaluated".
    """
    if deep_fn is not None:
        return deep_fn(lr_depth, hr_rgb, valid_mask, scale), True
    try:
        fn = _deep_hr_depth_adaptive if adaptive else _deep_hr_depth
        return fn(lr_depth, hr_rgb, valid_mask, scale), True
    except Exception:  # noqa: BLE001 -- deep branch is optional
        return None, False


def _v2_weights(Dc, Dd, lr_depth, valid_mask):
    """Regime-aware, validation-calibrated classical weight map (v2)."""
    from deep.adaptive import detect_regime

    regime = detect_regime(np.asarray(lr_depth, dtype=np.float32), valid_mask)
    wc = np.full(np.shape(Dc), HOLES_CLASSICAL_WEIGHT if regime == "holes" else 0.0, dtype=np.float32)
    invalid_deep = ~np.isfinite(Dd) | (Dd <= 0)
    wc = np.where(invalid_deep, 1.0, wc).astype(np.float32)
    wd = (1.0 - wc).astype(np.float32)
    return wc, wd, regime


def _bicubic_fallback(lr_depth, valid_mask, scale):
    """Cheap HR fallback used where the classical branch is skipped and the deep
    branch returned invalid pixels (bicubic upsample of the nearest-filled LR)."""
    import cv2

    from data.preprocessing import fill_invalid_nearest

    lr = np.asarray(lr_depth, dtype=np.float32)
    valid = np.isfinite(lr) & (lr > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        valid &= mask
    filled = fill_invalid_nearest(np.where(valid, lr, np.float32(0)), valid)
    h, w = lr.shape
    out = cv2.resize(filled, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return np.maximum(out, 0).astype(np.float32)


def hybrid_hr_depth(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale=4,
    *,
    use_consistency=True,
    consistency_strength=0.25,
    consistency_iterations=1,
    consistency_mode="bicubic",
    classical_weight=None,
    classical_alpha=None,
    use_refinement=False,
    use_boundary_refinement=None,
    selective_refinement=False,
    refinement_strength=1.0,
    adaptive=True,
    max_classical_weight=0.35,
    operator=None,
    deep_fn=None,
    return_intermediates=False,
    fusion="v2",
    skip_classical=None,
):
    """Reconstruct HR depth with the full hybrid pipeline.

    ``deep_fn`` is an injectable replacement for the deep branch (used by tests and
    by users who want to supply their own model). It must match the signature
    ``deep_fn(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32``.
    ``classical_weight`` (alias ``classical_alpha``) selects a fixed classical
    contribution for ablations and the frozen config; when omitted, the normal
    conservative spatial confidence fusion is used. Boundary refinement is
    opt-in because validation shows that full-image refinement can blur
    otherwise accurate deep predictions. Refinement and consistency can be
    enabled independently with ``use_refinement`` (alias
    ``use_boundary_refinement``) and ``use_consistency``.

    ``fusion``: "v2" (default) uses validation-calibrated regime-aware weights
    with the adaptive deep branch; "legacy" reproduces the original
    evidence-weighted fusion with the fixed deep branch.

    If the deep branch is unavailable (no checkpoint, unsupported scale, or
    missing torch), the classical branch is returned with
    ``deep_available=False`` in the intermediates; evaluation pre-checks the
    checkpoint and reports "not evaluated" instead. If the deep branch runs but
    returns invalid pixels, those pixels fall back to the classical branch (or a
    cheap bicubic upsample when the classical branch was skipped).

    ``skip_classical`` (v2 only): by default (None) the classical branch is
    skipped in regimes where its calibrated weight is 0 (clean / noise) —
    the v2 fusion gives it zero weight there, so computing it is pure latency.
    Any deep-output hole is then filled from a bicubic upsample of the LR.
    Force with ``skip_classical=False`` to always compute it.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))

    alpha = classical_weight if classical_weight is not None else classical_alpha
    if use_boundary_refinement is not None:
        use_refinement = use_boundary_refinement

    # Adaptive checkpoint selection only applies to the un-bypassed v2 path;
    # fixed-weight / legacy paths use the original deep branch so injected
    # stand-ins and the historical behaviour are preserved.
    Dd, deep_available = _run_deep(
        lr_depth, hr_rgb, valid_mask, scale, deep_fn,
        adaptive=(fusion == "v2" and alpha is None),
    )

    regime = None
    if not deep_available:
        # Degrade: the classical pipeline is the honest output here; scoring
        # guards report missing checkpoints before ever calling this path.
        Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
        Wc = np.ones_like(Dc, dtype=np.float32)
        Wd = np.zeros_like(Dc, dtype=np.float32)
        Df = Dc.copy()
        consistency_on = use_consistency
    elif alpha is not None:
        # Fixed-weight ablation / frozen config (classical_weight or
        # classical_alpha): bypasses v2 for reproducible comparisons.
        if not 0.0 <= alpha <= 1.0:
            which = "classical_weight" if classical_weight is not None else "classical_alpha"
            raise ValueError(f"{which} must be in [0, 1]")
        Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
        Wc = np.full_like(Dc, alpha, dtype=np.float32)
        Wd = np.full_like(Dc, 1.0 - alpha, dtype=np.float32)
        Df = fuse_fixed(Dc, Dd, alpha)
        consistency_on = use_consistency
    elif fusion == "v2":
        from deep.adaptive import detect_regime

        regime0 = detect_regime(np.asarray(lr_depth, dtype=np.float32), valid_mask)
        compute_classical = (regime0 == "holes") if skip_classical is None else not skip_classical
        Dc = (
            classical_blend_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
            if compute_classical
            else _bicubic_fallback(lr_depth, valid_mask, scale)
        )
        Wc, Wd, regime = _v2_weights(Dc, Dd, lr_depth, valid_mask)
        Df = fuse(Dc, Dd, Wc, Wd)
        # Validation-calibrated (results/hybrid_v2_sweep): consistency anchoring
        # helps on clean (-1.6%) and holes (-0.9%) but hurts under noise
        # (+17%: anchoring propagates sensor noise). Skip it there.
        consistency_on = use_consistency and regime != "noise"
    else:
        Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
        Wc, Wd = estimate_confidence(
            Dc, Dd, lr_depth, valid_mask, rgb_f, scale,
            adaptive=adaptive, max_classical_weight=max_classical_weight,
            operator=operator,
        )
        Df = fuse(Dc, Dd, Wc, Wd)
        consistency_on = use_consistency

    Df_boundary = (
        refine_hybrid_depth(
            Df, rgb_f, selective=selective_refinement, strength=refinement_strength
        )
        if use_refinement
        else Df
    )

    if consistency_on:
        Df = enforce_consistency(
            Df_boundary, lr_depth, valid_mask, scale, mode=consistency_mode,
            operator=operator, strength=consistency_strength,
            iterations=consistency_iterations,
        )
    else:
        Df = Df_boundary

    if return_intermediates:
        return Df, {
            "classical": Dc,
            "deep": Dd if deep_available else None,
            "deep_available": deep_available,
            "confidence_classical": Wc,
            "confidence_deep": Wd,
            "fused": Df,
            "fused_before_consistency": Df_boundary,
            "classical_weight": alpha,
            "refinement_enabled": use_refinement,
            "consistency_enabled": use_consistency,
            "selective_refinement": selective_refinement,
            "refinement_strength": refinement_strength,
            "consistency_mode": consistency_mode,
            "consistency_strength": consistency_strength,
            "consistency_iterations": consistency_iterations,
            "fusion": fusion,
            "regime": regime,
        }
    return Df


def ablations(lr_depth, hr_rgb, valid_mask=None, scale=4, deep_fn=None):
    """Return every hybrid ablation variant for comparison.

    Keys: classical, deep, DL+consistency, fixed fusion, fixed fusion+consistency,
    confidence/adaptive fusion (+consistency), v2 fusion, v2 final, plus the
    regime and the boundary-refined final ablation. Requires the deep branch.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd, deep_available = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn, adaptive=True)
    if not deep_available:
        raise RuntimeError("ablations require the deep branch (checkpoint missing)")

    op = forward_operator(scale, "bicubic")
    Wc, Wd = estimate_confidence(Dc, Dd, lr_depth, valid_mask, rgb_f, scale, operator=op)
    confidence_fusion = fuse(Dc, Dd, Wc, Wd)
    confidence_boundary = refine_hybrid_depth(confidence_fusion, rgb_f)

    Wc2, Wd2, regime = _v2_weights(Dc, Dd, lr_depth, valid_mask)
    v2_fusion = fuse(Dc, Dd, Wc2, Wd2)

    return {
        "classical": Dc,
        "deep": Dd,
        "deep_available": True,
        "dl_consistency": enforce_consistency(Dd, lr_depth, valid_mask, scale, operator=op, strength=0.25),
        "fixed_fusion": fuse_fixed(Dc, Dd, 0.02),
        "fixed_fusion_consistency": enforce_consistency(
            fuse_fixed(Dc, Dd, 0.02), lr_depth, valid_mask, scale, operator=op, strength=0.25
        ),
        "confidence_fusion": confidence_fusion,
        "confidence_boundary": confidence_boundary,
        "adaptive_fusion": confidence_fusion,
        "adaptive_fusion_consistency": enforce_consistency(
            confidence_fusion, lr_depth, valid_mask, scale, operator=op, strength=0.25
        ),
        "v2_fusion": v2_fusion,
        "v2_final": enforce_consistency(v2_fusion, lr_depth, valid_mask, scale),
        "regime": regime,
        "final": enforce_consistency(confidence_boundary, lr_depth, valid_mask, scale, operator=op, strength=0.25),
    }
