"""End-to-end hybrid depth reconstruction pipeline.

    hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32

The hybrid combines the classical edge-preserving reconstruction with the deep
branch using weights based on input quality, then applies boundary-aware refinement and,
where the forward degradation operator is known, measurement-consistency correction.

If the deep branch is unavailable (no checkpoint, unsupported scale, missing torch,
or CUDA-less machine), the pipeline degrades gracefully to the classical branch and
reports that fact rather than failing silently.
"""

from __future__ import annotations

import numpy as np

from classical import classical_hr_depth
from classical.jbu import to_float_rgb

from .boundary_refinement import refine_hybrid_depth
from .confidence import estimate_confidence, quality_blend
from .consistency import enforce_consistency
from .fusion import fuse, fuse_fixed


def _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale):
    """Lazy import: use routed path when quality routing is available, else plain deep.

    Returns (depth, route_info).  route_info is None when routing is unavailable.
    Passes the ORIGINAL LR validity mask (before any hole filling).
    """
    try:
        from deep.routing import deep_hr_depth_routed
        return deep_hr_depth_routed(lr_depth, hr_rgb, valid_mask, scale)
    except Exception:  # noqa: BLE001 -- routing env-vars not set; fall back to plain deep
        from deep.inference import deep_hr_depth as _deep
        return _deep(lr_depth, hr_rgb, valid_mask, scale), None


def _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn):
    """Return the deep prediction, routing metadata and any fallback reason."""
    if deep_fn is not None:
        return deep_fn(lr_depth, hr_rgb, valid_mask, scale), None, None
    try:
        depth, route_info = _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
        return depth, route_info, None
    except Exception as exc:  # noqa: BLE001 -- deep branch is optional
        return None, None, f"{type(exc).__name__}: {exc}"


def hybrid_hr_depth(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale=4,
    *,
    use_consistency=True,
    classical_weight=None,
    use_refinement=False,
    selective_refinement=False,
    refinement_strength=1.0,
    consistency_mode="bicubic",
    consistency_strength=1.0,
    consistency_iterations=1,
    deep_fn=None,
    return_intermediates=False,
):
    """Reconstruct HR depth with the full hybrid pipeline.

    ``deep_fn`` is an injectable replacement for the deep branch (used by tests and
    by users who want to supply their own model). It must match the signature
    ``deep_fn(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32``.
    ``classical_weight`` selects a fixed classical contribution for ablations;
    when omitted, the weight adapts to noise and missing LR measurements.
    Boundary refinement is opt-in because validation shows that full-image
    refinement can blur otherwise accurate deep predictions. Refinement and
    consistency can be enabled independently with ``use_refinement`` and
    ``use_consistency``.

    Returns the classical result directly when the deep branch is unavailable
    (no checkpoint, unsupported scale, or missing torch). If the deep branch
    returns invalid pixels, the confidence/fusion stages handle those pixels.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))

    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale, mode=consistency_mode)
    Dd, route_info, deep_error = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn)
    deep_available = deep_error is None

    if deep_available:
        weight = classical_weight
        if weight is None:
            weight, damping = quality_blend(lr_depth, valid_mask)
            consistency_strength *= damping
        if not 0.0 <= weight <= 1.0:
            raise ValueError("classical_weight must be between 0 and 1")
        Wc = np.full_like(Dc, weight, dtype=np.float32)
        Wd = np.full_like(Dd, 1.0 - weight, dtype=np.float32)
        Df = fuse_fixed(Dc, Dd, weight)
        use_consistency = use_consistency and consistency_strength > 0
    else:
        # The standalone solver already fits the measurements.
        Wc = np.ones_like(Dc)
        Wd = np.zeros_like(Dc)
        Df = Dc
        use_refinement = False
        use_consistency = False

    Df_boundary = (
        refine_hybrid_depth(
            Df,
            rgb_f,
            selective=selective_refinement,
            strength=refinement_strength,
        )
        if use_refinement
        else Df
    )

    if use_consistency:
        Df = Df_boundary
        for _ in range(max(1, consistency_iterations)):
            Df = enforce_consistency(Df, lr_depth, valid_mask, scale, mode=consistency_mode,
                                     strength=consistency_strength)
    else:
        Df = Df_boundary

    if return_intermediates:
        return Df, {
            "classical": Dc,
            "deep": Dd if deep_available else None,
            "deep_available": deep_available,
            "deep_error": deep_error,
            "route_info": route_info,
            "confidence_classical": Wc,
            "confidence_deep": Wd,
            "fused": Df_boundary,
            "classical_weight": classical_weight,
            "refinement_enabled": use_refinement,
            "consistency_enabled": use_consistency,
            "selective_refinement": selective_refinement,
            "refinement_strength": refinement_strength,
            "consistency_mode": consistency_mode,
            "consistency_strength": consistency_strength,
            "consistency_iterations": consistency_iterations,
        }
    return Df


def ablations(lr_depth, hr_rgb, valid_mask=None, scale=4, deep_fn=None):
    """Return every hybrid ablation variant for comparison.

    Keys: classical, deep, fixed_fusion, confidence_fusion, confidence_boundary,
    final (confidence + boundary + measurement consistency).
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd, _, deep_error = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn)
    if deep_error is not None:
        raise NotImplementedError(deep_error)

    Wc, Wd = estimate_confidence(Dc, Dd, lr_depth, valid_mask, rgb_f, scale)
    confidence_fusion = fuse(Dc, Dd, Wc, Wd)
    confidence_boundary = refine_hybrid_depth(confidence_fusion, rgb_f)

    return {
        "classical": Dc,
        "deep": Dd,
        "deep_available": True,
        "fixed_fusion": fuse_fixed(Dc, Dd, 0.5),
        "confidence_fusion": confidence_fusion,
        "confidence_boundary": confidence_boundary,
        "final": enforce_consistency(confidence_boundary, lr_depth, valid_mask, scale),
    }
