"""End-to-end hybrid depth reconstruction pipeline.

    hybrid_hr_depth(lr_depth, hr_rgb, valid_mask=None, scale=4) -> (H, W) float32

The hybrid combines the classical edge-preserving reconstruction with the deep
branch using spatial confidence maps, then applies boundary-aware refinement and,
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
from .confidence import estimate_confidence
from .consistency import enforce_consistency
from .fusion import fuse, fuse_fixed
from data.degradation import forward_operator


def _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale):
    """Lazy import + call of the real deep branch. Raises on unavailability."""
    from deep.inference import deep_hr_depth as _deep

    return _deep(lr_depth, hr_rgb, valid_mask, scale)


def _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn):
    """Return the deep HR depth, or raise NotImplementedError if it is unavailable.

    A genuine hybrid needs a working deep branch; silently substituting the
    classical output and labelling it "hybrid" would misreport results, so the
    unavailability is surfaced instead. The caller (evaluation framework or CLI)
    decides how to degrade.
    """
    if deep_fn is not None:
        return deep_fn(lr_depth, hr_rgb, valid_mask, scale)
    try:
        return _deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    except Exception as exc:  # noqa: BLE001 -- deep branch is optional
        raise NotImplementedError(
            f"hybrid unavailable: deep branch not available: {type(exc).__name__}: {exc}"
        ) from exc


def hybrid_hr_depth(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale=4,
    *,
    use_consistency=True,
    consistency_strength=0.25,
    consistency_iterations=1,
    use_boundary_refinement=False,
    adaptive=True,
    classical_alpha=None,
    max_classical_weight=0.1,
    operator=None,
    deep_fn=None,
    return_intermediates=False,
):
    """Reconstruct HR depth with the full hybrid pipeline.

    ``deep_fn`` is an injectable replacement for the deep branch (used by tests and
    by users who want to supply their own model). It must match the signature
    ``deep_fn(lr_depth, hr_rgb, valid_mask, scale) -> (H, W) float32``.

    Raises ``NotImplementedError`` when the deep branch is unavailable (no
    checkpoint, unsupported scale, or missing torch), so evaluation can report
    "not evaluated" honestly. If the deep branch runs but returns invalid pixels,
    those pixels are handled by the confidence/fusion stages.
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))

    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn)

    if classical_alpha is not None:
        if not 0.0 <= classical_alpha <= 1.0:
            raise ValueError("classical_alpha must be in [0, 1]")
        Df = fuse_fixed(Dc, Dd, classical_alpha)
        Wc = np.full_like(Dc, classical_alpha, dtype=np.float32)
        Wd = np.full_like(Dc, 1.0 - classical_alpha, dtype=np.float32)
    else:
        Wc, Wd = estimate_confidence(
            Dc, Dd, lr_depth, valid_mask, rgb_f, scale,
            adaptive=adaptive, max_classical_weight=max_classical_weight,
            operator=operator or forward_operator(scale, "bicubic"),
        )
        Df = fuse(Dc, Dd, Wc, Wd)
    Df_boundary = refine_hybrid_depth(Df, rgb_f) if use_boundary_refinement else Df

    if use_consistency:
        Df = enforce_consistency(
            Df_boundary, lr_depth, valid_mask, scale,
            operator=operator, strength=consistency_strength,
            iterations=consistency_iterations,
        )
    else:
        Df = Df_boundary

    if return_intermediates:
        return Df, {
            "classical": Dc,
            "deep": Dd,
            "deep_available": True,
            "confidence_classical": Wc,
            "confidence_deep": Wd,
            "fused": Df,
            "fused_before_consistency": Df_boundary,
        }
    return Df


def ablations(lr_depth, hr_rgb, valid_mask=None, scale=4, deep_fn=None):
    """Return every hybrid ablation variant for comparison.

    Keys: classical, deep, fixed_fusion, confidence_fusion, confidence_boundary,
    final (confidence + boundary + measurement consistency).
    """
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    Dc = classical_hr_depth(lr_depth, hr_rgb, valid_mask, scale)
    Dd = _run_deep(lr_depth, hr_rgb, valid_mask, scale, deep_fn)

    op = forward_operator(scale, "bicubic")
    Wc, Wd = estimate_confidence(Dc, Dd, lr_depth, valid_mask, rgb_f, scale, operator=op)
    confidence_fusion = fuse(Dc, Dd, Wc, Wd)
    confidence_boundary = refine_hybrid_depth(confidence_fusion, rgb_f)

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
        "final": enforce_consistency(confidence_boundary, lr_depth, valid_mask, scale, operator=op, strength=0.25),
    }
