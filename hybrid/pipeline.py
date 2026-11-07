# Team Owner: Rishita
# Workstream: Hybrid Fusion
# Module: hybrid/pipeline.py
# Description: End-to-end hybrid pipeline combining classical and deep learning with measurement consistency and boundary refinement.

from typing import Optional
import numpy as np
from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.confidence import compute_confidence_weights
from hybrid.fusion import fuse_depth_maps, fixed_weight_fusion
from hybrid.consistency import enforce_measurement_consistency
from hybrid.boundary_refinement import adaptive_boundary_refinement


def hybrid_hr_depth(
    lr_depth: np.ndarray,
    hr_rgb: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    scale: int = 4,
    alpha: float = 0.02,
    method: str = "fixed",
    use_consistency: bool = True,
    use_refinement: bool = False,
    deep_depth: Optional[np.ndarray] = None,
    classical_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Produce a high-resolution depth map by combining deep learning and classical CV branches,
    with measurement consistency enforcement and edge-preserving boundary refinement.

    Args:
        lr_depth: Low-resolution depth map of shape (H_lr, W_lr).
        hr_rgb: High-resolution RGB guidance image of shape (H_hr, W_hr, 3).
        valid_mask: Optional boolean mask of valid depth measurements.
        scale: Upsampling factor (e.g. 2, 4, 8).
        alpha: Weight assigned to classical branch in fixed-weight mode (default 0.02).
        method: Fusion strategy - 'fixed' (ablation baseline) or 'confidence' (adaptive).
        use_consistency: Whether to enforce measurement consistency r_LR = D_LR - S(D_f).
        use_refinement: Whether to apply adaptive boundary refinement.
        deep_depth: Optional precomputed deep branch HR depth (for testing/caching).
        classical_depth: Optional precomputed classical branch HR depth (for testing/caching).

    Returns:
        High-resolution hybrid depth map of shape (H_hr, W_hr) as float32.
    """
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    hr_rgb = np.asarray(hr_rgb)
    h_hr, w_hr = hr_rgb.shape[:2]

    # 1. Obtain branch reconstructions
    if classical_depth is None:
        d_c = classical_hr_depth(lr_depth, hr_rgb, valid_mask=valid_mask, scale=scale, method="jbu")
    else:
        d_c = np.asarray(classical_depth, dtype=np.float32)

    if deep_depth is None:
        d_d = deep_hr_depth(lr_depth, hr_rgb, valid_mask=valid_mask, scale=scale)
    else:
        d_d = np.asarray(deep_depth, dtype=np.float32)

    # 2. Branch Fusion
    if method == "confidence":
        w_c, w_d = compute_confidence_weights(d_c, d_d, hr_rgb=hr_rgb, valid_mask=valid_mask, base_alpha=alpha)
        d_fused = fuse_depth_maps(d_c, d_d, w_c, w_d)
    else:
        # Fixed-weight fusion (optimal alpha = 0.02)
        d_fused = fixed_weight_fusion(d_c, d_d, alpha=alpha)

    # 3. Measurement Consistency Enforcement
    if use_consistency:
        d_fused = enforce_measurement_consistency(
            d_fused, lr_depth, scale=scale, valid_mask=valid_mask, gain=0.3
        )

    # 4. Adaptive Boundary Refinement
    if use_refinement:
        d_fused = adaptive_boundary_refinement(d_fused, hr_rgb, valid_mask=valid_mask)

    # 5. Sanity check
    if not np.all(np.isfinite(d_fused)):
        d_fused = np.where(np.isfinite(d_fused), d_fused, d_d)

    return d_fused.astype(np.float32)
