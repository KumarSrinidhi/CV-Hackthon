import numpy as np
import pytest

from classical import classical_hr_depth
from classical.boundary_refinement import depth_boundaries, refine_boundaries
from classical.guided_filter import guided_filter
from classical.hole_filling import fill_small_holes, prepare_depth
from classical.interpolation import interpolate_depth
from classical.pipeline import reconstruct


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical"])
def test_constant_depth_and_texture(scale, method):
    depth = np.full((5, 7), 2.5, np.float32)
    rgb = np.random.default_rng(42).integers(0, 256, (5 * scale, 7 * scale, 3), dtype=np.uint8)
    if method in {"nearest", "bilinear", "bicubic"}:
        result = interpolate_depth(depth, scale=scale, method=method)
    else:
        result = reconstruct(depth, rgb, None, scale, method=method)
    assert result.dtype == np.float32
    assert result.shape == rgb.shape[:2]
    np.testing.assert_allclose(result, 2.5, atol=2e-5)


def test_hole_filling_respects_depth_boundaries_and_original_inputs():
    depth = np.full((7, 9), 2.0, np.float32)
    depth[:, 5:] = 5
    depth[2, 2] = np.nan
    depth[4, 4] = 99
    mask = np.ones_like(depth, dtype=bool)
    mask[4, 4] = False
    original, original_mask = depth.copy(), mask.copy()
    clean, valid = prepare_depth(depth, mask)
    filled, support = fill_small_holes(clean, valid)
    assert support[2, 2] and filled[2, 2] == 2
    assert not support[4, 4] and filled[4, 4] == 0
    classical_hr_depth(depth, np.zeros((14, 18, 3), np.uint8), mask, 2)
    np.testing.assert_array_equal(depth, original)
    np.testing.assert_array_equal(mask, original_mask)


def test_empty_and_unresolved_support():
    depth = np.zeros((11, 11), np.float32)
    rgb = np.zeros((22, 22, 3), np.float32)
    assert not classical_hr_depth(depth, rgb, None, 2).any()
    depth[1, 1] = 4
    out = classical_hr_depth(depth, rgb, None, 2)
    assert out.max() == pytest.approx(4)
    # With NN hole-filling the single valid pixel propagates everywhere;
    # the corner is no longer zero (that was the bug).
    assert out[-1, -1] > 0


def test_boundary_refinement_preserves_step_and_rejects_rgb_only_edges():
    depth = np.tile(np.array([2, 2, 2, 5, 5, 5], np.float32), (5, 1))
    rgb = np.zeros((*depth.shape, 3), np.float32)
    rgb[:, 1:2] = 1
    edges = depth_boundaries(depth)
    assert edges[:, 2:4].all()
    assert edges.sum() == 10
    np.testing.assert_allclose(refine_boundaries(depth, depth, rgb), depth, atol=1e-6)


def test_guided_filter_preserves_unsupported_pixels():
    depth = np.full((7, 7), 3, np.float32)
    depth[3, 3] = 0
    np.testing.assert_allclose(guided_filter(depth, depth / 3), depth, atol=1e-6)


def test_rgb_can_relocate_a_depth_supported_edge():
    depth = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    rgb = np.zeros((16, 16, 3), np.float32)
    rgb[:, 7:] = 1
    guided = classical_hr_depth(depth, rgb, None, 4)
    # Full A1→A4 pipeline: left half should be near 2, right half near 5
    assert guided[8, 3] == pytest.approx(2, abs=0.5)
    assert guided[8, 12] == pytest.approx(5, abs=0.5)


@pytest.mark.parametrize("scale", [0, -2, True, 2.5])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match="Scale"):
        classical_hr_depth([[1]], np.zeros((2, 2, 3), np.uint8), None, scale)


def test_invalid_mask_and_rgb():
    depth = np.ones((2, 2))
    rgb = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError, match="mask"):
        classical_hr_depth(depth, rgb, [[1]], 2)
    with pytest.raises(ValueError, match="shape"):
        classical_hr_depth(depth, rgb[:, :-1], None, 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classical_hr_depth(depth, rgb.astype(float) + 1.1, None, 2)


def test_public_wrapper_runs_full_pipeline():
    depth = np.array([[2, 3], [5, 6]], np.float32)
    rgb = np.random.default_rng(7).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = classical_hr_depth(depth, rgb, None)
    # Full A1→A4 pipeline produces a valid (8,8) float32 map
    assert out.shape == (8, 8) and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_jbu_excludes_masked_and_nonfinite_samples_before_accumulation():
    from classical.jbu import joint_bilateral_upsample

    depth = np.array([[2, np.nan, 99], [np.inf, 0, 2]], np.float32)
    valid = np.ones_like(depth, dtype=bool)
    valid[0, 2] = False
    rgb = np.zeros((4, 6, 3), np.float32)
    result = joint_bilateral_upsample(depth, valid, rgb)
    np.testing.assert_allclose(result, 2, atol=1e-6)


def test_rgb_only_edges_are_excluded_from_agreement():
    from classical.adaptive_jbu import edge_evidence

    depth = np.full((4, 8), 2, np.float32)
    rgb = np.zeros((16, 32, 3), np.float32)
    rgb[:, 16:] = 1
    band, agreement, _ = edge_evidence(depth, rgb)
    assert not band.any()
    assert not agreement.any()


def test_adaptive_jbu_preserves_measured_thin_depth_without_rgb_edge():
    depth = np.full((6, 9), 6, np.float32)
    depth[:, 4] = 2
    rgb = np.zeros((24, 36, 3), np.uint8)
    for method in ("adaptive_jbu", "refined_jbu"):
        result = reconstruct(depth, rgb, None, 4, method=method)
        np.testing.assert_allclose(result[:, 16:20], 2, atol=1e-6)
        np.testing.assert_allclose(result[:, 20:24], 6, atol=1e-6)


def test_agreement_can_use_a_nearby_rgb_edge():
    from classical.adaptive_jbu import edge_evidence

    depth = np.ones((4, 8), np.float32)
    depth[:, 4:] = 3
    rgb = np.zeros((16, 32, 3), np.float32)
    rgb[:, 15:] = 1
    band, agreement, _ = edge_evidence(depth, rgb)
    assert agreement[:, 15].all()
    assert not (agreement & ~band).any()
