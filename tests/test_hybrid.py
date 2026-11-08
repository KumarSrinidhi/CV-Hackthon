"""Unit tests for the hybrid confidence, fusion, consistency, and pipeline."""

import numpy as np
import pytest

from hybrid import hybrid_hr_depth
from hybrid.confidence import estimate_confidence
from hybrid.consistency import enforce_consistency
from hybrid.fusion import fuse, fuse_fixed
from data.degradation import downsample, forward_operator
from eval.evaluate import load_method


@pytest.fixture()
def no_deep_checkpoint(tmp_path, monkeypatch):
    """Isolate the deep branch: no checkpoints anywhere, so hybrid must degrade."""
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path / "empty_ckpt"))
    monkeypatch.delenv("SGNET_CKPT_X4", raising=False)
    monkeypatch.delenv("SGNET_CKPT_X8", raising=False)
    monkeypatch.delenv("SGNET_CKPT_X16", raising=False)


def _synthetic(scale=4):
    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    rgb = np.zeros((16, 16, 3), np.float32)
    rgb[:, 7:] = 1
    return lr, rgb


def _fake_deep(value=3.0, shape=(16, 16)):
    def fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full(shape, value, np.float32)

    return fn


def test_confidence_normalized_to_one():
    Dc = np.full((4, 4), 2.0, np.float32)
    Dd = np.full((4, 4), 5.0, np.float32)
    Wc, Wd = estimate_confidence(Dc, Dd, np.full((2, 2), 2.0, np.float32), None, np.zeros((4, 4, 3), np.float32), 2)
    np.testing.assert_allclose(Wc + Wd, 1.0, atol=1e-6)
    assert Wc.min() >= 0.0 and Wc.max() <= 1.0


def test_confidence_invalid_deep_forces_classical():
    Dc = np.full((4, 4), 2.0, np.float32)
    Dd = np.full((4, 4), np.nan, np.float32)
    Wc, Wd = estimate_confidence(Dc, Dd, np.full((2, 2), 2.0, np.float32), None, np.zeros((4, 4, 3), np.float32), 2)
    np.testing.assert_allclose(Wd, 0.0)
    np.testing.assert_allclose(Wc, 1.0)


def test_fuse_weighted_average():
    Dc = np.full((2, 2), 2.0, np.float32)
    Dd = np.full((2, 2), 4.0, np.float32)
    Wc = np.full((2, 2), 0.75, np.float32)
    Wd = np.full((2, 2), 0.25, np.float32)
    np.testing.assert_allclose(fuse(Dc, Dd, Wc, Wd), 0.75 * 2 + 0.25 * 4)


def test_fuse_invalid_deep_uses_classical():
    Dc = np.full((2, 2), 3.0, np.float32)
    Dd = np.full((2, 2), np.nan, np.float32)
    Wc = np.full((2, 2), 0.5, np.float32)
    Wd = np.full((2, 2), 0.5, np.float32)
    np.testing.assert_allclose(fuse(Dc, Dd, Wc, Wd), 3.0)


def test_fuse_invalid_classical_uses_deep():
    Dc = np.full((2, 2), np.nan, np.float32)
    Dd = np.full((2, 2), 4.0, np.float32)
    weights = np.full((2, 2), 0.5, np.float32)
    np.testing.assert_allclose(fuse(Dc, Dd, weights, weights), 4.0)


def test_fuse_both_invalid_returns_zero():
    Dc = np.zeros((2, 2), np.float32)
    Dd = np.full((2, 2), np.nan, np.float32)
    weights = np.full((2, 2), 0.5, np.float32)
    np.testing.assert_array_equal(fuse(Dc, Dd, weights, weights), 0.0)


def test_fuse_fixed_weight():
    Dc = np.full((2, 2), 2.0, np.float32)
    Dd = np.full((2, 2), 6.0, np.float32)
    np.testing.assert_allclose(fuse_fixed(Dc, Dd, 0.25), 0.25 * 2 + 0.75 * 6)


def test_zero_alpha_is_deep_equivalence():
    Dc = np.full((2, 2), 2.0, np.float32)
    Dd = np.full((2, 2), 6.0, np.float32)
    np.testing.assert_array_equal(fuse_fixed(Dc, Dd, 0.0), Dd)


def test_confidence_can_disable_classical_contribution():
    Dc = np.full((4, 4), 2.0, np.float32)
    Dd = np.full((4, 4), 6.0, np.float32)
    Wc, Wd = estimate_confidence(
        Dc, Dd, np.full((2, 2), 2.0, np.float32), None,
        np.zeros((4, 4, 3), np.float32), 2, adaptive=False,
    )
    np.testing.assert_array_equal(Wc, 0.0)
    np.testing.assert_array_equal(Wd, 1.0)


def test_consistency_constant_is_identity():
    lr = np.full((2, 2), 4.0, np.float32)
    fused = np.full((4, 4), 4.0, np.float32)
    np.testing.assert_allclose(enforce_consistency(fused, lr, None, 2), 4.0, atol=1e-3)


def test_consistency_is_deterministic():
    rng = np.random.default_rng(0)
    fused = rng.random((8, 8)).astype(np.float32) + 1.0
    lr = np.full((2, 2), 2.0, np.float32)
    a = enforce_consistency(fused, lr, None, 4)
    b = enforce_consistency(fused, lr, None, 4)
    np.testing.assert_array_equal(a, b)


def test_consistency_does_not_spread_invalid_hole_as_zero():
    lr = np.full((2, 2), 4.0, np.float32)
    valid = np.array([[True, False], [True, True]])
    lr[~valid] = 0.0
    fused = np.full((4, 4), 4.0, np.float32)
    out = enforce_consistency(fused, lr, valid, 2)
    np.testing.assert_allclose(out, 4.0, atol=1e-5)


def test_consistency_uses_exact_operator_and_is_configurable():
    gt = np.arange(64, dtype=np.float32).reshape(8, 8) / 10 + 1
    lr = downsample(gt, 2)
    fused = gt + 0.4
    op = forward_operator(2, "bicubic")
    uncorrected = enforce_consistency(fused, lr, None, 2, operator=op, strength=0.0, iterations=4)
    corrected = enforce_consistency(fused, lr, None, 2, operator=op, strength=0.25, iterations=2)
    np.testing.assert_array_equal(uncorrected, fused)
    assert not np.array_equal(corrected, fused)
    with pytest.raises(ValueError):
        enforce_consistency(fused, lr, None, 2, operator=op, strength=1.1)
    with pytest.raises(ValueError):
        enforce_consistency(fused, lr, None, 2, operator=op, iterations=-1)


def test_consistency_sweep_method_name_is_supported(monkeypatch):
    # Hermetic: exercises the method-name parsing, not checkpoint availability.
    monkeypatch.setattr(
        "hybrid.pipeline._deep_hr_depth",
        lambda lr, rgb, mask, scale: np.full((16, 16), 2.0, np.float32),
    )
    method = load_method("dl_consistency_s0.25_i2")
    out = method(
        np.full((4, 4), 2.0, np.float32),
        np.zeros((16, 16, 3), np.float32),
        np.ones((4, 4), bool),
        4,
    )
    assert out.shape == (16, 16) and np.isfinite(out).all()


def test_hybrid_raises_when_deep_unavailable(no_deep_checkpoint):
    lr, rgb = _synthetic(4)
    out, inter = hybrid_hr_depth(lr, rgb, None, 4, return_intermediates=True)
    assert out.shape == (16, 16) and np.isfinite(out).all()
    assert inter["deep_available"] is False


def test_hybrid_injectable_deep_fn():
    lr, rgb = _synthetic(4)
    out, inter = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0), return_intermediates=True)
    assert inter["deep_available"] is True
    np.testing.assert_allclose(inter["deep"], 3.0)
    assert out.shape == (16, 16)


def test_hybrid_boundary_refinement_is_optional():
    lr, rgb = _synthetic(4)
    plain = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0), use_boundary_refinement=False)
    refined = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0), use_boundary_refinement=True)
    assert plain.shape == refined.shape == (16, 16)


def test_hybrid_handles_invalid_deep_output():
    lr, rgb = _synthetic(4)
    out, inter = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(np.nan), return_intermediates=True)
    assert np.isfinite(out).all()


def test_hybrid_deterministic():
    lr, rgb = _synthetic(4)
    a = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0))
    b = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0))
    np.testing.assert_array_equal(a, b)


def test_adaptive_confidence_is_conservative_when_deep_matches_measurement():
    lr = np.full((2, 2), 4.0, np.float32)
    classical = np.full((8, 8), 2.0, np.float32)
    deep = np.full((8, 8), 4.0, np.float32)
    rgb = np.zeros((8, 8, 3), np.float32)
    wc, wd = estimate_confidence(classical, deep, lr, None, rgb, 4)
    assert float(wc.max()) <= 0.01 + 1e-6
    np.testing.assert_allclose(wc + wd, 1.0, atol=1e-6)


def test_adaptive_confidence_can_repair_strong_supported_edge():
    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (2, 1))
    classical = np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1)
    deep = np.full((8, 16), 3.5, np.float32)
    rgb = np.zeros((8, 16, 3), np.float32)
    rgb[:, 8:] = 1.0
    wc, _ = estimate_confidence(classical, deep, lr, None, rgb, 4)
    assert float(wc.max()) > 0.10
    assert float(wc.max()) <= 0.35 + 1e-6


def test_fixed_fusion_can_disable_postprocessing():
    lr, rgb = _synthetic(4)
    deep = _fake_deep(3.0)
    out = hybrid_hr_depth(
        lr,
        rgb,
        None,
        4,
        classical_weight=0.0,
        use_refinement=False,
        use_consistency=False,
        deep_fn=deep,
    )
    np.testing.assert_allclose(out, 3.0)


def test_fixed_fusion_rejects_invalid_classical_weight():
    lr, rgb = _synthetic(4)
    with pytest.raises(ValueError, match="classical_weight"):
        hybrid_hr_depth(lr, rgb, None, 4, classical_weight=1.1, deep_fn=_fake_deep())


def test_consistency_strength_zero_is_identity():
    """consistency_strength=0 must leave the fused map unchanged."""
    lr, rgb = _synthetic(4)
    out_no_cons = hybrid_hr_depth(
        lr, rgb, None, 4, classical_weight=0.02, use_consistency=False, deep_fn=_fake_deep(3.0)
    )
    out_zero = hybrid_hr_depth(
        lr, rgb, None, 4, classical_weight=0.02, use_consistency=True,
        consistency_strength=0.0, deep_fn=_fake_deep(3.0)
    )
    np.testing.assert_allclose(out_no_cons, out_zero, atol=1e-5)


def test_confidence_prefers_classical_near_holes():
    # One missing LR measurement: the deep input is nearest-filled there, so the
    # classical branch must gain weight in that band (wherever it is supported).
    lr = np.full((4, 4), 2.0, np.float32)
    lr[1, 1] = 0.0
    Dc = np.full((16, 16), 2.0, np.float32)
    Dd = np.full((16, 16), 5.0, np.float32)
    Wc, Wd = estimate_confidence(
        Dc, Dd, lr, None, np.zeros((16, 16, 3), np.float32), 4
    )
    np.testing.assert_allclose(Wc + Wd, 1.0, atol=1e-6)
    # Hole at LR (1,1) -> HR block [4:8, 4:8]; its 3x3 LR dilation covers
    # HR [0:12, 0:12], so the far corner is the clean comparison region.
    assert Wc[4:8, 4:8].mean() > Wc[12:16, 12:16].mean()


def test_confidence_hole_weight_requires_classical_support():
    lr = np.full((4, 4), 2.0, np.float32)
    lr[1, 1] = 0.0
    Dc = np.zeros((16, 16), np.float32)  # classical has nothing to offer
    Dd = np.full((16, 16), 5.0, np.float32)
    Wc, Wd = estimate_confidence(
        Dc, Dd, lr, None, np.zeros((16, 16, 3), np.float32), 4
    )
    np.testing.assert_allclose(Wc, 0.0)
    np.testing.assert_allclose(Wd, 1.0)


def test_confidence_no_holes_no_extra_classical_weight():
    # Dense measurement grid: the hole band must not fire at all.
    lr = np.full((4, 4), 2.0, np.float32)
    Dc = np.full((16, 16), 2.0, np.float32)
    Dd = np.full((16, 16), 5.0, np.float32)
    Wc, _ = estimate_confidence(
        Dc, Dd, lr, None, np.zeros((16, 16, 3), np.float32), 4
    )
    assert Wc.max() <= 0.1 + 1e-6


def test_hybrid_hole_band_uses_classical_measurements():
    lr = np.full((4, 4), 2.0, np.float32)
    lr[1, 1] = 0.0
    rgb = np.zeros((16, 16, 3), np.float32)
    out, inter = hybrid_hr_depth(
        lr, rgb, None, 4, deep_fn=_fake_deep(9.0), return_intermediates=True,
        use_consistency=False,
    )
    # Deep says 9 everywhere; near the hole the classical value 2 must pull the
    # result below the pure deep value.
    assert out[8, 8] < 9.0
    assert inter["confidence_classical"][8, 8] > 0.0
