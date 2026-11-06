"""Unit tests for the hybrid confidence, fusion, consistency, and pipeline."""

import numpy as np
import pytest

from hybrid import hybrid_hr_depth
from hybrid.confidence import estimate_confidence
from hybrid.consistency import enforce_consistency
from hybrid.fusion import fuse, fuse_fixed


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


def test_fuse_fixed_weight():
    Dc = np.full((2, 2), 2.0, np.float32)
    Dd = np.full((2, 2), 6.0, np.float32)
    np.testing.assert_allclose(fuse_fixed(Dc, Dd, 0.25), 0.25 * 2 + 0.75 * 6)


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


def test_hybrid_raises_when_deep_unavailable(no_deep_checkpoint):
    lr, rgb = _synthetic(4)
    with pytest.raises(NotImplementedError, match="deep branch not available"):
        hybrid_hr_depth(lr, rgb, None, 4)


def test_hybrid_injectable_deep_fn():
    lr, rgb = _synthetic(4)
    out, inter = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0), return_intermediates=True)
    assert inter["deep_available"] is True
    np.testing.assert_allclose(inter["deep"], 3.0)
    assert out.shape == (16, 16)


def test_hybrid_handles_invalid_deep_output():
    lr, rgb = _synthetic(4)
    out, inter = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(np.nan), return_intermediates=True)
    assert np.isfinite(out).all()


def test_hybrid_deterministic():
    lr, rgb = _synthetic(4)
    a = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0))
    b = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep(3.0))
    np.testing.assert_array_equal(a, b)


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


def test_consistency_requires_exact_operator():
    lr, rgb = _synthetic(4)
    with pytest.raises(ValueError, match="exact known degradation operator"):
        hybrid_hr_depth(
            lr,
            rgb,
            None,
            4,
            classical_weight=0.02,
            operator_exact=False,
            deep_fn=_fake_deep(),
        )
