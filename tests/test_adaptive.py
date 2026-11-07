"""Hermetic regression tests for deep/adaptive.py and hybrid v2 fusion (plan.txt §10).

Runs on CPU, needs NO checkpoint and NO dataset (synthetic fixtures only)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------- regime detection
def test_detect_regime_clean():
    from deep.adaptive import detect_regime

    lr = np.linspace(1.0, 3.0, 64, dtype=np.float32).reshape(8, 8)
    assert detect_regime(lr, np.ones_like(lr, bool)) == "clean"


def test_detect_regime_holes():
    from deep.adaptive import detect_regime

    lr = np.full((8, 8), 2.0, np.float32)
    valid = np.ones_like(lr, bool)
    valid[2:4, 2:4] = False
    lr[~valid] = 0.0
    assert detect_regime(lr, valid) == "holes"


def test_detect_regime_noise_separation():
    """Val-calibrated threshold: smooth LR is clean, noisy LR is noise."""
    from deep.adaptive import NOISE_THRESHOLD, detect_regime, laplacian_score

    rng = np.random.default_rng(0)
    smooth = np.cumsum(np.full((32, 32), 0.02, np.float32), axis=1) + 1.0
    assert laplacian_score(smooth, np.ones_like(smooth, bool)) < NOISE_THRESHOLD
    noisy = smooth + rng.normal(0, 0.02, smooth.shape).astype(np.float32)
    assert detect_regime(noisy, np.ones_like(noisy, bool)) == "noise"


def test_default_robust_disabled_under_checkpoint_isolation(monkeypatch):
    """Hermeticity: SGNET_CKPT_DIR isolation must disable the ambient robust default."""
    import deep.adaptive as adaptive

    monkeypatch.setenv("SGNET_CKPT_DIR", "some/isolated/dir")
    assert adaptive._default_robust() is None
    monkeypatch.delenv("SGNET_CKPT_DIR")
    monkeypatch.setenv("SGNET_CKPT_X4", "x.pth")
    assert adaptive._default_robust() is None


# --------------------------------------------------------------------- hybrid v2 fusion
@pytest.fixture
def synthetic_pair():
    H = W = 16
    lr = np.full((H // 2, W // 2), 2.0, np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    return lr, rgb


def test_v2_weights_clean_is_deep_only(synthetic_pair):
    from hybrid.pipeline import _v2_weights

    lr, _ = synthetic_pair
    Dc = np.full((16, 16), 2.5, np.float32)
    Dd = np.full((16, 16), 2.0, np.float32)
    Wc, Wd, regime = _v2_weights(Dc, Dd, lr, None)
    assert regime == "clean"
    np.testing.assert_allclose(Wc, 0.0)
    np.testing.assert_allclose(Wd, 1.0)


def test_v2_weights_holes_uses_calibrated_weight(synthetic_pair):
    from hybrid.pipeline import HOLES_CLASSICAL_WEIGHT, _v2_weights

    lr, _ = synthetic_pair
    valid = np.ones_like(lr, bool)
    valid[4:6, 4:6] = False
    lr = lr.copy()
    lr[~valid] = 0.0
    Dc = np.full((16, 16), 2.5, np.float32)
    Dd = np.full((16, 16), 2.0, np.float32)
    Wc, Wd, regime = _v2_weights(Dc, Dd, lr, valid)
    assert regime == "holes"
    np.testing.assert_allclose(Wc, HOLES_CLASSICAL_WEIGHT)
    np.testing.assert_allclose(Wc + Wd, 1.0)


def test_v2_invalid_deep_forces_classical(synthetic_pair):
    from hybrid.pipeline import _v2_weights

    lr, _ = synthetic_pair
    Dc = np.full((16, 16), 2.5, np.float32)
    Dd = np.full((16, 16), 2.0, np.float32)
    Dd[0, 0] = np.nan
    Dd[0, 1] = -1.0
    Wc, Wd, _ = _v2_weights(Dc, Dd, lr, None)
    assert Wc[0, 0] == 1.0 and Wc[0, 1] == 1.0
    assert Wd[0, 0] == 0.0 and Wd[0, 1] == 0.0


def test_hybrid_v2_with_injected_deep(synthetic_pair):
    from hybrid import hybrid_hr_depth

    lr, rgb = synthetic_pair

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full((16, 16), 2.1, np.float32)

    out, inter = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn, return_intermediates=True)
    assert inter["fusion"] == "v2"
    assert inter["regime"] == "clean"
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()


def test_hybrid_legacy_fusion_still_available(synthetic_pair):
    from hybrid import hybrid_hr_depth

    lr, rgb = synthetic_pair

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full((16, 16), 2.1, np.float32)

    out, inter = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn,
                                 return_intermediates=True, fusion="legacy")
    assert inter["fusion"] == "legacy"
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()


def test_hybrid_v2_deterministic(synthetic_pair):
    from hybrid import hybrid_hr_depth

    lr, rgb = synthetic_pair

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full((16, 16), 2.1, np.float32)

    a = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn)
    b = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------- skip classical
def test_v2_skips_classical_outside_holes(synthetic_pair, monkeypatch):
    """Latency: in regimes where the v2 classical weight is 0, the classical branch
    must not run at all."""
    import hybrid.pipeline
    from hybrid import hybrid_hr_depth

    lr, rgb = synthetic_pair

    def boom(*args, **kwargs):
        raise AssertionError("classical branch must be skipped in clean regimes")

    monkeypatch.setattr(hybrid.pipeline, "classical_blend_hr_depth", boom)

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full((16, 16), 2.1, np.float32)

    out = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()


def test_v2_still_calls_classical_in_holes(synthetic_pair, monkeypatch):
    """The holes regime has a calibrated non-zero classical weight: it must run."""
    import hybrid.pipeline
    from hybrid import hybrid_hr_depth

    lr = np.full((8, 8), 2.0, np.float32)
    valid = np.ones_like(lr, bool)
    valid[2:4, 2:4] = False
    lr = lr.copy()
    lr[~valid] = 0.0
    rgb = np.zeros((16, 16, 3), np.float32)

    called = []
    monkeypatch.setattr(
        hybrid.pipeline, "classical_blend_hr_depth",
        lambda *a, **k: called.append(1) or np.full((16, 16), 2.5, np.float32),
    )

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full((16, 16), 2.1, np.float32)

    out = hybrid_hr_depth(lr, rgb, valid, 2, deep_fn=deep_fn)
    assert called
    assert np.isfinite(out).all()


def test_bicubic_fallback_fills_invalid_deep(synthetic_pair, monkeypatch):
    """With classical skipped, invalid deep pixels fall back to a bicubic upsample."""
    import hybrid.pipeline
    from hybrid import hybrid_hr_depth

    lr, rgb = synthetic_pair

    def boom(*args, **kwargs):
        raise AssertionError("classical branch must be skipped in clean regimes")

    monkeypatch.setattr(hybrid.pipeline, "classical_blend_hr_depth", boom)

    def deep_fn(lr_depth, hr_rgb, valid_mask, scale):
        out = np.full((16, 16), 2.1, np.float32)
        out[0, 0] = -1.0  # invalid deep pixel
        return out

    out = hybrid_hr_depth(lr, rgb, None, 2, deep_fn=deep_fn)
    assert np.isfinite(out).all()
    assert out[0, 0] > 0  # filled from the LR, not the invalid value


# --------------------------------------------------------------------- classical blend
def test_classical_blend_constant_depth():
    from classical.classical_hr_depth import reconstruct

    depth = np.full((6, 8), 2.5, np.float32)
    rgb = np.zeros((24, 32, 3), np.float32)
    out = reconstruct(depth, rgb, None, 4, method="classical_blend")
    assert out.dtype == np.float32
    assert out.shape == (24, 32)
    np.testing.assert_allclose(out, 2.5, atol=2e-5)


def test_classical_blend_gain_endpoints():
    """gain=0 is the smooth back-projection; gain=1 is the dense path."""
    from classical.classical_hr_depth import reconstruct
    from classical.edge_upsampling import upsample_dense_depth, _backproject_bicubic

    depth = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    rgb = np.zeros((16, 16, 3), np.float32)
    g0 = reconstruct(depth, rgb, None, 4, method="classical_blend")
    # default gain blend must sit between the two endpoints
    smooth = _backproject_bicubic(depth, 4)
    dense = upsample_dense_depth(depth, rgb, 4)
    blend = 0.6 * smooth + 0.4 * dense
    np.testing.assert_allclose(g0, blend, atol=1e-5)
