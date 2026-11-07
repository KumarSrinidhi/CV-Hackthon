# Team Owner: Rishita
# Workstream: Hybrid Fusion Testing
# Module: tests/test_hybrid.py
# Description: Unit tests for branch confidence estimation, fusion blending, and end-to-end hybrid pipeline.

import numpy as np
import pytest
from numpy.testing import assert_allclose
from hybrid.confidence import compute_confidence_weights
from hybrid.fusion import fuse_depth_maps, fixed_weight_fusion
from hybrid.consistency import enforce_measurement_consistency
from hybrid.pipeline import hybrid_hr_depth


@pytest.fixture
def mock_predictions():
    np.random.seed(42)
    h_lr, w_lr = 24, 32
    scale = 4
    h_hr, w_hr = h_lr * scale, w_lr * scale

    lr_depth = np.random.uniform(0.8, 4.5, (h_lr, w_lr)).astype(np.float32)
    hr_rgb = np.random.randint(0, 256, (h_hr, w_hr, 3), dtype=np.uint8)

    d_c = np.random.uniform(0.8, 4.5, (h_hr, w_hr)).astype(np.float32)
    d_d = np.random.uniform(0.8, 4.5, (h_hr, w_hr)).astype(np.float32)

    return lr_depth, hr_rgb, d_c, d_d, scale


def test_confidence_normalization(mock_predictions):
    _, hr_rgb, d_c, d_d, _ = mock_predictions
    w_c, w_d = compute_confidence_weights(d_c, d_d, hr_rgb=hr_rgb)

    assert w_c.shape == d_c.shape
    assert w_d.shape == d_d.shape
    assert w_c.dtype == np.float32
    assert w_d.dtype == np.float32

    # Normalization guarantee: W_c + W_d == 1.0 everywhere
    total = w_c + w_d
    assert_allclose(total, np.ones_like(w_c), rtol=1e-5, atol=1e-5)


def test_fusion_blending(mock_predictions):
    _, hr_rgb, d_c, d_d, _ = mock_predictions
    w_c, w_d = compute_confidence_weights(d_c, d_d, hr_rgb=hr_rgb)

    fused = fuse_depth_maps(d_c, d_d, w_c, w_d)
    assert fused.shape == d_c.shape
    assert fused.dtype == np.float32
    assert np.all(np.isfinite(fused))

    fixed = fixed_weight_fusion(d_c, d_d, alpha=0.02)
    assert fixed.shape == d_c.shape
    assert fixed.dtype == np.float32
    assert np.all(np.isfinite(fixed))


def test_hybrid_pipeline_end_to_end(mock_predictions):
    lr_depth, hr_rgb, d_c, d_d, scale = mock_predictions
    h_hr, w_hr = hr_rgb.shape[:2]

    # Test with precomputed branch outputs
    out = hybrid_hr_depth(
        lr_depth,
        hr_rgb,
        scale=scale,
        deep_depth=d_d,
        classical_depth=d_c,
        method="confidence",
        use_consistency=True,
    )
    assert out.shape == (h_hr, w_hr)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))

    # Test standalone with automatic branch invocation
    out_auto = hybrid_hr_depth(lr_depth, hr_rgb, scale=scale, method="fixed", use_consistency=True)
    assert out_auto.shape == (h_hr, w_hr)
    assert out_auto.dtype == np.float32
    assert np.all(np.isfinite(out_auto))
