"""End-to-end integration tests across classical and hybrid branches."""

import numpy as np
import pytest

from classical import classical_hr_depth
from classical.interpolation import interpolate_depth
from classical.pipeline import reconstruct
from hybrid import hybrid_hr_depth


def _scene(scale=4):
    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    rgb = np.zeros((16, 16, 3), np.float32)
    rgb[:, 7:] = 1
    return lr, rgb


def _fake_deep(value=3.0, shape=(16, 16)):
    def fn(lr_depth, hr_rgb, valid_mask, scale):
        return np.full(shape, value, np.float32)

    return fn


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
def test_interpolation_baselines(method):
    lr, _ = _scene(4)
    out = interpolate_depth(lr, None, 4, method)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("method", ["jbu", "guided", "classical", "adaptive_jbu",
                                   "refined_jbu", "masked_jbu", "tuned_jbu"])
def test_classical_methods_produce_valid_output(method):
    lr, rgb = _scene(4)
    out = reconstruct(lr, rgb, None, 4, method=method)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.max() <= 5.0 + 1e-3


def test_classical_and_hybrid_share_shape():
    lr, rgb = _scene(4)
    c = classical_hr_depth(lr, rgb, None, 4)
    h = hybrid_hr_depth(lr, rgb, None, 4, deep_fn=_fake_deep())
    assert c.shape == h.shape == (16, 16)


def test_hybrid_requires_deep_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path / "empty_ckpt"))
    lr, rgb = _scene(4)
    with pytest.raises(NotImplementedError):
        hybrid_hr_depth(lr, rgb, None, 4)


def test_scale_two_classical_and_hybrid():
    lr = np.full((6, 8), 3.0, np.float32)
    rgb = np.zeros((12, 16, 3), np.float32)
    assert classical_hr_depth(lr, rgb, None, 2).shape == (12, 16)
    assert hybrid_hr_depth(lr, rgb, None, 2, deep_fn=_fake_deep(shape=(12, 16))).shape == (12, 16)
