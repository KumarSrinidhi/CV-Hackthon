import numpy as np
import pytest

from ps4.data.degrade import degrade, downsample


def make_depth(h=64, w=64):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return 1.0 + 3.0 * (xx / w)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_downsample_shape(factor):
    out = downsample(make_depth(), factor)
    assert out.shape == (64 // factor, 64 // factor)


def test_degrade_deterministic_with_seed():
    d = make_depth()
    a = degrade(d, factor=4, noise_sigma=0.02, hole_ratio=0.2, seed=42)
    b = degrade(d, factor=4, noise_sigma=0.02, hole_ratio=0.2, seed=42)
    np.testing.assert_array_equal(a, b)


def test_degrade_different_seeds_differ():
    d = make_depth()
    a = degrade(d, factor=4, noise_sigma=0.02, hole_ratio=0.2, seed=1)
    b = degrade(d, factor=4, noise_sigma=0.02, hole_ratio=0.2, seed=2)
    assert not np.array_equal(a, b)


def test_degrade_no_holes_keeps_nonzero():
    d = make_depth()
    out = degrade(d, factor=4, noise_sigma=0.0, hole_ratio=0.0, seed=0)
    assert (out > 0).all()


def test_bad_factor_raises():
    with pytest.raises(ValueError, match="positive integer"):
        downsample(make_depth(), 3.5)
