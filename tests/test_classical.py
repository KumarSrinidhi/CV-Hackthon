# Team Owner: Tanush
# Workstream: Classical Computer Vision Testing
# Module: tests/test_classical.py
# Description: Unit tests for interpolation baselines, JBU, guided filter, boundary refinement, and hole filling.

import numpy as np
import pytest
from classical.interpolation import nearest_upsample, bilinear_upsample, bicubic_upsample
from classical.jbu import joint_bilateral_upsample
from classical.guided_filter import guided_filter_depth
from classical.hole_filling import conservative_hole_filling, mask_invalid_depth
from classical.pipeline import classical_hr_depth


@pytest.fixture
def dummy_pair():
    np.random.seed(42)
    h_lr, w_lr = 30, 40
    scale = 4
    h_hr, w_hr = h_lr * scale, w_lr * scale
    lr_depth = np.random.uniform(0.5, 5.0, (h_lr, w_lr)).astype(np.float32)
    hr_rgb = np.random.randint(0, 256, (h_hr, w_hr, 3), dtype=np.uint8)
    return lr_depth, hr_rgb, scale


def test_interpolation_shapes(dummy_pair):
    lr_depth, _, scale = dummy_pair
    h_lr, w_lr = lr_depth.shape
    target_size = (h_lr * scale, w_lr * scale)

    nn_out = nearest_upsample(lr_depth, target_size=target_size)
    assert nn_out.shape == target_size
    assert nn_out.dtype == np.float32

    bi_out = bilinear_upsample(lr_depth, target_size=target_size)
    assert bi_out.shape == target_size
    assert bi_out.dtype == np.float32

    cu_out = bicubic_upsample(lr_depth, target_size=target_size)
    assert cu_out.shape == target_size
    assert cu_out.dtype == np.float32


def test_jbu_execution(dummy_pair):
    lr_depth, hr_rgb, scale = dummy_pair
    h_hr, w_hr = hr_rgb.shape[:2]
    out = joint_bilateral_upsample(lr_depth, hr_rgb, scale=scale, radius=2)
    assert out.shape == (h_hr, w_hr)
    assert np.all(np.isfinite(out))


def test_guided_filter(dummy_pair):
    lr_depth, hr_rgb, scale = dummy_pair
    h_hr, w_hr = hr_rgb.shape[:2]
    init_hr = bicubic_upsample(lr_depth, target_size=(h_hr, w_hr))
    out = guided_filter_depth(hr_rgb, init_hr, radius=4, eps=1e-3)
    assert out.shape == (h_hr, w_hr)
    assert np.all(np.isfinite(out))


def test_hole_filling():
    np.random.seed(42)
    depth = np.ones((20, 20), dtype=np.float32) * 2.5
    valid_mask = np.ones((20, 20), dtype=bool)
    # create a hole
    depth[5:10, 5:10] = 0.0
    valid_mask[5:10, 5:10] = False

    filled, updated_mask = conservative_hole_filling(depth, valid_mask, max_hole_radius=5)
    # Valid pixels must stay unchanged
    assert np.all(filled[valid_mask] == depth[valid_mask])
    # The hole must be filled with non-zero values
    assert np.all(filled[~valid_mask] > 0.0)


def test_classical_pipeline_wrapper(dummy_pair):
    lr_depth, hr_rgb, scale = dummy_pair
    h_hr, w_hr = hr_rgb.shape[:2]
    hr = classical_hr_depth(lr_depth, hr_rgb, scale=scale, method="jbu")
    assert hr.shape == (h_hr, w_hr)
    assert hr.dtype == np.float32
    assert np.all(np.isfinite(hr))
