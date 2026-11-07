# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: tests/test_deep.py
# Description: Unit tests for deep learning architecture, forward pass, and inference wrapper.

import numpy as np
import pytest
from deep.deep_model import DeepDepthSRModel
from deep.inference import deep_hr_depth


def test_deep_model_forward():
    model_wrapper = DeepDepthSRModel(device="cpu")
    assert model_wrapper.model is not None

    h_lr, w_lr = 24, 32
    scale = 4
    h_hr, w_hr = h_lr * scale, w_lr * scale

    lr_depth = np.random.uniform(0.5, 4.0, (h_lr, w_lr)).astype(np.float32)
    hr_rgb = np.random.randint(0, 256, (h_hr, w_hr, 3), dtype=np.uint8)

    pred = model_wrapper.predict(lr_depth, hr_rgb, scale=scale)
    assert pred.shape == (h_hr, w_hr)
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


def test_deep_hr_depth_wrapper():
    h_lr, w_lr = 16, 20
    scale = 4
    h_hr, w_hr = h_lr * scale, w_lr * scale

    lr_depth = np.random.uniform(1.0, 5.0, (h_lr, w_lr)).astype(np.float32)
    hr_rgb = np.random.randint(0, 256, (h_hr, w_hr, 3), dtype=np.uint8)

    pred = deep_hr_depth(lr_depth, hr_rgb, scale=scale, device="cpu")
    assert pred.shape == (h_hr, w_hr)
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))
