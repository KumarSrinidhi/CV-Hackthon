"""Tests for visualization, export, and the CLI reconstruction dispatch."""

import os

import numpy as np
import pytest

from app.visualization import (
    boundaries_overlay,
    depth_colormap,
    error_map,
    export_npy,
    export_png,
    normalize_depth,
)
from main import reconstruct


def test_depth_colormap_shape_and_type():
    d = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    cmap = depth_colormap(d)
    assert cmap.shape == (4, 4, 3)
    assert cmap.dtype == np.uint8


def test_normalize_depth_maps_invalid_to_zero():
    d = np.array([[1, 2], [0, 0]], np.float32)
    norm = normalize_depth(d)
    assert norm[0, 1] == pytest.approx(1.0)
    assert norm[0, 0] == pytest.approx(0.0)
    assert norm[1, 1] == pytest.approx(0.0)


def test_boundaries_overlay_marks_depth_edges():
    rgb = np.zeros((4, 4, 3), np.uint8)
    d = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    out = boundaries_overlay(rgb, d)
    assert out.shape == (4, 4, 3)
    assert (out[:, 1] == (0, 0, 255)).all()
    assert (out[:, 2] == (0, 0, 255)).all()


def test_error_map_zero_for_identical():
    gt = np.full((4, 4), 3.0, np.float32)
    emap = error_map(gt, gt)
    assert (emap == 0).all()


def test_export_png_and_npy(tmp_path):
    d = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    png = export_png(d, str(tmp_path / "d.png"))
    npy = export_npy(d, str(tmp_path / "d.npy"))
    assert os.path.exists(png) and os.path.exists(npy)
    np.testing.assert_array_equal(np.load(npy), d)


def test_reconstruct_nearest_shape():
    rgb = np.zeros((8, 8, 3), np.uint8)
    lr = np.full((2, 2), 4.0, np.float32)
    out = reconstruct(rgb, lr, None, 4, "nearest")
    assert out["depth"].shape == (8, 8)
    assert out["depth"].dtype == np.float32


def test_reconstruct_hybrid_shape():
    rgb = np.zeros((16, 16, 3), np.uint8)
    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    out = reconstruct(rgb, lr, None, 4, "hybrid")
    assert out["depth"].shape == (16, 16)
    assert np.isfinite(out["depth"]).all()


def test_reconstruct_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        reconstruct(np.zeros((8, 8, 3), np.uint8), np.ones((2, 2), np.float32), None, 4, "nope")
