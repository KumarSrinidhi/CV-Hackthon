"""Tests for visualization, export, and the CLI reconstruction dispatch."""

import os

import numpy as np
import pytest

from app.visualization import (
    boundaries_overlay,
    confidence_overlay,
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


def test_confidence_overlay_heat_is_rgb():
    # JET heat at confidence 0 is blue in RGB; the blend with a red base must
    # therefore contain blue. (A BGR heat would show zero blue here.)
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = 255
    out = confidence_overlay(rgb, np.zeros((4, 4), np.float32))
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    assert out[..., 2].max() > 50


def test_run_reconstruction_returns_rgb_and_red_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path / "empty_ckpt"))
    from app.ui import run_reconstruction

    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    depth_file = tmp_path / "lr.npy"
    np.save(depth_file, lr)
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:, 7:] = 200
    colormap, boundaries, conf_c, conf_d = run_reconstruction(rgb, str(depth_file), None, 4, "hybrid")
    for img in (colormap, boundaries, conf_c, conf_d):
        assert img.shape == (16, 16, 3) and img.dtype == np.uint8
    assert (boundaries == np.uint8([255, 0, 0])).all(axis=2).any()  # red in RGB
    assert not (boundaries == np.uint8([0, 0, 255])).all(axis=2).any()  # no BGR-red artefact


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
