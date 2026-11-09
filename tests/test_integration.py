"""Application-layer integration tests: config, validation, hybrid wiring, CLI, UI core."""

import cv2
import numpy as np
import pytest

import hybrid.pipeline as hp
from app import config as appcfg
from app.validation import validate_inputs, validate_mask, validate_scale
from app.visualization import shared_depth_colormap, side_by_side
from classical import classical_hr_depth
from hybrid.consistency import enforce_consistency
from hybrid.fusion import fuse_fixed
from main import comparison_image, main, reconstruct


def _scene():
    lr = np.tile(np.array([2, 2, 5, 5], np.float32), (4, 1))
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:, 7:] = 200
    return lr, rgb


@pytest.fixture
def fake_deep(monkeypatch):
    """Stand-in deep branch (no torch / checkpoint needed)."""
    def fn(lr_depth, hr_rgb, valid_mask, scale):
        return cv2.resize(np.asarray(lr_depth, np.float32), (16, 16), interpolation=cv2.INTER_CUBIC) * 1.01
    monkeypatch.setattr(hp, "_deep_hr_depth", lambda *args: (fn(*args), None))
    return fn


# ---- config -------------------------------------------------------------
def test_default_hybrid_config_is_adaptive_plus_consistency():
    kw = appcfg.hybrid_kwargs(appcfg.load_config())
    assert kw["classical_weight"] is None
    assert kw["use_consistency"] is True
    assert kw["use_refinement"] is False


def test_config_file_missing_falls_back_to_defaults(tmp_path):
    cfg = appcfg.load_config(tmp_path / "nope.yaml")
    assert cfg["hybrid"]["classical_alpha"] is None


def test_config_rejects_bad_alpha(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("hybrid:\n  classical_alpha: 1.5\n")
    with pytest.raises(ValueError, match="classical_alpha"):
        appcfg.load_config(p)


# ---- validation ---------------------------------------------------------
def test_validation_geometry_message():
    lr, rgb = _scene()
    with pytest.raises(ValueError, match="at scale x8"):
        validate_inputs(rgb, lr, None, 8)


def test_validation_rejects_bad_inputs():
    lr, rgb = _scene()
    with pytest.raises(ValueError, match="RGB guidance image is required"):
        validate_inputs(None, lr, None, 4)
    with pytest.raises(ValueError, match="no valid"):
        validate_inputs(rgb, np.zeros((4, 4), np.float32), None, 4)
    with pytest.raises(ValueError, match="deep branch supports"):
        validate_scale(2, "deep")
    with pytest.raises(ValueError, match="whole number"):
        validate_scale(2.5, "classical")


def test_mask_validation():
    assert validate_mask(None, (4, 4)) is None
    m = validate_mask(np.array([[255, 0], [1, 0]], np.uint8), (2, 2))
    assert m.dtype == bool and m.tolist() == [[True, False], [True, False]]
    with pytest.raises(ValueError, match="must match"):
        validate_mask(np.ones((3, 3)), (2, 2))
    with pytest.raises(ValueError, match="every pixel invalid"):
        validate_mask(np.zeros((2, 2)), (2, 2))


# ---- hybrid wiring ------------------------------------------------------
def test_hybrid_uses_fixed_alpha_and_consistency(fake_deep):
    lr, rgb = _scene()
    cfg = appcfg.load_config()
    cfg["hybrid"]["classical_alpha"] = 0.02
    res = reconstruct(rgb, lr, None, 4, cfg)
    assert not res["fallback"] and res["method"] == "hybrid"
    inter = res["intermediates"]
    assert np.allclose(inter["confidence_classical"], 0.02)
    assert np.allclose(inter["confidence_deep"], 0.98)

    Dc = classical_hr_depth(lr, rgb, None, 4)
    Dd = fake_deep(lr, rgb, None, 4)
    expected = enforce_consistency(fuse_fixed(Dc, Dd, 0.02), lr, None, 4, strength=0.25)
    np.testing.assert_allclose(res["depth"], expected, atol=1e-5)
    assert not np.allclose(res["depth"], fuse_fixed(Dc, Dd, 0.02))


def test_hybrid_degrades_without_deep(monkeypatch, tmp_path):
    from app.ui import process

    def unavailable(*args):
        raise OSError("PyTorch runtime could not load")

    monkeypatch.setattr(hp, "_deep_hr_depth", unavailable)
    lr, rgb = _scene()
    res = reconstruct(rgb, lr, None, 4)
    assert not res["fallback"] and res["method"] == "hybrid"
    assert not res["intermediates"].get("deep_available", True)
    np.testing.assert_array_equal(res["depth"], classical_hr_depth(lr, rgb, None, 4))

    depth_path = tmp_path / "lr.npy"
    np.save(depth_path, lr)
    status = process(rgb, str(depth_path), None, 4, export_dir=str(tmp_path))["status"]
    assert "Pipeline: classical fallback" in status
    assert status.count("Deep branch unavailable") == 1
    assert "PyTorch runtime could not load" in status


def test_hybrid_scale2_is_classical_fallback():
    lr = np.full((6, 8), 3.0, np.float32)
    rgb = np.zeros((12, 16, 3), np.uint8)
    res = reconstruct(rgb, lr, None, 2)
    assert res["fallback"] and "x2" in res["notes"][0]
    assert res["depth"].shape == (12, 16)


# ---- comparison image ---------------------------------------------------
def test_comparison_image_contains_branches(fake_deep):
    lr, rgb = _scene()
    res = reconstruct(rgb, lr, None, 4)
    img = comparison_image(rgb, res, lr_depth=lr)
    assert img.dtype == np.uint8 and img.ndim == 3


def test_side_by_side_and_shared_colormap():
    a = np.full((4, 4), 1.0, np.float32)
    b = np.full((4, 4), 9.0, np.float32)
    ca, cb = shared_depth_colormap([a, b])
    assert not np.array_equal(ca, cb)
    assert side_by_side([("a", ca), ("b", cb)]).shape == (4 + 22, 8, 3)
    with pytest.raises(ValueError):
        side_by_side([])


# ---- CLI ----------------------------------------------------------------
def test_cli_end_to_end_with_gt_and_compare(tmp_path, fake_deep):
    lr, rgb = _scene()
    cv2.imwrite(str(tmp_path / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(tmp_path / "lr.npy", lr)
    gt = cv2.resize(lr, (16, 16), interpolation=cv2.INTER_NEAREST)
    np.save(tmp_path / "gt.npy", gt)
    out = tmp_path / "out"
    rc = main(["--rgb", str(tmp_path / "rgb.png"), "--depth", str(tmp_path / "lr.npy"),
               "--gt", str(tmp_path / "gt.npy"), "--compare", "--output", str(out)])
    assert rc == 0
    for name in ("hybrid_x4.npy", "hybrid_x4.png", "hybrid_x4_colormap.png",
                 "hybrid_x4_error.png", "comparison_x4.png"):
        assert (out / name).exists(), name
    assert np.load(out / "hybrid_x4.npy").shape == (16, 16)


def test_cli_reports_clear_errors(tmp_path):
    assert main(["--rgb", "missing.png", "--depth", "missing.npy"]) == 1


# ---- UI core ------------------------------------------------------------
def test_ui_process_outputs_and_exports(tmp_path, fake_deep):
    from app.ui import _metrics_html, process

    lr, rgb = _scene()
    np.save(tmp_path / "lr.npy", lr)
    gt = cv2.resize(lr, (16, 16), interpolation=cv2.INTER_NEAREST)
    np.save(tmp_path / "gt.npy", gt)
    o = process(rgb, str(tmp_path / "lr.npy"), None, 4, str(tmp_path / "gt.npy"),
                True, export_dir=str(tmp_path / "ex"))
    for key in ("depth", "boundaries", "conf_classical", "conf_deep", "error"):
        assert o[key].shape == (16, 16, 3) and o[key].dtype == np.uint8
    assert o["comparison"].ndim == 3
    assert "input-adaptive (classical 2.0%, deep 98.0%)" in o["status"]
    assert "Branch RMSE" in o["status"]
    report = _metrics_html(o["run_metrics"], o["metrics"])
    assert "Ground-truth accuracy" in report and "Classical RMSE" in report and "Deep RMSE" in report
    assert len(o["files"]) == 2 and all(__import__("os").path.exists(f) for f in o["files"])


def test_ui_reports_run_metrics_without_ground_truth(tmp_path, monkeypatch):
    from app.ui import _metrics_html, process

    monkeypatch.setattr(hp, "_run_deep", lambda *args: (None, None, "Test runtime unavailable"))
    rgb = np.zeros((32, 32, 3), np.uint8)
    lr = np.full((8, 8), 6, np.float32)
    mask = np.ones_like(lr)
    mask[:, :2] = 0
    lr[:, :2] = 99
    depth_path, mask_path = tmp_path / "lr.npy", tmp_path / "mask.npy"
    np.save(depth_path, lr)
    np.save(mask_path, mask)

    out = process(rgb, depth_path, mask_path, 4, export_dir=str(tmp_path))
    assert out["metrics"] is None
    assert out["run_metrics"]["lr_coverage"] == 0.75
    assert out["run_metrics"]["output_coverage"] == 1
    assert out["run_metrics"]["measurement_rmse"] == pytest.approx(0)
    report = _metrics_html(out["run_metrics"], out["metrics"])
    assert "75.0%" in report and "LR fit RMSE" in report
    assert "Add aligned ground truth" in report and "PSNR" not in report


def test_ui_x8_masks_missing_ground_truth_and_downsamples_full_size_depth(tmp_path, monkeypatch):
    from app.ui import process

    monkeypatch.setattr(hp, "_run_deep", lambda *args: (None, None, "Test runtime unavailable"))
    rgb = np.zeros((64, 64, 3), np.uint8)
    depth = np.full((64, 64), 6, np.float32)
    depth[:, 24:40] = 0
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, depth)

    out = process(rgb, depth_path, None, 8, depth_path, export_dir=str(tmp_path))
    assert out["metrics"]["rmse"] == 0
    assert "Depth downsampled from 64x64 to 8x8 using bicubic" in out["status"]
    assert "LR input vs bicubic-downsampled GT RMSE 0.0000 over 48/64 LR pixels" in out["status"]


def test_ui_process_requires_inputs(tmp_path):
    from app.ui import process

    with pytest.raises(ValueError, match="RGB"):
        process(None, "x", None, 4)
    with pytest.raises(ValueError, match="depth map is required"):
        process(np.zeros((16, 16, 3), np.uint8), None, None, 4)
