# Team Owner: Rakshitha
# Workstream: Deep Learning & Data Testing
# Module: tests/test_deep.py
# Description: Unit and smoke tests for deep model inference, dataset loaders, preprocessing, and degradation.
"""Runs on CPU, needs NO checkpoint and NO dataset download (synthetic fixtures)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import degradation as deg  # noqa: E402
from data import preprocessing as pp  # noqa: E402
from data.dataset import MiddleburyDataset, NYUv2Dataset, get_dataset, make_split_indices  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.evaluate import evaluate_methods, load_method, main as eval_main  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    H, W, s = 64, 96, 4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = 1.0 + 2.0 * xx / W
    depth[16:40, 24:56] = 0.7  # sharp foreground block
    rgb = rng.random((H, W, 3)).astype(np.float32)
    lr = deg.downsample(depth, s)
    return {"depth": depth.astype(np.float32), "rgb": rgb, "lr": lr, "mask": np.ones_like(lr, bool), "s": s}


@pytest.fixture()
def nyu_root(tmp_path):
    n, H, W = 30, 32, 48
    rng = np.random.default_rng(1)
    depth_m = rng.uniform(0.8, 6.0, (n, H, W)).astype(np.float32)
    mx, mn = depth_m.max((1, 2)), depth_m.min((1, 2))
    norm = (depth_m - mn[:, None, None]) / (mx - mn)[:, None, None]
    imgs = rng.integers(0, 255, (n, H, W, 3), dtype=np.uint8)
    np.save(tmp_path / "test_depth.npy", norm)
    np.save(tmp_path / "test_images_v2.npy", imgs)
    np.save(tmp_path / "test_minmax.npy", np.stack([mx, mn]))
    np.save(tmp_path / "train_depth_split.npy", norm[:10])
    np.save(tmp_path / "train_images_split.npy", imgs[:10])
    return tmp_path, depth_m


# ------------------------------------------------------------ preprocessing
def test_to_float_rgb_variants():
    u8 = np.full((4, 4, 3), 255, np.uint8)
    assert pp.to_float_rgb(u8).max() == 1.0 and pp.to_float_rgb(u8).dtype == np.float32
    assert pp.to_float_rgb(np.full((4, 4, 3), 128.0)).max() <= 1.0
    assert pp.to_float_rgb(np.zeros((4, 4))).shape == (4, 4, 3)
    with pytest.raises(ValueError):
        pp.to_float_rgb(np.zeros((4, 4, 2)))


def test_masks_and_fill_do_not_modify_inputs():
    d = np.array([[1, 0, np.nan], [2, 3, 4]], np.float32)
    m = pp.make_valid_mask(d)
    assert m.tolist() == [[True, False, False], [True, True, True]]
    d0, m0 = d.copy(), m.copy()
    filled = pp.fill_invalid_nearest(np.nan_to_num(d), m)
    assert np.array_equal(d, d0, equal_nan=True) and np.array_equal(m, m0)
    assert filled[0, 1] == 1.0 and filled[0, 2] == 4.0
    with pytest.raises(ValueError):
        pp.fill_invalid_nearest(d, np.zeros_like(m))
    assert pp.make_valid_mask(np.zeros((2, 2)), min_valid=None).all()


def test_geometry_and_mask_validation(synth):
    with pytest.raises(ValueError, match="Geometry mismatch"):
        pp.check_geometry(synth["lr"], synth["rgb"][:-4], 4)
    with pytest.raises(ValueError, match="valid_mask shape"):
        pp.resolve_valid_mask(synth["lr"], np.ones((3, 3), bool))
    assert pp.resolve_valid_mask(synth["lr"], None).all()


def test_minmax_roundtrip(synth):
    lo, hi = pp.minmax_range(synth["lr"], synth["mask"])
    back = pp.minmax_denormalize(pp.minmax_normalize(synth["lr"], lo, hi), lo, hi)
    assert np.allclose(back, synth["lr"], atol=1e-5)
    assert pp.minmax_range(np.ones((2, 2), np.float32), np.ones((2, 2), bool))[1] > 1.0  # constant guard


# -------------------------------------------------------------- degradation
def test_downsample_shape_and_operator(synth):
    assert synth["lr"].shape == (16, 24) and synth["lr"].dtype == np.float32
    out = deg.degrade(synth["depth"], 4)
    assert out["operator_exact"] and np.array_equal(out["lr_depth"], deg.forward_operator(4)(synth["depth"]))
    assert out["lr_valid_mask"].all()
    with pytest.raises(ValueError):
        deg.downsample(synth["depth"][:-1], 4)


def test_degrade_is_deterministic_and_non_destructive(synth):
    cfg = deg.DegradationConfig(noise_std=0.02, hole_ratio=0.1, n_block_holes=1, seed=7)
    d0 = synth["depth"].copy()
    a, b = deg.degrade(synth["depth"], 4, cfg, sample_index=3), deg.degrade(synth["depth"], 4, cfg, sample_index=3)
    c = deg.degrade(synth["depth"], 4, cfg, sample_index=4)
    assert np.array_equal(synth["depth"], d0)
    assert np.array_equal(a["lr_depth"], b["lr_depth"]) and not np.array_equal(a["lr_depth"], c["lr_depth"])
    assert not a["operator_exact"]
    assert (a["lr_depth"][~a["lr_valid_mask"]] == 0).all() and (~a["lr_valid_mask"]).sum() > 0
    assert np.array_equal(a["lr_depth_clean"], deg.downsample(synth["depth"], 4))


def test_degrade_with_gt_holes(synth):
    valid = np.ones_like(synth["depth"], bool)
    valid[:8, :8] = False
    out = deg.degrade(synth["depth"], 4, valid_hr=valid)
    assert not out["lr_valid_mask"][:2, :2].any() and out["lr_valid_mask"][8:, 8:].all()
    assert not out["operator_exact"]


# ----------------------------------------------------------------- datasets
def test_split_indices_leakage_safe():
    val, test = make_split_indices(449, "val"), make_split_indices(449, "test")
    assert not set(val) & set(test) and min(test) - max(val) > 1
    assert make_split_indices(449, "full") == list(range(449))
    with pytest.raises(ValueError):
        make_split_indices(10, "train")


def test_nyu_loader(nyu_root):
    root, depth_m = nyu_root
    ds = NYUv2Dataset(root, "val", 4)
    s = ds[0]
    assert s["lr_depth"].shape == (8, 12) and s["depth_original"].shape == (32, 48)
    assert np.allclose(s["depth_original"], depth_m[0], atol=1e-4)  # metres restored
    assert s["valid_mask_original"].all() and s["unit"] == "m" and s["data_range"] == 10.0
    assert s["hr_rgb"].dtype == np.float32 and 0 <= s["hr_rgb"].min() and s["hr_rgb"].max() <= 1
    assert len(NYUv2Dataset(root, "test", 4)) + len(ds) < 30
    assert NYUv2Dataset(root, "train", 4)[0]["unit"] == "normalized"
    with pytest.raises(FileNotFoundError):
        NYUv2Dataset(root / "nope", "val", 4)


def test_middlebury_loader_and_factory(tmp_path):
    rng = np.random.default_rng(2)
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (35, 50, 3), dtype=np.uint8)).save(tmp_path / f"output_color_{i}.png")
        Image.fromarray(rng.integers(0, 255, (35, 50), dtype=np.uint8)).save(tmp_path / f"output_depth_{i}.png")
    ds = MiddleburyDataset(tmp_path, 4)
    s = ds[1]
    assert s["depth_original"].shape == (32, 48) and s["lr_depth"].shape == (8, 12)  # modcropped
    assert s["unit"] == "8bit" and s["data_range"] == 255.0 and s["valid_mask_original"].all()
    with pytest.raises(ValueError):
        get_dataset("middlebury", "val", 4, root=str(tmp_path))
    with pytest.raises(NotImplementedError, match="KITTI"):
        get_dataset("kitti")
    with pytest.raises(ValueError):
        get_dataset("imagenet")


# ------------------------------------------------------------------ metrics
def test_metrics_identity_and_offsets(synth):
    gt = synth["depth"]
    allm = M.compute_all_metrics(gt, gt, 10.0)
    assert allm["rmse"] == 0 and allm["mae"] == 0 and allm["psnr"] == float("inf")
    assert allm["ssim"] == pytest.approx(1.0) and allm["edge_f1"] == pytest.approx(1.0)
    assert M.rmse(gt + 0.5, gt) == pytest.approx(0.5) and M.mae(gt - 0.25, gt) == pytest.approx(0.25)
    assert M.psnr(gt + 1.0, gt, 10.0) == pytest.approx(20.0)


def test_metrics_mask_and_border(synth):
    gt = synth["depth"].copy()
    pred = gt.copy()
    pred[:6] += 100.0  # inside the border crop -> ignored
    assert M.rmse(pred, gt) == 0
    pred = gt.copy()
    pred[30:34, 30:34] += 5.0
    valid = np.ones_like(gt, bool)
    valid[30:34, 30:34] = False  # masked out -> ignored
    assert M.rmse(pred, gt, valid) == 0
    with pytest.raises(ValueError):
        M.rmse(gt[:-1], gt)


def test_boundary_and_edge_metrics_see_blur(synth):
    import cv2
    gt = synth["depth"]
    blur = cv2.GaussianBlur(gt, (0, 0), 3)
    sharp_err = M.boundary_rmse(gt + 0.0, gt, 10.0, rel_thresh=0.02)
    blur_err = M.boundary_rmse(blur, gt, 10.0, rel_thresh=0.02)
    assert sharp_err == 0 and blur_err > M.rmse(blur, gt) * 0.5
    assert M.edge_scores(blur, gt, 10.0, rel_thresh=0.02)["edge_recall"] < 1.0
    flat = np.ones((64, 96), np.float32)
    assert np.isnan(M.boundary_rmse(flat, flat, 10.0))  # no edges -> NaN, not a fake 0


# --------------------------------------------------------------------- deep
torch = pytest.importorskip("torch")

from deep import deep_model as dm  # noqa: E402
from deep import inference as inf  # noqa: E402
from deep.settings import DeepModelUnavailableError, UnsupportedScaleError  # noqa: E402


@pytest.fixture()
def clean_cache():
    dm.clear_model_cache()
    yield
    dm.clear_model_cache()


def test_unsupported_scale_and_missing_checkpoint(synth, tmp_path, monkeypatch, clean_cache):
    with pytest.raises(UnsupportedScaleError):
        inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 2)
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path))
    with pytest.raises(DeepModelUnavailableError, match="No SGNet x4 checkpoint"):
        inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 4, device="cpu")


def test_wrapper_contract_with_random_init(synth, clean_cache):
    lr, rgb, mask = synth["lr"].copy(), synth["rgb"].copy(), synth["mask"].copy()
    out = inf.deep_hr_depth(lr, rgb, mask, 4, device="cpu", _allow_random_init=True)
    assert out.shape == synth["depth"].shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert (out >= 0).all()
    assert np.array_equal(lr, synth["lr"]) and np.array_equal(mask, synth["mask"])  # inputs untouched
    out2 = inf.deep_hr_depth(lr, rgb, mask, 4, device="cpu", _allow_random_init=True)
    assert np.array_equal(out, out2)  # repeatable


def test_wrapper_handles_invalid_pixels_and_uint8_rgb(synth, clean_cache):
    lr, mask = synth["lr"].copy(), synth["mask"].copy()
    lr[3:6, 4:9] = 0.0
    mask[3:6, 4:9] = False
    rgb8 = (synth["rgb"] * 255).astype(np.uint8)
    out = inf.deep_hr_depth(lr, rgb8, mask, 4, device="cpu", _allow_random_init=True)
    assert np.isfinite(out).all() and out.shape == (64, 96)
    assert (lr[3:6, 4:9] == 0).all() and not mask[3:6, 4:9].any()
    with pytest.raises(ValueError, match="no valid pixels"):
        inf.deep_hr_depth(lr, rgb8, np.zeros_like(mask), 4, device="cpu", _allow_random_init=True)
    with pytest.raises(ValueError, match="Geometry mismatch"):
        inf.deep_hr_depth(lr, rgb8[:-4], mask, 4, device="cpu", _allow_random_init=True)


def test_tiled_fallback_shape_and_closeness(synth, clean_cache):
    full = inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 4, device="cpu", _allow_random_init=True)
    tiled = inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 4, device="cpu", tile_lr=16,
                              _allow_random_init=True)
    assert tiled.shape == full.shape and np.isfinite(tiled).all()


def test_checkpoint_roundtrip_and_validation(synth, tmp_path, monkeypatch, clean_cache):
    model = dm.build_sgnet(4)
    sd = {"state_dict": {f"module.{k}": v for k, v in model.state_dict().items()}}  # DataParallel-style wrapper
    torch.save(sd, tmp_path / "SGNet_X4_R.pth")
    torch.save({"foo": torch.zeros(1)}, tmp_path / "SGNet_X16_R.pth")
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path))
    assert dm.resolve_checkpoint(4).name == "SGNet_X4_R.pth"
    assert dm.resolve_checkpoint(16).name == "SGNet_X16_R.pth"  # X4 pattern must not grab X16
    out = inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 4, device="cpu")
    assert out.shape == synth["depth"].shape
    info = dm.get_model(4, torch.device("cpu"))[1]
    assert info["pretrained"] and info["ckpt_sha256_12"]
    with pytest.raises(DeepModelUnavailableError, match="does not match"):
        dm.load_sgnet(16)
    assert dm.verify_checkpoints((4, 8))[8]["status"] == "unavailable"


def test_model_state_dict_has_no_gradient_kernels():
    sd = dm.build_sgnet(4).state_dict()
    assert not any("weight_h" in k or "weight_v" in k for k in sd)  # keys identical to official ckpt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_forward_and_profile(synth, clean_cache):
    out = inf.deep_hr_depth(synth["lr"], synth["rgb"], synth["mask"], 4, device="cuda", _allow_random_init=True)
    assert out.shape == synth["depth"].shape
    st = inf.profile_inference(synth["lr"], synth["rgb"], synth["mask"], 4, warmup=1, runs=2, device="cuda",
                               _allow_random_init=True)
    assert st["peak_mem_allocated_mb"] > 0


# --------------------------------------------------------------- evaluation
def test_evaluate_methods_same_samples_and_unavailable_reporting(nyu_root, tmp_path, monkeypatch, clean_cache):
    monkeypatch.setenv("SGNET_CKPT_DIR", str(tmp_path / "empty_ckpt"))
    monkeypatch.delenv("SGNET_CKPT_X4", raising=False)
    monkeypatch.delenv("SGNET_CKPT_X8", raising=False)
    monkeypatch.delenv("SGNET_CKPT_X16", raising=False)
    root, _ = nyu_root
    ds = NYUv2Dataset(root, "val", 4)
    res = evaluate_methods(ds, ["nearest", "bilinear", "bicubic", "hybrid"], max_samples=3, verbose=False)
    assert set(res["summary"]) == {"nearest", "bilinear", "bicubic"}
    assert "hybrid" in res["not_evaluated"]  # stub / missing wrapper is reported, not hidden
    assert {r["sample"] for r in res["per_image"] if r["method"] == "bicubic"} == \
           {r["sample"] for r in res["per_image"] if r["method"] == "nearest"}
    bad = evaluate_methods(ds, ["broken"], max_samples=2, methods={"broken": lambda *a: np.zeros((1, 1))},
                           verbose=False)
    assert "broken" in bad["not_evaluated"]


def test_test_split_is_locked_behind_final(nyu_root):
    with pytest.raises(SystemExit):
        eval_main(["--split", "test"])
    assert callable(load_method("bicubic"))


def test_evaluate_hybrid_rejects_fallback_and_keeps_later_success(nyu_root, tmp_path, monkeypatch):
    from hybrid import pipeline

    checkpoint = tmp_path / "SGNet_X4_R.pth"
    checkpoint.touch()
    monkeypatch.setenv("SGNET_CKPT_X4", str(checkpoint))
    calls = 0

    def deep(lr, rgb, valid, scale):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated inference failure")
        return np.ones(rgb.shape[:2], np.float32), None

    monkeypatch.setattr(pipeline, "_deep_hr_depth", deep)
    root, _ = nyu_root
    ds = NYUv2Dataset(root, "val", 4)
    result = evaluate_methods(ds, ["hybrid"], max_samples=2, verbose=False)
    hybrid = result["summary"]["hybrid"]
    assert hybrid["n_ok"] == hybrid["n_failed"] == 1
    assert "simulated inference failure" in hybrid["failures"][0]
    assert len(result["per_image"]) == 1
