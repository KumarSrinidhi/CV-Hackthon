# End-to-end integration and smoke tests across all three reconstruction branches.

import numpy as np
import pytest
from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.pipeline import hybrid_hr_depth
from eval.metrics import compute_all_metrics


def test_end_to_end_contracts():
    np.random.seed(42)
    h_lr, w_lr = 20, 30
    scale = 4
    h_hr, w_hr = h_lr * scale, w_lr * scale

    hr_rgb = np.random.randint(0, 256, (h_hr, w_hr, 3), dtype=np.uint8)
    gt_depth = np.random.uniform(1.0, 4.0, (h_hr, w_hr)).astype(np.float32)
    lr_depth = gt_depth[::scale, ::scale].copy()

    # 1. Classical branch
    d_classical = classical_hr_depth(lr_depth, hr_rgb, scale=scale, method="jbu")
    assert d_classical.shape == (h_hr, w_hr)
    assert d_classical.dtype == np.float32
    assert np.all(np.isfinite(d_classical))

    # 2. Deep branch
    d_deep = deep_hr_depth(lr_depth, hr_rgb, scale=scale, device="cpu")
    assert d_deep.shape == (h_hr, w_hr)
    assert d_deep.dtype == np.float32
    assert np.all(np.isfinite(d_deep))

    # 3. Hybrid branch
    d_hybrid = hybrid_hr_depth(
        lr_depth,
        hr_rgb,
        scale=scale,
        deep_depth=d_deep,
        classical_depth=d_classical,
        method="fixed",
        use_consistency=True,
    )
    assert d_hybrid.shape == (h_hr, w_hr)
    assert d_hybrid.dtype == np.float32
    assert np.all(np.isfinite(d_hybrid))

    # 4. Evaluation metrics
    metrics = compute_all_metrics(d_hybrid, gt_depth)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "psnr" in metrics
    assert "boundary_rmse" in metrics
    assert "edge_f1" in metrics
    assert metrics["psnr"] >= 0.0


def test_official_depthsr_testset():
    import os, cv2
    from data.dataset import RGBDDataset

    testset_dir = "data/depthsr_testset"
    if not os.path.isdir(testset_dir):
        pytest.skip("Official testset data/depthsr_testset not present")

    ds = RGBDDataset(data_dir=testset_dir, scale=4)
    assert len(ds) == 20, f"Expected 20 testset samples, got {len(ds)}"

    sample = ds[0]
    assert sample["rgb"].shape == (512, 512, 3)
    assert sample["gt_depth"].shape == (512, 512)
    assert sample["lr_depth"].shape == (128, 128)
    assert sample["valid_mask"].shape == (128, 128)

    # Run Classical
    d_c = classical_hr_depth(sample["lr_depth"], sample["rgb"], valid_mask=sample["valid_mask"], scale=4, method="jbu")
    assert d_c.shape == (512, 512)
    assert np.all(np.isfinite(d_c))

    # Run Deep
    d_d = deep_hr_depth(sample["lr_depth"], sample["rgb"], valid_mask=sample["valid_mask"], scale=4, device="cpu")
    assert d_d.shape == (512, 512)
    assert np.all(np.isfinite(d_d))

    # Run Hybrid
    d_h = hybrid_hr_depth(
        sample["lr_depth"], sample["rgb"], valid_mask=sample["valid_mask"],
        scale=4, alpha=0.5, deep_depth=d_d, classical_depth=d_c,
        method="fixed", use_consistency=True
    )
    assert d_h.shape == (512, 512)
    assert np.all(np.isfinite(d_h))

    metrics = compute_all_metrics(d_h, sample["gt_depth"], valid_mask=sample["valid_mask"])
    assert metrics["rmse"] < 0.20, f"RMSE too high: {metrics['rmse']}"
    assert metrics["psnr"] > 30.0, f"PSNR too low: {metrics['psnr']}"
    assert metrics["edge_f1"] > 0.70, f"Edge F1 too low: {metrics['edge_f1']}"
