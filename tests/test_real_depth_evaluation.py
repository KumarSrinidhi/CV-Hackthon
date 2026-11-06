import cv2
import numpy as np
import pytest

from classical.evaluation.real_depth import METRICS, evaluate, middlebury_frames, project_velodyne, quality, summarize


def test_dense_metrics_mask_unknown_reference():
    target = np.ones((24, 24), dtype=np.float32)
    target[:, 12:] = 2
    valid = np.ones_like(target, dtype=bool)
    valid[0, :] = False
    prediction = target.copy()
    prediction[0, :] = 100
    metrics = quality(prediction, target, valid, sparse=False, peak=10)
    assert metrics["rmse_m"] == metrics["mae_m"] == metrics["boundary_rmse_m"] == 0
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1)
    assert metrics["edge_f1"] == metrics["coverage"] == 1


def test_kitti_scores_only_lidar_points_excluded_from_input():
    target = np.ones((16, 24), dtype=np.float32)
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rows = list(evaluate([("drive", "frame", rgb, target)], "kitti360", 1, None))
    for row in rows:
        assert row["scored"] == target.size - sum(target.size // scale**2 for scale in (2, 4, 8))
        assert row["lr_valid"] == target.size // row["scale"]**2
        assert row["rmse_m"] == 0
        assert row["ssim"] is row["psnr_db"] is row["edge_f1"] is row["boundary_rmse_m"] is None
        assert row["latency_ms"] > 0 and row["peak_traced_mib"] > 0


def test_middlebury_disparity_conversion(tmp_path):
    folder = tmp_path / "example-perfect"
    folder.mkdir()
    disparity = np.array([[1, 2], [np.inf, 3]], dtype=np.float32)
    assert cv2.imwrite(str(folder / "disp0.pfm"), disparity)
    assert cv2.imwrite(str(folder / "im0.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (folder / "calib.txt").write_text("cam0=[100 0 1; 0 100 1; 0 0 1]\nbaseline=200\ndoffs=2\n")
    [(group, frame, _, depth)] = list(middlebury_frames(tmp_path))
    assert (group, frame) == ("example", "example-perfect")
    np.testing.assert_allclose(depth, [[20 / 3, 5], [0, 4]])


def test_summary_averages_frames_within_each_group():
    rows = [dict.fromkeys(METRICS) | dict(dataset="kitti360", scale=2, method="jbu",
                                         group=group, rmse_m=score)
            for group, score in (("drive-a", 1), ("drive-a", 3), ("drive-b", 8))]
    [result] = list(summarize(rows))
    assert result["groups"] == 2
    assert result["mean"] == 5


def test_kitti_projection_keeps_closest_camera_depth():
    points = np.array([[1, 1, 1], [2, 2, 2], [6, 0, 2], [0, 0, -1]])
    transform = np.eye(4)[:3]
    result = project_velodyne(points, (2, 2), transform, transform)
    np.testing.assert_array_equal(result, [[0, 0], [0, 1]])
