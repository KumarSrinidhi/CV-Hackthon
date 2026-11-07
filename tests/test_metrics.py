import numpy as np
import pytest

from ps4.eval.metrics import evaluate, mae, psnr, rmse


def make_gt(h=32, w=32):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return 1.0 + 2.0 * (xx / w)


def test_identical_maps_zero_error():
    gt = make_gt()
    assert rmse(gt, gt) == 0.0
    assert mae(gt, gt) == 0.0
    assert psnr(gt, gt) == float("inf")


def test_degraded_map_scores_worse():
    gt = make_gt()
    small_err = gt + 0.01
    big_err = gt + 0.5
    assert rmse(big_err, gt) > rmse(small_err, gt)
    assert mae(big_err, gt) > mae(small_err, gt)
    assert psnr(big_err, gt) < psnr(small_err, gt)


def test_invalid_gt_masked_out():
    gt = make_gt()
    gt[0, 0] = 0.0
    gt[5, 5] = np.nan
    pred = gt + 0.0
    assert rmse(pred, gt) == 0.0
    with pytest.raises(ValueError, match="no valid pixels"):
        rmse(gt, np.zeros_like(gt))


def test_evaluate_keys():
    gt = make_gt()
    out = evaluate(gt + 0.1, gt)
    assert set(out) == {"RMSE (m)", "MAE (m)", "PSNR (dB)"}
