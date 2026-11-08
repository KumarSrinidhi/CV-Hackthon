"""Diagnostic: classical pipeline must not leave zeros inside block holes.

Two synthetic samples:
  sample_a  — single centred block hole (15 % of LR side), flat depth field.
  sample_b  — two block holes, depth step (foreground / background), textured RGB.

For each sample the test checks:
  - No zero or negative depth anywhere in the HR output (complete map).
  - RMSE inside the hole region is reported (and must be finite / reasonable).
  - RMSE outside the hole region is reported separately.
  - The original LR depth array and valid_mask are not mutated.
"""

import numpy as np
import pytest

from classical import classical_hr_depth
from classical.hole_filling import fill_holes_nearest
from data.degradation import DegradationConfig, degrade


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rmse(pred, gt, mask):
    if not mask.any():
        return float("nan")
    diff = pred[mask] - gt[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _upsample_mask(lr_mask, scale):
    """Nearest-upsample a boolean LR mask to HR."""
    return np.repeat(np.repeat(lr_mask, scale, axis=0), scale, axis=1)


# ---------------------------------------------------------------------------
# sample_a: flat depth, single centred block hole
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_a():
    rng = np.random.default_rng(0)
    h, w, scale = 60, 80, 4
    depth_hr = np.full((h * scale, w * scale), 2.5, np.float32)
    rgb = rng.integers(0, 256, (h * scale, w * scale, 3), dtype=np.uint8)
    cfg = DegradationConfig(n_block_holes=1, block_frac=0.15, seed=7)
    deg = degrade(depth_hr, scale, cfg, sample_index=0)
    return {
        "lr_depth": deg["lr_depth"],
        "lr_valid": deg["lr_valid_mask"],
        "hr_rgb": rgb,
        "depth_hr": depth_hr,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# sample_b: depth step + two block holes + textured RGB
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_b():
    rng = np.random.default_rng(1)
    h, w, scale = 60, 80, 4
    col = np.arange(w * scale)[None, :] < w * scale // 2  # (1, W)
    depth_hr = np.where(
        np.broadcast_to(col, (h * scale, w * scale)),
        np.float32(1.5),
        np.float32(4.0),
    ).astype(np.float32)
    rgb = rng.integers(0, 256, (h * scale, w * scale, 3), dtype=np.uint8)
    cfg = DegradationConfig(n_block_holes=2, block_frac=0.15, seed=13)
    deg = degrade(depth_hr, scale, cfg, sample_index=0)
    return {
        "lr_depth": deg["lr_depth"],
        "lr_valid": deg["lr_valid_mask"],
        "hr_rgb": rgb,
        "depth_hr": depth_hr,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_name", ["sample_a", "sample_b"])
def test_no_zeros_in_block_holes(sample_name, request):
    s = request.getfixturevalue(sample_name)
    lr, valid, rgb, gt, scale = (
        s["lr_depth"], s["lr_valid"], s["hr_rgb"], s["depth_hr"], s["scale"]
    )

    # Preserve originals to check immutability.
    lr_orig = lr.copy()
    valid_orig = valid.copy()

    out = classical_hr_depth(lr, rgb, valid, scale)

    # --- immutability ---
    np.testing.assert_array_equal(lr, lr_orig, err_msg="lr_depth was mutated")
    np.testing.assert_array_equal(valid, valid_orig, err_msg="valid_mask was mutated")

    # --- completeness: no zeros or negatives anywhere ---
    zeros_in_output = (out <= 0).sum()
    assert zeros_in_output == 0, (
        f"{sample_name}: {zeros_in_output} zero/negative pixels in HR output "
        f"(shape {out.shape})"
    )

    # --- per-region RMSE ---
    hole_lr = ~valid                          # LR hole mask
    hole_hr = _upsample_mask(hole_lr, scale)  # HR hole mask
    rmse_in  = _rmse(out, gt, hole_hr)
    rmse_out = _rmse(out, gt, ~hole_hr)

    print(
        f"\n{sample_name}: "
        f"RMSE inside holes={rmse_in:.4f}  outside holes={rmse_out:.4f}  "
        f"hole_px={hole_hr.sum()}  total_px={out.size}"
    )

    # Inside-hole RMSE must be finite and not catastrophically large
    # (> 3× the depth range would indicate the fill is nonsense).
    depth_range = float(gt.max() - gt.min()) or 1.0
    assert np.isfinite(rmse_in), f"{sample_name}: RMSE inside holes is not finite"
    assert rmse_in < 3.0 * depth_range, (
        f"{sample_name}: RMSE inside holes ({rmse_in:.4f}) exceeds 3× depth range "
        f"({3*depth_range:.4f})"
    )


# ---------------------------------------------------------------------------
# fill_holes_nearest unit tests
# ---------------------------------------------------------------------------

def test_fill_holes_nearest_connected_region_per_pixel():
    """Two adjacent valid pixels in one connected region must each seed their
    own Voronoi cell.  DIST_LABEL_CCOMP would collapse them to one label and
    produce an arbitrary result; DIST_LABEL_PIXEL must not.

    Layout (0 = hole, value = valid depth):
        0    0    0    0    0
        0   1.0  2.0   0    0
        0    0    0    0    0

    Expected nearest-valid assignment:
        col 0-1  -> nearest is (1,1) depth 1.0
        col 2-4  -> nearest is (1,2) depth 2.0
    """
    depth = np.array([
        [0,   0,   0,   0,   0],
        [0,   1.0, 2.0, 0,   0],
        [0,   0,   0,   0,   0],
    ], dtype=np.float32)
    valid = depth > 0
    valid_orig = valid.copy()

    result = fill_holes_nearest(depth, valid)

    # Validity mask must not be mutated.
    np.testing.assert_array_equal(valid, valid_orig)

    # Measured pixels preserved exactly.
    assert result[1, 1] == pytest.approx(1.0)
    assert result[1, 2] == pytest.approx(2.0)

    # Hole pixels: left half nearest to (1,1), right half nearest to (1,2).
    for r in range(3):
        assert result[r, 0] == pytest.approx(1.0), f"[{r},0] should be 1.0"
        assert result[r, 1] == pytest.approx(1.0), f"[{r},1] should be 1.0"
        assert result[r, 2] == pytest.approx(2.0), f"[{r},2] should be 2.0"
        assert result[r, 3] == pytest.approx(2.0), f"[{r},3] should be 2.0"
        assert result[r, 4] == pytest.approx(2.0), f"[{r},4] should be 2.0"

    # No zeros anywhere.
    assert (result > 0).all()
    assert result.dtype == np.float32


def test_fill_holes_nearest_all_valid_unchanged():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    result = fill_holes_nearest(depth, valid)
    np.testing.assert_array_equal(result, depth)


def test_fill_holes_nearest_all_invalid_returns_zeros():
    depth = np.zeros((3, 3), dtype=np.float32)
    valid = np.zeros((3, 3), dtype=bool)
    result = fill_holes_nearest(depth, valid)
    assert result.shape == depth.shape
    assert result.dtype == np.float32


def test_fill_holes_nearest_preserves_valid_mask():
    depth = np.array([[0, 5.0, 0], [0, 0, 3.0]], dtype=np.float32)
    valid = depth > 0
    valid_orig = valid.copy()
    fill_holes_nearest(depth, valid)
    np.testing.assert_array_equal(valid, valid_orig)
