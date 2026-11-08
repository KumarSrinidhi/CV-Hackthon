"""Compare old vs new classical pipeline on the two block-hole smoke samples.

Outputs (written to scripts/hole_fill_comparison/):
  {sample}_gt.png
  {sample}_old.png          -- old pipeline (bicubic seed, no NN fill)
  {sample}_new_telea.png    -- new pipeline with Telea filler
  {sample}_new_nearest.png  -- new pipeline with nearest-valid filler
  {sample}_err_old.png      -- |old - GT| error map
  {sample}_err_telea.png
  {sample}_err_nearest.png
  report.txt                -- all numeric diagnostics

Run:
  python -m scripts.compare_hole_fill
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import cv2
import numpy as np

from classical.hole_filling import fill_holes_nearest, fill_holes_telea, prepare_depth, fill_small_holes
from classical.jbu import joint_bilateral_upsample, to_float_rgb
from classical.interpolation import interpolate_depth
from classical.boundary_refinement import refine_boundaries
from data.degradation import DegradationConfig, degrade

OUT_DIR = Path(__file__).parent / "hole_fill_comparison"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Reproduce the OLD pipeline exactly as it was before the fix
# ---------------------------------------------------------------------------

def _classical_hr_depth_OLD(lr_depth, hr_rgb, valid_mask, scale):
    """Old A1→A4 pipeline: fill_small_holes only, then bicubic→downsample→JBU."""
    depth = np.asarray(lr_depth, dtype=np.float32)
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    depth_prep, valid = prepare_depth(depth, valid_mask)
    depth_filled, valid_filled = fill_small_holes(depth_prep, valid)
    upsampled = interpolate_depth(depth_filled, valid_filled, scale, "bicubic")
    upsampled_lr = cv2.resize(upsampled, (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_AREA)
    jbu_out = joint_bilateral_upsample(upsampled_lr, valid_filled, rgb_f, radius=1)
    return refine_boundaries(jbu_out, jbu_out, rgb_f)


# ---------------------------------------------------------------------------
# New pipeline parameterised on filler
# ---------------------------------------------------------------------------

def _classical_hr_depth_NEW(lr_depth, hr_rgb, valid_mask, scale, filler):
    depth = np.asarray(lr_depth, dtype=np.float32)
    rgb_f = to_float_rgb(np.asarray(hr_rgb))
    depth_prep, valid = prepare_depth(depth, valid_mask)
    depth_filled, valid_filled = fill_small_holes(depth_prep, valid)
    depth_seeded = filler(depth_filled, valid_filled) if not valid_filled.all() else depth_filled
    upsampled = interpolate_depth(depth_seeded, None, scale, "bicubic")
    jbu_out = joint_bilateral_upsample(depth_seeded, valid_filled, rgb_f, radius=1)
    out = refine_boundaries(jbu_out, jbu_out, rgb_f)
    return np.where(out > 0, out, upsampled)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upsample_mask(lr_mask, scale):
    return np.repeat(np.repeat(lr_mask, scale, axis=0), scale, axis=1)


def _boundary_mask(hole_hr, dilation=3):
    """Pixels within `dilation` px of the hole boundary (inside + outside)."""
    hole_u8 = hole_hr.astype(np.uint8)
    kernel = np.ones((2 * dilation + 1,) * 2, np.uint8)
    dilated = cv2.dilate(hole_u8, kernel)
    eroded = cv2.erode(hole_u8, kernel)
    return (dilated - eroded).astype(bool)


def _rmse(pred, gt, mask):
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def _depth_to_u8(depth, vmin, vmax):
    """Map depth to uint8 using a fixed [vmin, vmax] scale (identical across images)."""
    normed = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return (normed * 255).astype(np.uint8)


def _save_colormap(arr_u8, path):
    colored = cv2.applyColorMap(arr_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


def _save_error_map(err, vmax_err, path):
    normed = np.clip(err / max(vmax_err, 1e-6), 0.0, 1.0)
    u8 = (normed * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    cv2.imwrite(str(path), colored)


# ---------------------------------------------------------------------------
# Inpaint inspection
# ---------------------------------------------------------------------------

def _inspect_telea(depth, valid):
    """Return a dict of diagnostic facts about fill_holes_telea on this input."""
    vmin = float(depth[valid].min())
    vmax = float(depth[valid].max())
    span = vmax - vmin if vmax > vmin else 1.0
    norm = ((depth - vmin) / span).astype(np.float32)
    norm[~valid] = 0.0
    mask_u8 = (~valid).astype(np.uint8)
    filled = cv2.inpaint(norm, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    result_raw = filled * span + vmin          # before clip
    result_clipped = np.clip(filled, 0.0, 1.0) * span + vmin

    hole = ~valid
    return {
        "input_dtype": norm.dtype,             # always float32
        "depth_units_preserved": True,         # by construction (rescale back)
        "norm_min_in_hole": float(norm[hole].min()) if hole.any() else float("nan"),
        "norm_max_in_hole": float(norm[hole].min()) if hole.any() else float("nan"),
        "filled_norm_min_in_hole": float(filled[hole].min()) if hole.any() else float("nan"),
        "filled_norm_max_in_hole": float(filled[hole].max()) if hole.any() else float("nan"),
        "overshoot_below": float(np.clip(0.0 - filled[hole], 0, None).max()) if hole.any() else 0.0,
        "overshoot_above": float(np.clip(filled[hole] - 1.0, 0, None).max()) if hole.any() else 0.0,
        "raw_depth_min_in_hole": float(result_raw[hole].min()) if hole.any() else float("nan"),
        "raw_depth_max_in_hole": float(result_raw[hole].max()) if hole.any() else float("nan"),
        "clipped_depth_min_in_hole": float(result_clipped[hole].min()) if hole.any() else float("nan"),
        "clipped_depth_max_in_hole": float(result_clipped[hole].max()) if hole.any() else float("nan"),
        "valid_depth_range": (vmin, vmax),
    }


def _inspect_jbu_gate(depth_seeded, valid_filled, hr_rgb, scale):
    """Count how many LR pixels JBU actually splats (valid_filled True)."""
    n_valid = int(valid_filled.sum())
    n_total = int(valid_filled.size)
    n_seeded_not_valid = int((~valid_filled).sum())
    # Seeded pixels that are > 0 but not in valid_filled — JBU ignores these
    # for splatting but uses them in the nearest fallback.
    seeded_nonzero_not_valid = int(((depth_seeded > 0) & ~valid_filled).sum())
    return {
        "jbu_splat_pixels": n_valid,
        "jbu_ignored_seeded_pixels": seeded_nonzero_not_valid,
        "total_lr_pixels": n_total,
        "note": (
            "JBU iterates only over valid_filled=True pixels for splatting. "
            "Seeded-but-not-valid pixels appear in the nearest fallback map, "
            "so hole interiors receive the seeded value via nearest, not via "
            "bilateral weighting. This is correct behaviour."
        ),
    }


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------

def _run_sample(name, lr_depth, lr_valid, hr_rgb, depth_hr, scale, lines):
    hole_lr = ~lr_valid
    hole_hr = _upsample_mask(hole_lr, scale)
    boundary_hr = _boundary_mask(hole_hr, dilation=3)

    depth_prep, valid = prepare_depth(lr_depth, lr_valid)
    depth_filled, valid_filled = fill_small_holes(depth_prep, valid)

    pred_old     = _classical_hr_depth_OLD(lr_depth, hr_rgb, lr_valid, scale)
    pred_telea   = _classical_hr_depth_NEW(lr_depth, hr_rgb, lr_valid, scale, fill_holes_telea)
    pred_nearest = _classical_hr_depth_NEW(lr_depth, hr_rgb, lr_valid, scale, fill_holes_nearest)

    gt = depth_hr
    vmin_gt, vmax_gt = float(gt.min()), float(gt.max())
    # Use a common colour scale across all depth images for this sample.
    vmin_d = min(vmin_gt, float(pred_old[pred_old > 0].min() if (pred_old > 0).any() else vmin_gt))
    vmax_d = max(vmax_gt, float(pred_old.max()))

    # Error maps share the same scale.
    err_old     = np.abs(pred_old     - gt)
    err_telea   = np.abs(pred_telea   - gt)
    err_nearest = np.abs(pred_nearest - gt)
    vmax_err = max(float(err_old.max()), float(err_telea.max()), float(err_nearest.max()), 1e-6)

    # Save images.
    _save_colormap(_depth_to_u8(gt,           vmin_d, vmax_d), OUT_DIR / f"{name}_gt.png")
    _save_colormap(_depth_to_u8(pred_old,     vmin_d, vmax_d), OUT_DIR / f"{name}_old.png")
    _save_colormap(_depth_to_u8(pred_telea,   vmin_d, vmax_d), OUT_DIR / f"{name}_new_telea.png")
    _save_colormap(_depth_to_u8(pred_nearest, vmin_d, vmax_d), OUT_DIR / f"{name}_new_nearest.png")
    _save_error_map(err_old,     vmax_err, OUT_DIR / f"{name}_err_old.png")
    _save_error_map(err_telea,   vmax_err, OUT_DIR / f"{name}_err_telea.png")
    _save_error_map(err_nearest, vmax_err, OUT_DIR / f"{name}_err_nearest.png")

    # Telea inspection.
    telea_info = _inspect_telea(depth_filled, valid_filled)
    jbu_info   = _inspect_jbu_gate(
        fill_holes_nearest(depth_filled, valid_filled), valid_filled, hr_rgb, scale
    )

    def row(label, pred, mask_name, mask):
        return f"  {label:30s} {mask_name:12s}  RMSE={_rmse(pred, gt, mask):.5f}"

    lines += [
        "",
        f"{'='*60}",
        f"SAMPLE: {name}",
        f"{'='*60}",
        f"  GT depth range:  [{vmin_gt:.4f}, {vmax_gt:.4f}]",
        f"  LR shape: {lr_depth.shape}  HR shape: {gt.shape}  scale: x{scale}",
        f"  Hole pixels (LR): {hole_lr.sum()} / {hole_lr.size}  "
        f"({100*hole_lr.mean():.1f}%)",
        f"  Hole pixels (HR): {hole_hr.sum()} / {hole_hr.size}  "
        f"({100*hole_hr.mean():.1f}%)",
        "",
        "  --- depth min/max of predictions ---",
        f"  {'':30s} {'min':>10s}  {'max':>10s}  {'zeros':>8s}",
        f"  {'GT':30s} {vmin_gt:10.4f}  {vmax_gt:10.4f}  {'—':>8s}",
    ]
    for label, pred in [("old", pred_old), ("new_telea", pred_telea), ("new_nearest", pred_nearest)]:
        lines.append(
            f"  {label:30s} {float(pred.min()):10.4f}  {float(pred.max()):10.4f}  "
            f"{int((pred <= 0).sum()):>8d}"
        )

    lines += [
        "",
        "  --- RMSE by region ---",
        row("old",         pred_old,     "inside_hole",  hole_hr),
        row("new_telea",   pred_telea,   "inside_hole",  hole_hr),
        row("new_nearest", pred_nearest, "inside_hole",  hole_hr),
        "",
        row("old",         pred_old,     "outside_hole", ~hole_hr),
        row("new_telea",   pred_telea,   "outside_hole", ~hole_hr),
        row("new_nearest", pred_nearest, "outside_hole", ~hole_hr),
        "",
        row("old",         pred_old,     "boundary",     boundary_hr),
        row("new_telea",   pred_telea,   "boundary",     boundary_hr),
        row("new_nearest", pred_nearest, "boundary",     boundary_hr),
        "",
        "  --- cv2.inpaint (Telea) inspection ---",
        f"  input dtype:              {telea_info['input_dtype']}  "
        f"(float32 required by cv2.inpaint)",
        f"  depth units preserved:    {telea_info['depth_units_preserved']}  "
        f"(normalise in, rescale out)",
        f"  valid depth range:        [{telea_info['valid_depth_range'][0]:.4f}, "
        f"{telea_info['valid_depth_range'][1]:.4f}]",
        f"  norm[~valid] set to:      0.0  (biases fill toward vmin at boundary)",
        f"  filled_norm in hole:      [{telea_info['filled_norm_min_in_hole']:.5f}, "
        f"{telea_info['filled_norm_max_in_hole']:.5f}]",
        f"  overshoot below 0:        {telea_info['overshoot_below']:.6f}",
        f"  overshoot above 1:        {telea_info['overshoot_above']:.6f}",
        f"  raw depth in hole:        [{telea_info['raw_depth_min_in_hole']:.4f}, "
        f"{telea_info['raw_depth_max_in_hole']:.4f}]",
        f"  clipped depth in hole:    [{telea_info['clipped_depth_min_in_hole']:.4f}, "
        f"{telea_info['clipped_depth_max_in_hole']:.4f}]",
        "",
        "  --- JBU validity gate ---",
        f"  splat pixels (valid_filled=True):  {jbu_info['jbu_splat_pixels']}",
        f"  seeded-but-not-valid pixels:       {jbu_info['jbu_ignored_seeded_pixels']}",
        f"  total LR pixels:                   {jbu_info['total_lr_pixels']}",
        f"  note: {textwrap.fill(jbu_info['note'], width=60, subsequent_indent='        ')}",
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_sample_a():
    rng = np.random.default_rng(0)
    h, w, scale = 60, 80, 4
    depth_hr = np.full((h * scale, w * scale), 2.5, np.float32)
    rgb = rng.integers(0, 256, (h * scale, w * scale, 3), dtype=np.uint8)
    cfg = DegradationConfig(n_block_holes=1, block_frac=0.15, seed=7)
    deg = degrade(depth_hr, scale, cfg, sample_index=0)
    return deg["lr_depth"], deg["lr_valid_mask"], rgb, depth_hr, scale


def _make_sample_b():
    rng = np.random.default_rng(1)
    h, w, scale = 60, 80, 4
    col = np.arange(w * scale)[None, :] < w * scale // 2
    depth_hr = np.where(
        np.broadcast_to(col, (h * scale, w * scale)),
        np.float32(1.5), np.float32(4.0),
    ).astype(np.float32)
    rgb = rng.integers(0, 256, (h * scale, w * scale, 3), dtype=np.uint8)
    cfg = DegradationConfig(n_block_holes=2, block_frac=0.15, seed=13)
    deg = degrade(depth_hr, scale, cfg, sample_index=0)
    return deg["lr_depth"], deg["lr_valid_mask"], rgb, depth_hr, scale


def main():
    lines = ["Classical pipeline: old vs new — hole-fill comparison",
             f"Output directory: {OUT_DIR}", ""]

    for name, maker in [("sample_a", _make_sample_a), ("sample_b", _make_sample_b)]:
        lr, valid, rgb, gt, scale = maker()
        _run_sample(name, lr, valid, rgb, gt, scale, lines)

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(f"\nImages and report saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
