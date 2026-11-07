# Team Owner: Kumar
# Workstream: Application + Visualization
# Module: app/visualization.py
# Description: Depth colormap rendering, error map visualization, and side-by-side comparison figure export.

from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np


def colorize_depth(
    depth: np.ndarray,
    cmap: str = "turbo",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """
    Render a float depth map into a colorized RGB image using matplotlib colormaps.
    """
    depth = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0)

    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if min_val is None:
        min_val = float(np.percentile(depth[valid], 2))
    if max_val is None:
        max_val = float(np.percentile(depth[valid], 98))

    denom = max(max_val - min_val, 1e-6)
    normalized = np.clip((depth - min_val) / denom, 0.0, 1.0)

    # Invert so closer objects are brighter/distinct
    colormap = plt.get_cmap(cmap)
    color_rgba = colormap(normalized)
    color_rgb = (color_rgba[:, :, :3] * 255.0).astype(np.uint8)

    # Invalid pixels rendered as dark gray
    color_rgb[~valid] = [30, 30, 30]
    return color_rgb


def create_side_by_side_comparison(
    rgb: np.ndarray,
    lr_depth: np.ndarray,
    d_bicubic: np.ndarray,
    d_classical: np.ndarray,
    d_deep: np.ndarray,
    d_hybrid: np.ndarray,
    gt_depth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a comprehensive side-by-side multi-panel visualization.
    """
    panels = [
        ("RGB Guidance", rgb),
        ("LR Depth (Input)", colorize_depth(cv2.resize(lr_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST))),
        ("Bicubic Baseline", colorize_depth(d_bicubic)),
        ("Classical (JBU)", colorize_depth(d_classical)),
        ("Deep Branch", colorize_depth(d_deep)),
        ("Hybrid (Optimal)", colorize_depth(d_hybrid)),
    ]
    if gt_depth is not None:
        panels.append(("Ground Truth", colorize_depth(gt_depth)))

    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 3.5), squeeze=False)

    for i, (title, img) in enumerate(panels):
        axes[0, i].imshow(img)
        axes[0, i].set_title(title, fontsize=11, fontweight="bold")
        axes[0, i].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig
