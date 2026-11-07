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
def plot_scanline_profile(
    bicubic: np.ndarray,
    classical: np.ndarray,
    deep: np.ndarray,
    hybrid: np.ndarray,
    gt: Optional[np.ndarray] = None,
    row_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a 1D cross-section depth profile plot along a horizontal scanline
    to visually demonstrate edge sharpness, step response, and texture stability.
    """
    h, w = hybrid.shape[:2]
    if row_idx is None:
        row_idx = h // 2
    row_idx = max(0, min(h - 1, int(row_idx)))

    x = np.arange(w)
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if gt is not None:
        ax.plot(x, gt[row_idx], label="Ground Truth", color="#ffffff", linewidth=2.2, linestyle="--", alpha=0.9)
    ax.plot(x, bicubic[row_idx], label="Bicubic Baseline (Blurry Edge)", color="#ef4444", linewidth=1.8, alpha=0.85)
    ax.plot(x, classical[row_idx], label="Classical JBU", color="#f59e0b", linewidth=1.8, linestyle=":", alpha=0.85)
    ax.plot(x, deep[row_idx], label="Deep (EdgeGuidedSRNet)", color="#38bdf8", linewidth=1.8, alpha=0.85)
    ax.plot(x, hybrid[row_idx], label="Hybrid (Optimal Sharp Step)", color="#10b981", linewidth=2.5, alpha=0.95)

    ax.set_title(f"1D Depth Discontinuity Profile (Horizontal Scanline y = {row_idx})", fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax.set_xlabel("Horizontal Pixel Coordinate (X)", fontsize=9, color="#cbd5e1")
    ax.set_ylabel("Depth (meters)", fontsize=9, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        img = cv2.imread(save_path)[:, :, ::-1]
        return img

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img
