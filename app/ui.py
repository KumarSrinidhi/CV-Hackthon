# Team Owner: Kumar
# Workstream: Application + Integration + Deployment
# Module: app/ui.py
# Description: Interactive runner and CLI demonstration handler for depth super-resolution.

import os
import sys
from typing import Optional
import cv2
import numpy as np
import pandas as pd

from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.pipeline import hybrid_hr_depth
from eval.metrics import compute_all_metrics
from app.visualization import create_side_by_side_comparison, colorize_depth
from data.dataset import generate_synthetic_rgbd_scene
from data.degradation import create_lr_depth


def run_pipeline_demo(
    scale: int = 4,
    alpha: float = 0.02,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Run end-to-end demonstration comparing all 4 tracks, computing metrics,
    and exporting visualization assets.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[1/4] Generating synthetic evaluation scene at {scale}x scale...")
    rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=42)
    lr_depth, valid_mask = create_lr_depth(gt_depth, scale=scale)

    print("[2/4] Executing reconstruction tracks (Classical, Deep, Hybrid)...")
    d_bicubic = classical_hr_depth(lr_depth, rgb, scale=scale, method="bicubic")
    d_classical = classical_hr_depth(lr_depth, rgb, valid_mask=valid_mask, scale=scale, method="jbu")
    d_deep = deep_hr_depth(lr_depth, rgb, valid_mask=valid_mask, scale=scale)
    d_hybrid = hybrid_hr_depth(
        lr_depth,
        rgb,
        valid_mask=valid_mask,
        scale=scale,
        alpha=alpha,
        deep_depth=d_deep,
        classical_depth=d_classical,
        method="fixed",
        use_consistency=True,
    )

    print("[3/4] Computing evaluation metrics...")
    metrics_bicubic = compute_all_metrics(d_bicubic, gt_depth)
    metrics_classical = compute_all_metrics(d_classical, gt_depth)
    metrics_deep = compute_all_metrics(d_deep, gt_depth)
    metrics_hybrid = compute_all_metrics(d_hybrid, gt_depth)

    metrics_map = {
        "rmse": "RMSE",
        "mae": "MAE",
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "boundary_rmse": "Boundary RMSE",
        "edge_f1": "Edge F1",
    }
    rows = []
    for method_name, m_dict in [
        ("Bicubic (baseline)", metrics_bicubic),
        ("Classical (JBU)", metrics_classical),
        ("Deep (SGNet branch)", metrics_deep),
        (f"Hybrid (alpha={alpha})", metrics_hybrid),
    ]:
        row = {"Method": method_name}
        for k, v in m_dict.items():
            col_name = metrics_map.get(k, k.upper())
            row[col_name] = f"{v:.4f}" if "rmse" in k or "mae" in k or "ssim" in k else f"{v:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    print("[4/4] Exporting side-by-side visualization...")
    vis_path = os.path.join(output_dir, "depth_sr_comparison.png")
    create_side_by_side_comparison(
        rgb, lr_depth, d_bicubic, d_classical, d_deep, d_hybrid, gt_depth, save_path=vis_path
    )
    print(f"-> Visualization saved to: {vis_path}")

    # Save output numpy depths
    np.save(os.path.join(output_dir, "pred_hybrid_depth.npy"), d_hybrid)
    cv2.imwrite(os.path.join(output_dir, "pred_hybrid_depth_vis.png"), colorize_depth(d_hybrid))

    return df
