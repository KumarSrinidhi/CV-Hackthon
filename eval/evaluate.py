# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/evaluate.py
# Description: Method comparison harness evaluating Bicubic, Classical, Deep, and Hybrid approaches.

import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.pipeline import hybrid_hr_depth
from eval.metrics import compute_all_metrics
from data.dataset import RGBDDataset


def evaluate_sample(
    lr_depth: np.ndarray,
    hr_rgb: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    scale: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run all 4 tracks (Bicubic, Classical, Deep, Hybrid) on a single RGB-D sample and compute metrics.
    """
    results = {}

    # 1. Bicubic Baseline
    t0 = time.perf_counter()
    d_bicubic = classical_hr_depth(lr_depth, hr_rgb, scale=scale, method="bicubic")
    lat_bicubic = (time.perf_counter() - t0) * 1000.0
    m_bicubic = compute_all_metrics(d_bicubic, gt_depth, valid_mask=valid_mask)
    m_bicubic["latency_ms"] = lat_bicubic
    results["Bicubic"] = m_bicubic

    # 2. Classical (JBU)
    t0 = time.perf_counter()
    d_classical = classical_hr_depth(lr_depth, hr_rgb, valid_mask=valid_mask, scale=scale, method="jbu")
    lat_classical = (time.perf_counter() - t0) * 1000.0
    m_classical = compute_all_metrics(d_classical, gt_depth, valid_mask=valid_mask)
    m_classical["latency_ms"] = lat_classical
    results["Classical"] = m_classical

    # 3. Deep Learning
    t0 = time.perf_counter()
    d_deep = deep_hr_depth(lr_depth, hr_rgb, valid_mask=valid_mask, scale=scale, device=device)
    lat_deep = (time.perf_counter() - t0) * 1000.0
    m_deep = compute_all_metrics(d_deep, gt_depth, valid_mask=valid_mask)
    m_deep["latency_ms"] = lat_deep
    results["Deep"] = m_deep

    # 4. Hybrid (Optimal: alpha=0.02 + Consistency)
    t0 = time.perf_counter()
    d_hybrid = hybrid_hr_depth(
        lr_depth,
        hr_rgb,
        valid_mask=valid_mask,
        scale=scale,
        alpha=0.02,
        deep_depth=d_deep,
        classical_depth=d_classical,
        method="fixed",
        use_consistency=True,
    )
    lat_hybrid = (time.perf_counter() - t0) * 1000.0 + lat_deep + lat_classical
    m_hybrid = compute_all_metrics(d_hybrid, gt_depth, valid_mask=valid_mask)
    m_hybrid["latency_ms"] = lat_hybrid
    results["Hybrid"] = m_hybrid

    return results


def run_evaluation_benchmark(
    dataset: Optional[RGBDDataset] = None,
    num_samples: int = 5,
    scale: int = 4,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run evaluation across multiple samples and produce the summary comparison table.
    """
    if dataset is None:
        dataset = RGBDDataset(num_synthetic_samples=num_samples, scale=scale)

    all_results = {"Bicubic": [], "Classical": [], "Deep": [], "Hybrid": []}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        sample_res = evaluate_sample(
            sample["lr_depth"],
            sample["rgb"],
            sample["gt_depth"],
            valid_mask=sample.get("valid_mask"),
            scale=scale,
            device=device,
        )
        for method, metrics in sample_res.items():
            all_results[method].append(metrics)

    summary_rows = []
    for method, metric_list in all_results.items():
        df_m = pd.DataFrame(metric_list)
        mean_s = df_m.mean()
        summary_rows.append({
            "Method": method,
            "RMSE": f"{mean_s['rmse']:.4f}",
            "MAE": f"{mean_s['mae']:.4f}",
            "PSNR (dB)": f"{mean_s['psnr']:.2f}",
            "SSIM": f"{mean_s['ssim']:.4f}",
            "Boundary RMSE": f"{mean_s['boundary_rmse']:.4f}",
            "Edge F1": f"{mean_s['edge_f1']:.3f}",
            "Latency": f"{mean_s['latency_ms']:.1f} ms",
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


if __name__ == "__main__":
    df = run_evaluation_benchmark(num_samples=3, scale=4)
    print("\n=== Benchmark Comparison ===")
    print(df.to_markdown(index=False))
