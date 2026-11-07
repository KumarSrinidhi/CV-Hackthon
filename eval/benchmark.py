# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/benchmark.py
# Description: Scalability benchmark across upsampling factors (2x, 4x, 8x), latency, and peak memory profiling.

import os
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.pipeline import hybrid_hr_depth
from eval.metrics import compute_all_metrics
from data.dataset import generate_synthetic_rgbd_scene
from data.degradation import create_lr_depth


def profile_memory_and_latency(func, *args, **kwargs):
    """Profile inference latency and peak CUDA memory (if available)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    else:
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_mem_mb = 0.0

    return out, latency_ms, peak_mem_mb


def run_scale_benchmarks(scales: List[int] = [2, 4, 8], num_runs: int = 3) -> pd.DataFrame:
    """Benchmark performance across different scale factors."""
    rows = []
    rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=42)

    for scale in scales:
        lr_depth, valid_mask = create_lr_depth(gt_depth, scale=scale)

        # 1. Classical
        d_c, lat_c, _ = profile_memory_and_latency(
            classical_hr_depth, lr_depth, rgb, scale=scale, method="jbu"
        )
        m_c = compute_all_metrics(d_c, gt_depth)
        rows.append({
            "Scale": f"{scale}x",
            "Method": "Classical (JBU)",
            "RMSE": f"{m_c['rmse']:.4f}",
            "PSNR (dB)": f"{m_c['psnr']:.2f}",
            "Edge F1": f"{m_c['edge_f1']:.3f}",
            "Latency": f"{lat_c:.1f} ms",
        })

        # 2. Deep
        d_d, lat_d, mem_d = profile_memory_and_latency(
            deep_hr_depth, lr_depth, rgb, scale=scale
        )
        m_d = compute_all_metrics(d_d, gt_depth)
        rows.append({
            "Scale": f"{scale}x",
            "Method": "Deep",
            "RMSE": f"{m_d['rmse']:.4f}",
            "PSNR (dB)": f"{m_d['psnr']:.2f}",
            "Edge F1": f"{m_d['edge_f1']:.3f}",
            "Latency": f"{lat_d:.1f} ms",
        })

        # 3. Hybrid
        d_h, lat_h, _ = profile_memory_and_latency(
            hybrid_hr_depth, lr_depth, rgb, scale=scale, deep_depth=d_d, classical_depth=d_c
        )
        m_h = compute_all_metrics(d_h, gt_depth)
        rows.append({
            "Scale": f"{scale}x",
            "Method": "Hybrid (Optimal)",
            "RMSE": f"{m_h['rmse']:.4f}",
            "PSNR (dB)": f"{m_h['psnr']:.2f}",
            "Edge F1": f"{m_h['edge_f1']:.3f}",
            "Latency": f"{(lat_h + lat_d + lat_c):.1f} ms",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_scale = run_scale_benchmarks()
    print("\n=== Scale Factor Benchmark (2x, 4x, 8x) ===")
    print(df_scale.to_markdown(index=False))
