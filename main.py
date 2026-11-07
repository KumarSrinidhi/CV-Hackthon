# Team Owner: Kumar (Coordinates integration)
# Workstream: Application + Integration + Deployment
# Module: main.py
# Description: Application entry point connecting branch interfaces (deep, classical, hybrid) to UI and CLI workflows.

import argparse
import os
import sys

from app.ui import run_pipeline_demo
from eval.evaluate import run_evaluation_benchmark
from eval.benchmark import run_scale_benchmarks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge-Preserving Hybrid Depth Map Upsampling and Refinement (P4)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "benchmark", "scale_benchmark"],
        help="Execution mode: demo (single scene demo & visualization), benchmark (dataset evaluation), scale_benchmark (2x, 4x, 8x profiling)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Super-resolution upsampling factor (default: 4)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.02,
        help="Classical blending weight in hybrid fusion (default: 0.02)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate in benchmark mode (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where output depth maps and comparison figures are saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("   Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Super-Resolution")
    print("   Hybrid Architecture: Classical CV (JBU) + Deep Guidance + Consistency")
    print("=" * 70)

    if args.mode == "demo":
        summary_df = run_pipeline_demo(
            scale=args.scale,
            alpha=args.alpha,
            output_dir=args.output_dir,
        )
        print("\n=== Single-Scene Demonstration Results ===")
        print(summary_df.to_markdown(index=False))

    elif args.mode == "benchmark":
        print(f"\nRunning benchmark across {args.num_samples} samples at {args.scale}x scale...")
        df_bench = run_evaluation_benchmark(num_samples=args.num_samples, scale=args.scale)
        print("\n=== Multi-Track Benchmark Summary ===")
        print(df_bench.to_markdown(index=False))

    elif args.mode == "scale_benchmark":
        print("\nRunning multi-scale profiling across 2x, 4x, 8x...")
        df_scale = run_scale_benchmarks()
        print("\n=== Multi-Scale Performance & Latency Profile ===")
        print(df_scale.to_markdown(index=False))

    print("\n[Done] Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
