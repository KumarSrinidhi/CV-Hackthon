# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/benchmark.py
# Description: Benchmarking suite to measure latency, peak GPU memory usage, and generate comparison tables and plots.
"""Final comparison tables + plots across scales.

    python -m eval.benchmark --dataset nyu --split val  --scales 4 8 --methods bicubic deep --max-samples 30
    python -m eval.benchmark --dataset nyu --split test --scales 4 8 --final                # final numbers

For each scale it (1) runs eval.evaluate on identical samples, (2) profiles latency
(warm-up + repeated timed runs, CUDA-synchronised) and peak GPU memory on one fixed sample,
(3) checks repeatability (two runs, max abs difference), (4) writes
comparison_table.{csv,md} and PNG plots (needs matplotlib; skipped with a notice otherwise).
Scales a method does not support are reported as 'not evaluated', never as compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from data.dataset import get_dataset
from data.degradation import DegradationConfig

from .evaluate import BASELINES, Method, evaluate_methods, load_method, MethodUnavailable, save_results


def _torch_cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _process_rss_mb():
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / 2**20)
    except ImportError:
        if os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return float(counters.peak_working_set_size / 2**20)
        return None
    except Exception:
        return None


def profile_method(fn: Method, sample: dict, warmup: int = 2, runs: int = 10) -> dict:
    """Latency (ms) and peak GPU memory (MB, None without CUDA) of ``fn`` on one sample,
    plus repeatability = max |run1 - run2|."""
    args = (sample["lr_depth"], sample["hr_rgb"], sample["lr_valid_mask"], sample["scale"])
    torch = _torch_cuda()
    ram_before = _process_rss_mb()
    for _ in range(warmup):
        fn(*[a.copy() if hasattr(a, "copy") else a for a in args])
    if torch:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    outs = []
    for _ in range(max(2, runs)):
        t0 = time.perf_counter()
        o = fn(*[a.copy() if hasattr(a, "copy") else a for a in args])
        if torch:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        outs.append(np.asarray(o, dtype=np.float32))
    t = np.asarray(times)
    return {
        "latency_ms_mean": float(t.mean()),
        "latency_ms_median": float(np.median(t)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "peak_gpu_alloc_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch else None,
        "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved() / 2**20) if torch else None,
        "peak_ram_mb": _process_rss_mb() if ram_before is not None else None,
        "repeat_max_abs_diff": float(np.abs(outs[0] - outs[1]).max()),
        "image_hw": tuple(sample["depth_original"].shape),
    }


def run_benchmark(
    dataset_name: str, split: str, scales: Sequence[int], method_names: Sequence[str],
    out_dir: str = "results", max_samples: Optional[int] = None, root: Optional[str] = None,
    degradation: Optional[DegradationConfig] = None, runs: int = 10,
) -> List[dict]:
    rows: List[dict] = []
    for scale in scales:
        try:
            ds = get_dataset(dataset_name, split, scale, root, degradation)
        except (NotImplementedError, FileNotFoundError, ValueError) as e:
            rows.append({"scale": scale, "method": "*", "status": f"not evaluated: {e}"})
            continue
        print(f"== {ds.name}/{split} x{scale}: {len(ds)} samples")
        res = evaluate_methods(ds, method_names, max_samples)
        save_results(res, out_dir, f"{ds.name}_{split}_x{scale}")
        prof_sample = ds[0]
        for name in method_names:
            if name in res["not_evaluated"]:
                rows.append({"dataset": ds.name, "split": split, "scale": scale, "method": name,
                             "status": f"not evaluated: {res['not_evaluated'][name]}"})
                continue
            e = res["summary"][name]
            row = {"dataset": ds.name, "split": split, "scale": scale, "method": name, "status": "ok",
                   "n": e["n_ok"], **{k: e[k] for k in ("rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_f1")}}
            try:
                row.update(profile_method(load_method(name), prof_sample, runs=runs))
            except Exception as ex:  # noqa: BLE001
                row["status"] = f"ok (profiling failed: {type(ex).__name__}: {ex})"
            rows.append(row)
    write_tables(rows, out_dir)
    make_plots(rows, out_dir)
    return rows


_COLS = ["dataset", "split", "scale", "method", "status", "n", "rmse", "mae", "psnr", "ssim", "boundary_rmse",
         "edge_f1", "latency_ms_median", "latency_ms_p95", "peak_ram_mb", "peak_gpu_alloc_mb", "peak_gpu_reserved_mb",
         "repeat_max_abs_diff"]


def write_tables(rows: List[dict], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if v is None:
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(_COLS) + " |", "|" + "---|" * len(_COLS)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in _COLS) + " |" for r in rows]
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n")
    (out / "comparison_table.json").write_text(json.dumps(rows, indent=2, default=str))


def make_plots(rows: List[dict], out_dir: str) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("matplotlib not installed: plots skipped (pip install matplotlib).")
        return []
    ok = [r for r in rows if r.get("status", "").startswith("ok") and "rmse" in r]
    if not ok:
        return []
    out = Path(out_dir)
    methods = sorted({r["method"] for r in ok})
    scales = sorted({r["scale"] for r in ok})
    paths = []

    def by_scale(metric, fname, ylabel):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in methods:
            pts = sorted((r["scale"], r[metric]) for r in ok if r["method"] == m and r.get(metric) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=m)
        ax.set_xlabel("upsampling factor")
        ax.set_ylabel(ylabel)
        ax.set_xticks(scales)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    by_scale("rmse", "rmse_vs_scale.png", "RMSE (dataset unit)")
    by_scale("boundary_rmse", "boundary_rmse_vs_scale.png", "boundary RMSE (dataset unit)")
    by_scale("latency_ms_median", "latency_vs_scale.png", "median latency (ms)")

    mem = [(r["method"], r["scale"], r["peak_gpu_alloc_mb"]) for r in ok if r.get("peak_gpu_alloc_mb")]
    if mem:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{m} x{s}" for m, s, _ in mem]
        ax.bar(labels, [v for *_, v in mem])
        ax.set_ylabel("peak GPU memory allocated (MB)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        p = out / "peak_gpu_memory.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nyu")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scales", type=int, nargs="+", default=[4])
    ap.add_argument("--methods", nargs="+", default=list(BASELINES) + ["classical", "deep", "hybrid"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--hole-ratio", type=float, default=0.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    if a.split in ("test", "full") and not a.final:
        ap.error("split 'test'/'full' is reserved for final reporting: tune on --split val, then add --final.")
    deg = DegradationConfig(noise_std=a.noise_std, hole_ratio=a.hole_ratio)
    rows = run_benchmark(a.dataset, a.split, a.scales, a.methods, a.out, a.max_samples, a.root, deg, a.runs)
    print(f"wrote {a.out}/comparison_table.(csv|md|json) with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
