# Team Owner: Rakshitha
# Workstream: Evaluation
# Module: eval/evaluate.py
# Description: Evaluation scripts across held-out test/val splits with strict masked comparison across all branches.
"""Evaluate every available method on the SAME samples, scale, unit and mask.

    python -m eval.evaluate --dataset nyu --split val --scale 4 --methods bicubic deep --max-samples 20
    python -m eval.evaluate --dataset nyu --split test --scale 4 --final          # final numbers only

Methods
-------
nearest / bilinear / bicubic : internal reference baselines (masked-safe, cv2)
classical                    : classical.pipeline.classical_hr_depth   (Tanush)
deep                         : deep.inference.deep_hr_depth            (Rakshitha)
hybrid                       : hybrid.pipeline.hybrid_hr_depth         (Rishita)

A branch that is not implemented yet (stub) or whose checkpoint is missing is reported as
"not evaluated" with the reason; it never aborts the run and is never silently dropped.
The 'test' split is locked behind --final so tuning cannot touch it by accident.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from data.dataset import DepthSRDataset, get_dataset
from data.degradation import DegradationConfig
from data.preprocessing import fill_invalid_nearest

from .metrics import DEFAULT_BORDER, compute_all_metrics

Method = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
METRIC_KEYS = ["rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_precision", "edge_recall", "edge_f1"]
BASELINES = ("nearest", "bilinear", "bicubic")
BRANCHES = {
    "classical": ("classical.pipeline", "classical_hr_depth"),
    "deep": ("deep.inference", "deep_hr_depth"),
    "hybrid": ("hybrid.pipeline", "hybrid_hr_depth"),
}
_CV2 = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


class MethodUnavailable(RuntimeError):
    pass


def _make_baseline(kind: str) -> Method:
    def fn(lr_depth, hr_rgb, valid_mask, scale=4):
        mask = np.ones_like(lr_depth, bool) if valid_mask is None else valid_mask.astype(bool)
        filled = fill_invalid_nearest(lr_depth, mask)  # working copy; inputs untouched
        h, w = filled.shape
        out = cv2.resize(filled, (w * scale, h * scale), interpolation=_CV2[kind])
        return np.maximum(out, 0).astype(np.float32)
    return fn


def load_method(name: str) -> Method:
    if name in BASELINES:
        return _make_baseline(name)
    if name == "hybrid_fixed02":
        from hybrid.pipeline import hybrid_hr_depth
        return lambda lr, rgb, valid, scale=4: hybrid_hr_depth(
            lr, rgb, valid, scale, classical_weight=0.02,
            use_refinement=False, use_consistency=False,
        )
    if name == "hybrid_adaptive":
        from hybrid.pipeline import hybrid_hr_depth
        return lambda lr, rgb, valid, scale=4: hybrid_hr_depth(
            lr, rgb, valid, scale, use_refinement=False, use_consistency=False,
        )
    if name == "hybrid_selected_clean":
        from hybrid.pipeline import hybrid_hr_depth
        return lambda lr, rgb, valid, scale=4: hybrid_hr_depth(
            lr, rgb, valid, scale, classical_weight=0.02,
            use_refinement=False, use_consistency=True,
            consistency_mode="bicubic", operator_exact=True,
        )
    if name not in BRANCHES:
        raise MethodUnavailable(f"unknown method '{name}'")
    mod_name, fn_name = BRANCHES[name]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001
        raise MethodUnavailable(f"cannot import {mod_name}: {type(e).__name__}: {e}") from e
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise MethodUnavailable(f"{mod_name}.{fn_name} is not implemented yet")
    return fn


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def evaluate_methods(
    dataset: DepthSRDataset,
    method_names: Sequence[str],
    max_samples: Optional[int] = None,
    border: int = DEFAULT_BORDER,
    rel_thresh: float = 0.03,
    methods: Optional[Dict[str, Method]] = None,
    verbose: bool = True,
) -> dict:
    """Return {'per_image': [...], 'summary': {...}, 'not_evaluated': {...}, 'meta': {...}}.
    ``methods`` lets callers inject callables (tests, mocks)."""
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    callables: Dict[str, Method] = {}
    not_eval: Dict[str, str] = {}
    for name in method_names:
        if methods and name in methods:
            callables[name] = methods[name]
            continue
        try:
            callables[name] = load_method(name)
        except MethodUnavailable as e:
            not_eval[name] = str(e)

    rows: List[dict] = []
    failures: Dict[str, List[str]] = {k: [] for k in callables}
    for i in range(n):
        s = dataset[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        for name, fn in list(callables.items()):
            try:
                _sync()
                t0 = time.perf_counter()
                pred = fn(s["lr_depth"].copy(), s["hr_rgb"].copy(), s["lr_valid_mask"].copy(), s["scale"])
                _sync()
                dt = (time.perf_counter() - t0) * 1000.0
                pred = np.asarray(pred, dtype=np.float32)
                if pred.shape != gt.shape:
                    raise ValueError(f"output shape {pred.shape} != expected {gt.shape}")
                if not np.isfinite(pred).all():
                    raise ValueError("output contains NaN/Inf")
                m = compute_all_metrics(pred, gt, s["data_range"], valid, border, rel_thresh)
                rows.append({"sample": s["name"], "method": name, "latency_ms": dt, **m})
            except (NotImplementedError, TypeError) as e:
                not_eval[name] = f"{type(e).__name__}: {e}"
                callables.pop(name, None)
            except Exception as e:  # noqa: BLE001
                failures.setdefault(name, []).append(f"{s['name']}: {type(e).__name__}: {e}")
        if verbose and (i + 1) % 10 == 0:
            print(f"  evaluated {i + 1}/{n}")

    summary: Dict[str, dict] = {}
    for name in callables:
        mr = [r for r in rows if r["method"] == name]
        if not mr:
            not_eval[name] = f"all samples failed, e.g. {failures[name][:1]}"
            continue
        entry = {"n_ok": len(mr), "n_failed": len(failures.get(name, []))}
        for k in METRIC_KEYS + ["latency_ms"]:
            v = np.array([r[k] for r in mr], dtype=np.float64)
            fin = v[np.isfinite(v)]
            entry[k] = float(fin.mean()) if fin.size else float("nan")
            entry[k + "_std"] = float(fin.std()) if fin.size else float("nan")
        entry["failures"] = failures.get(name, [])[:5]
        summary[name] = entry
    first = dataset[0]
    return {
        "per_image": rows,
        "summary": summary,
        "not_evaluated": not_eval,
        "meta": {
            "dataset": dataset.name, "scale": dataset.scale, "n_samples": n, "unit": first["unit"],
            "data_range": first["data_range"], "border": border, "boundary_rel_thresh": rel_thresh,
            "degradation": dataset.degradation.as_dict(), "split": getattr(dataset, "split", "test"),
        },
    }


def save_results(res: dict, out_dir: str, tag: str) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = out / f"{tag}_per_image.csv"
    if res["per_image"]:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["per_image"][0].keys()))
            w.writeheader()
            w.writerows(res["per_image"])
        paths["per_image"] = p
    p = out / f"{tag}_summary.json"
    p.write_text(json.dumps({k: res[k] for k in ("summary", "not_evaluated", "meta")}, indent=2))
    paths["summary_json"] = p
    p = out / f"{tag}_summary.md"
    p.write_text(summary_markdown(res))
    paths["summary_md"] = p
    return paths


def summary_markdown(res: dict) -> str:
    m = res["meta"]
    unit = {"m": "m", "8bit": "grey levels", "normalized": "normalised"}.get(m["unit"], m["unit"])
    lines = [
        f"### {m['dataset']} / {m['split']} / x{m['scale']}  (n={m['n_samples']}, unit={unit}, "
        f"data_range={m['data_range']}, border={m['border']}px, degradation={m['degradation']})",
        "",
        "| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in res["summary"].items():
        lines.append(
            f"| {name} | {e['rmse']:.4f} | {e['mae']:.4f} | {e['psnr']:.2f} | {e['ssim']:.4f} | "
            f"{e['boundary_rmse']:.4f} | {e['edge_f1']:.3f} | {e['latency_ms']:.1f} |"
        )
    for name, why in res["not_evaluated"].items():
        lines.append(f"| {name} | not evaluated | | | | | | {why} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nyu")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--root", default=None)
    ap.add_argument("--methods", nargs="+", default=list(BASELINES) + ["classical", "deep", "hybrid"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--hole-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--final", action="store_true", help="required for split test/full (no tuning allowed on them)")
    a = ap.parse_args(argv)

    if a.split in ("test", "full") and not a.final:
        ap.error("split 'test'/'full' is reserved for final reporting: tune on --split val, then add --final.")
    deg = DegradationConfig(noise_std=a.noise_std, hole_ratio=a.hole_ratio, seed=a.seed)
    ds = get_dataset(a.dataset, a.split, a.scale, a.root, deg)
    print(f"{ds.name}/{a.split} x{a.scale}: {len(ds)} samples")
    res = evaluate_methods(ds, a.methods, a.max_samples)
    tag = f"{ds.name}_{a.split}_x{a.scale}" + ("_noisy" if a.noise_std else "") + ("_holes" if a.hole_ratio else "")
    paths = save_results(res, a.out, tag)
    print(summary_markdown(res))
    print("saved:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
