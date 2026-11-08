# Team Owner: Kumar
# Workstream: Hybrid
# Module: scripts/task_a_eval.py
# Description: Task A - cheap inference-time wins. Caches the expensive deep runs
# (orig ckpt, robust ckpt, flip TTA) once per regime, then evaluates all cheap
# post-processing variants offline from the cache.
"""Examples:
    python scripts/task_a_eval.py --cache --regimes clean holes --samples 150
    python scripts/task_a_eval.py --variants --regimes clean holes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from deep.inference import deep_hr_depth
from eval.metrics import compute_all_metrics
from eval.evaluate import _sync
from hybrid.consistency import enforce_consistency

ROOT = Path(__file__).resolve().parents[1]
CKPT_ORIG = str(ROOT / "checkpoints" / "sgnet" / "SGNet_X4_R.pth")
CKPT_ROBUST = str(ROOT / "runs" / "e7_6epoch" / "best.pth")
CACHE = ROOT / "tmp" / "task_a_cache"
OUT = ROOT / "results" / "task_a"

REGIMES = {
    "clean": {},
    "holes": {"hole_ratio": 0.1},
    "noise": {"noise_std": 0.02},
    "noise_strong": {"noise_std": 0.05},
    "block_holes": {"n_block_holes": 4},
    "combined": {"noise_std": 0.02, "hole_ratio": 0.1},
}


def _norm01(x):
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / max(hi - lo, 1e-8)


def masked_rmse(pred, gt, mask):
    m = mask & np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 50:
        return float("nan")
    e = pred[m].astype(np.float64) - gt[m].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def eval_row(pred, rec):
    m = compute_all_metrics(pred, rec["gt"], 10.0, rec["valid"], 6, 0.03)
    return {"rmse": m["rmse"], "mae": m["mae"], "boundary_rmse": m["boundary_rmse"],
            "edge_f1": m["edge_f1"], "psnr": m["psnr"], "ssim": m["ssim"]}


def cache_regime(regime, n_samples):
    cfg = DegradationConfig(**REGIMES[regime])
    ds = NYUv2Dataset(root=str(ROOT / "nyu_data" / "nyu_data"), split="val", scale=4, degradation=cfg)
    n = min(n_samples, len(ds))
    out_dir = CACHE / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        f = out_dir / f"{i:04d}.npz"
        if f.exists():
            continue
        s = ds[i]
        lr, lrm, rgb = s["lr_depth"].copy(), s["lr_valid_mask"].copy(), s["hr_rgb"].copy()
        gt, valid = s["depth_original"], s["valid_mask_original"]
        from deep.adaptive import detect_regime
        rec_regime = detect_regime(lr, lrm)
        # flip TTA inputs (flip both RGB and depth together)
        lr_f = np.ascontiguousarray(lr[:, ::-1])
        lrm_f = np.ascontiguousarray(lrm[:, ::-1])
        rgb_f = np.ascontiguousarray(rgb[:, ::-1])
        t0 = time.perf_counter()
        dd_orig = deep_hr_depth(lr, rgb, lrm, 4, ckpt_path=CKPT_ORIG)
        dd_robust = deep_hr_depth(lr, rgb, lrm, 4, ckpt_path=CKPT_ROBUST)
        # flip the regime-selected checkpoint
        flip_ckpt = CKPT_ORIG if rec_regime == "clean" else CKPT_ROBUST
        dd_flip = deep_hr_depth(lr_f, rgb_f, lrm_f, 4, ckpt_path=flip_ckpt)[:, ::-1]
        t_deep = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        from classical import classical_hr_depth
        dc = classical_hr_depth(lr, rgb, lrm, 4)
        t_cl = (time.perf_counter() - t0) * 1000
        np.savez_compressed(
            f, gt=gt.astype(np.float32), valid=valid.astype(bool), lr=lr.astype(np.float32),
            lrm=lrm.astype(bool), dc=dc, dd_orig=dd_orig, dd_robust=dd_robust,
            dd_flip=np.ascontiguousarray(dd_flip), name=str(s["name"]),
            t_deep=t_deep, t_classical=t_cl,
        )
        print(f"  {regime} {i + 1}/{n} deep={t_deep:.0f}ms classical={t_cl:.0f}ms", flush=True)


def load_regime(regime, n_samples):
    rows = []
    for i in range(min(n_samples, 150)):
        f = CACHE / regime / f"{i:04d}.npz"
        if not f.exists():
            break
        d = np.load(f, allow_pickle=False)
        rows.append({k: d[k] for k in d.files})
    return rows


def variants_for(rows, regime):
    """Yield (variant_name, per_image_rows, post_ms)."""
    def sel(r):
        return r["dd_orig"] if regime == "clean" else r["dd_robust"]

    results = {}

    def add(name, fn):
        per, ts = [], []
        for r in rows:
            t0 = time.perf_counter()
            pred = fn(r).astype(np.float32)
            ts.append((time.perf_counter() - t0) * 1000)
            per.append({"sample": str(r["name"]), "latency_ms": ts[-1], **eval_row(pred, r)})
        results[name] = {"per_image": per, "post_ms": float(np.mean(ts)),
                         "summary": {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "sample"}}

    noise = regime.startswith("noise")

    def cons(strength, iters):
        return lambda r: enforce_consistency(sel(r), r["lr"], r["lrm"], 4, strength=strength, iterations=iters)

    def clamp(margin):
        def fn(r):
            lr, m = r["lr"], r["lrm"]
            lo = float(lr[m & (lr > 0)].min()) - margin
            hi = float(lr[m & (lr > 0)].max()) + margin
            return np.clip(sel(r), lo, hi)
        return fn

    def median_gated(thresh):
        def fn(r):
            d = sel(r)
            gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
            gate = _norm01(cv2.magnitude(gx, gy)) > thresh
            med = cv2.medianBlur(d, 3)
            return np.where(gate, med, d)
        return fn

    add("deep_adaptive", sel)
    add("tta_flip", lambda r: 0.5 * (sel(r) + r["dd_flip"]))
    add("avg_ckpt", lambda r: 0.5 * (r["dd_orig"] + r["dd_robust"]))
    if not noise:
        add("deep_adaptive_cons", cons(0.25, 1))
        add("tta_flip_cons", lambda r: enforce_consistency(0.5 * (sel(r) + r["dd_flip"]), r["lr"], r["lrm"], 4, strength=0.25))
        add("avg_ckpt_cons", lambda r: enforce_consistency(0.5 * (r["dd_orig"] + r["dd_robust"]), r["lr"], r["lrm"], 4, strength=0.25))
    for s_ in (0.1, 0.25, 0.5):
        for it in (1, 2, 3):
            if not noise:
                add(f"cons_s{s_}_i{it}", cons(s_, it))
    for mg in (0.01, 0.05, 0.1, 0.2):
        add(f"clamp_m{mg}", clamp(mg))
    for th in (0.5, 0.7, 0.9):
        add(f"median_g{th}", median_gated(th))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--keep-cache", action="store_true")
    a = ap.parse_args(argv)

    for regime in a.regimes:
        if a.cache:
            print(f"== caching {regime} ==", flush=True)
            cache_regime(regime, a.samples)
        if a.variants:
            rows = load_regime(regime, a.samples)
            if not rows:
                print(f"no cache for {regime}; run --cache first", flush=True)
                continue
            print(f"== variants {regime} ({len(rows)} samples) ==", flush=True)
            res = variants_for(rows, regime)
            out = OUT / regime
            out.mkdir(parents=True, exist_ok=True)
            (out / "variants_summary.json").write_text(json.dumps(res, indent=2))
            lines = ["| variant | RMSE | MAE | bnd RMSE | edge F1 | post ms |",
                     "|---|---|---|---|---|---|"]
            for name, v in sorted(res.items(), key=lambda kv: kv[1]["summary"]["rmse"]):
                s = v["summary"]
                lines.append(f"| {name} | {s['rmse']:.5f} | {s['mae']:.5f} | {s['boundary_rmse']:.4f} | "
                             f"{s['edge_f1']:.3f} | {v['post_ms']:.1f} |")
            (out / "variants_summary.md").write_text("\n".join(lines) + "\n")
            print("\n".join(lines), flush=True)
            if not a.keep_cache:
                for f in (CACHE / regime).glob("*.npz"):
                    f.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
