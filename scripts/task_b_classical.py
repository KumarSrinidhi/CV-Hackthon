# Team Owner: Kumar
# Workstream: Classical
# Module: scripts/task_b_classical.py
# Description: Task B - diagnose and repair the classical branch (target: RMSE below
# bicubic). Staged variants: baseline, stage-isolation, MAD outlier rejection,
# progressive 2x JBU, WLS solve, weighted-median edge-band refinement.
"""Examples:
    python scripts/task_b_classical.py --samples 30 --variants diag
    python scripts/task_b_classical.py --samples 150 --variants all --regimes clean
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
from PIL import Image

from classical.edge_upsampling import upsample_dense_depth
from classical.hole_filling import prepare_depth
from classical.jbu import joint_bilateral_upsample, to_float_rgb
from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from data.preprocessing import fill_invalid_nearest
from eval.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task_b"

REGIMES = {
    "clean": {},
    "holes": {"hole_ratio": 0.1},
    "noise": {"noise_std": 0.02},
    "noise_strong": {"noise_std": 0.05},
    "block_holes": {"n_block_holes": 4},
    "combined": {"noise_std": 0.02, "hole_ratio": 0.1},
}


def eval_row(pred, gt, valid):
    m = compute_all_metrics(pred.astype(np.float32), gt, 10.0, valid, 6, 0.03)
    return {k: m[k] for k in ("rmse", "mae", "boundary_rmse", "edge_f1", "psnr", "ssim")}


def mad_reject(lr, valid, k=3.0):
    """Mark LR outliers invalid via median absolute deviation vs a 5x5 median."""
    med = cv2.medianBlur(lr, 5)
    dev = np.abs(lr - med)
    mad = cv2.medianBlur(dev, 5)
    bad = valid & (dev > k * (1.4826 * mad + 1e-3))
    return lr.copy(), valid & ~bad


def progressive_jbu(lr, valid, rgb_f, scale, radius=1):
    """JBU in log2(scale) 2x stages, guided by progressively finer RGB."""
    out, out_v = lr, valid
    s = 2
    while s <= scale:
        g = rgb_f[:: max(1, scale // s), :: max(1, scale // s)]
        out = joint_bilateral_upsample(out, out_v, g, radius=radius)
        out_v = np.ones_like(out, bool)
        s *= 2
    return out


def classical_smooth(lr, valid, rgb_f, scale):
    """Dense path without transition sharpening (back-projection only)."""
    depth, _ = prepare_depth(lr, valid)
    if not valid.all():
        from data.preprocessing import fill_invalid_nearest
        depth = fill_invalid_nearest(depth, valid)
    h, w = depth.shape
    H, W = rgb_f.shape[:2]
    out = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(depth.min()), float(depth.max())
    for _ in range(8):
        projected = np.asarray(
            Image.fromarray(out).resize((w, h), Image.Resampling.BICUBIC))
        residual = depth - projected
        img = Image.fromarray(residual).resize((W, H), Image.Resampling.BICUBIC)
        out = np.clip(out + 1.1 * np.asarray(img, np.float32), lo, hi)
    return out


def classical_wls(lr, valid, rgb_f, scale, lam=0.4, iters=40, cap=20.0):
    """Weighted Least Squares solve: data term anchors the bicubic upsample to the
    LR measurements, smoothness term is RGB-edge-aware (scipy sparse CG)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    depth, _ = prepare_depth(lr, valid)
    from data.preprocessing import fill_invalid_nearest
    if not valid.all():
        depth = fill_invalid_nearest(depth, valid)
    H, W = rgb_f.shape[:2]
    h, w = depth.shape
    z = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    # anchor every LR pixel across its HR cell: measured cells strong, holes weak
    cell_y = np.minimum(np.arange(H) // scale, h - 1)
    cell_x = np.minimum(np.arange(W) // scale, w - 1)
    data_w = np.where(valid[np.ix_(cell_y, cell_x)], 1.0, 0.02).astype(np.float64)

    gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY).astype(np.float64)
    dkin = np.array([0.25, 0.5, 0.25])
    ekin = np.array([-1.0, 0.0, 1.0])
    wx = np.abs(cv2.sepFilter2D(gray, -1, dkin, ekin))
    wy = np.abs(cv2.sepFilter2D(gray, -1, ekin, dkin))
    # small weights where RGB has an edge -> depth edges preserved there
    wx = np.minimum(lam / (np.abs(wx) + 0.02), cap)
    wy = np.minimum(lam / (np.abs(wy) + 0.02), cap)

    dw = data_w.ravel()
    n = H * W
    idx = np.arange(n).reshape(H, W)
    # pair weights: wh[y, x] weights the horizontal pair (y, x-1)-(y, x) for x >= 1;
    # wv[y, x] weights the vertical pair (y-1, x)-(y, x) for y >= 1.
    wh = wx[:, 1:]  # (H, W-1)
    wv = wy[1:, :]  # (H-1, W)
    inc_h = np.zeros((H, W))
    inc_h[:, 1:] += wh
    inc_h[:, :-1] += wh
    inc_v = np.zeros((H, W))
    inc_v[1:, :] += wv
    inc_v[:-1, :] += wv
    diag = data_w + inc_h + inc_v

    rows_h = idx[:, 1:].ravel()
    cols_h = idx[:, :-1].ravel()
    vals_h = -wh.ravel()
    rows_v = idx[1:, :].ravel()
    cols_v = idx[:-1, :].ravel()
    vals_v = -wv.ravel()

    off_r = np.concatenate([rows_h, rows_v])
    off_c = np.concatenate([cols_h, cols_v])
    off_v = np.concatenate([vals_h, vals_v])
    A = sp.coo_matrix(
        (np.concatenate([diag.ravel(), off_v, off_v]),
         (np.concatenate([np.arange(n), off_r, off_c]),
          np.concatenate([np.arange(n), off_c, off_r]))),
        shape=(n, n),
    ).tocsr()
    b = dw * z.ravel()
    d, info = spla.cg(A, b, maxiter=iters, rtol=1e-6)
    out = np.clip(d.reshape(H, W), float(depth.min()), float(depth.max()))
    return out.astype(np.float32)


def weighted_median_edge_band(pred, rgb_f, k=3, edge_p=80):
    """On high-gradient RGB pixels, replace with the colour-weighted median of the
    k x k neighbourhood."""
    gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    band = grad > np.percentile(grad, edge_p)
    med = cv2.medianBlur(pred, k)
    return np.where(band, med, pred)


def classical_baseline(lr, valid, rgb_f, scale):
    from classical.pipeline import reconstruct
    return reconstruct(lr, rgb_f, valid, scale, method="classical")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--regimes", nargs="+", default=["clean"])
    ap.add_argument("--variants", nargs="+", default=[
        "classical", "smooth_only", "dense_noBP", "mad_classical",
        "jbu_prog", "jbu_prog_mad", "wls", "wls_lam0.1", "wls_lam1.0"])
    ap.add_argument("--edge-refine", action="store_true",
                    help="also evaluate weighted-median edge-band refinement of classical")
    ap.add_argument("--bicubic-ref", action="store_true")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for regime in a.regimes:
        cfg = DegradationConfig(**REGIMES[regime])
        ds = NYUv2Dataset(root=str(ROOT / "nyu_data" / "nyu_data"), split="val", scale=4, degradation=cfg)
        n = min(a.samples, len(ds))
        res = {}
        rows_all = {name: [] for name in a.variants}
        for i in range(n):
            s = ds[i]
            gt, valid = s["depth_original"], s["valid_mask_original"]
            lr, lrm, rgb = s["lr_depth"].copy(), s["lr_valid_mask"].copy(), s["hr_rgb"].copy()
            rgb_f = to_float_rgb(np.asarray(rgb))
            for name in a.variants:
                t0 = time.perf_counter()
                try:
                    if name == "classical":
                        pred = classical_baseline(lr, lrm, rgb_f, 4)
                    elif name == "smooth_only":
                        pred = classical_smooth(lr, lrm, rgb_f, 4)
                    elif name.startswith("blend_g"):
                        g = float(name.split("g")[1])
                        d, v2 = prepare_depth(lr, lrm)
                        d = d if v2.all() else fill_invalid_nearest(np.where(v2, d, np.float32(0)), v2)
                        pred = (1.0 - g) * classical_smooth(lr, lrm, rgb_f, 4) \
                            + g * upsample_dense_depth(d, rgb_f, 4)
                    elif name == "dense_noBP":
                        d, v2 = prepare_depth(lr, lrm)
                        d = d if v2.all() else fill_invalid_nearest(np.where(v2, d, np.float32(0)), v2)
                        pred = upsample_dense_depth(d, rgb_f, 4)
                    elif name == "mad_classical":
                        lr2, v2 = mad_reject(np.asarray(lr, np.float32),
                                             (np.isfinite(lr) & (lr > 0)) if lrm is None
                                             else np.asarray(lrm, bool))
                        pred = classical_baseline(lr2, v2, rgb_f, 4)
                    elif name == "jbu_prog":
                        pred = progressive_jbu(lr, lrm, rgb_f, 4)
                    elif name == "jbu_prog_mad":
                        lr2, v2 = mad_reject(np.asarray(lr, np.float32),
                                             (np.isfinite(lr) & (lr > 0)) if lrm is None
                                             else np.asarray(lrm, bool))
                        pred = progressive_jbu(lr2, v2, rgb_f, 4)
                    elif name.startswith("wls"):
                        lam = float(name.split("lam")[1]) if "lam" in name else 0.4
                        pred = classical_wls(lr, lrm, rgb_f, 4, lam=lam)
                    else:
                        continue
                    dt = (time.perf_counter() - t0) * 1000
                    rows_all[name].append({"sample": str(s["name"]), "latency_ms": dt,
                                           **eval_row(pred, gt, valid)})
                except Exception as e:  # noqa: BLE001
                    print(f"  {name} failed on {s['name']}: {type(e).__name__}: {e}", flush=True)
            if a.bicubic_ref:
                from eval.evaluate import _make_baseline
                pred = _make_baseline("bicubic")(lr, rgb, lrm, 4)
                rows_all.setdefault("bicubic", []).append(
                    {"sample": str(s["name"]), "latency_ms": 0.0, **eval_row(pred, gt, valid)})
            print(f"  {regime} {i + 1}/{n}", flush=True)
        for name, rows in rows_all.items():
            if rows:
                res[name] = {k: float(np.mean([r[k] for r in rows]))
                             for k in ("rmse", "mae", "boundary_rmse", "edge_f1", "latency_ms")}
                res[name]["n"] = len(rows)
        report[regime] = res
        lines = [f"### {regime} (n={n})", "",
                 "| variant | RMSE | MAE | bnd RMSE | edge F1 | ms |", "|---|---|---|---|---|---|"]
        for name, e in sorted(res.items(), key=lambda kv: kv[1]["rmse"]):
            lines.append(f"| {name} | {e['rmse']:.5f} | {e['mae']:.5f} | {e['boundary_rmse']:.4f} | "
                         f"{e['edge_f1']:.3f} | {e['latency_ms']:.0f} |")
        print("\n".join(lines), flush=True)
        (OUT / f"{regime}_summary.json").write_text(json.dumps(report[regime], indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
