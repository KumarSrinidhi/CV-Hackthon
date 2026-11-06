# Team Owner: Kumar
# Workstream: Deep Learning / Forensics
# Module: scripts/error_analysis.py
# Description: Per-scene, per-region and RGB-guidance forensic error analysis (plan.txt §6.C).
"""Per-image metrics, region-wise error breakdown, rankings, visual failure atlas and
RGB-guidance diagnostic variants over the HF NYU val benchmark.

    python scripts/error_analysis.py --tars datasets/val-000000.tar --max-samples 100 \
        --methods bicubic deep classical hybrid --out results/error_analysis

    python scripts/error_analysis.py --tars datasets/val-000000.tar --rgb-probe \
        --max-samples 100 --rgb-probe-samples 25 --out results/error_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from data.dataset import DepthSRDataset
from data.degradation import DegradationConfig
from eval.evaluate import load_method
from eval.metrics import compute_all_metrics, depth_discontinuity_mask

from scripts.eval_all3_hf import HFNYUValDataset

METRICS = ["rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_precision", "edge_recall", "edge_f1"]


def region_masks(gt: np.ndarray, valid: np.ndarray, rgb: np.ndarray, data_range: float) -> dict:
    """Diagnostic region definitions, consistent across methods (plan.txt §6.C1)."""
    edges = depth_discontinuity_mask(gt, valid, data_range)
    edges_near = ndi.binary_dilation(edges, iterations=2)
    band = edges_near & ~edges
    smooth = valid & ~edges_near
    invalid = ~valid

    gx = cv2.Sobel(rgb.mean(axis=2), cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(rgb.mean(axis=2), cv2.CV_32F, 0, 1, 3)
    rgb_grad = cv2.magnitude(gx, gy)
    strong_rgb = rgb_grad > np.percentile(rgb_grad[valid], 90) if valid.any() else np.zeros_like(valid)

    gy_d = np.abs(np.diff(gt, axis=0, prepend=gt[:1]))
    gx_d = np.abs(np.diff(gt, axis=1, prepend=gt[:, :1]))
    depth_var = ndi.uniform_filter(gx_d + gy_d, 5)
    texture_flat_depth = strong_rgb & (depth_var < 0.3 * data_range / 10.0) & smooth
    edge_density = ndi.uniform_filter(edges.astype(np.float32), 15)
    thin = edges & (edge_density < 0.15)
    strong_depth_edges = edges & ~thin
    weak_rgb_edges = strong_depth_edges & ~ndi.binary_dilation(strong_rgb, iterations=3)

    return {
        "region_smooth": smooth,
        "region_edge_band": band & valid,
        "region_on_edges": edges & valid,
        "region_thin_structures": thin & valid,
        "region_strong_depth_edges": strong_depth_edges & valid,
        "region_invalid": invalid,
        "region_rgb_texture_flat_depth": texture_flat_depth & valid,
        "region_depth_edges_weak_rgb": weak_rgb_edges & valid,
    }


def masked_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 50:
        return float("nan")
    e = pred[m].astype(np.float64) - gt[m].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def save_atlas(sample: dict, preds: dict, out_path: Path, data_range: float, valid: np.ndarray) -> None:
    gt = sample["depth_original"]
    vmin, vmax = float(np.nanmin(gt[valid])), float(np.nanmax(gt[valid]))
    gt_edges = depth_discontinuity_mask(gt, valid, data_range)
    panels, titles = [], []
    lr_up = cv2.resize(
        sample["lr_depth"], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    for arr, title in [
        (gt, "GT depth"), (lr_up, "LR input (NN up)"), (sample["hr_rgb"], "RGB"),
    ]:
        panels.append(arr)
        titles.append(title)
    for name in ["bicubic", "deep", "classical", "hybrid"]:
        if name in preds:
            panels.append(preds[name])
            titles.append(name)
    for name in ["bicubic", "deep", "classical", "hybrid"]:
        if name in preds:
            err = np.abs(preds[name] - gt)
            panels.append(np.where(valid, err, np.nan))
            titles.append(f"|err| {name}")
    panels.append(np.where(valid, gt, np.nan))
    titles.append("GT validity domain")
    if "deep" in preds:
        pe = depth_discontinuity_mask(preds["deep"], valid, data_range)
        overlay = np.stack([gt] * 3, axis=-1)
        overlay = (overlay - vmin) / max(vmax - vmin, 1e-6)
        overlay[..., 0][gt_edges] = 1.0
        overlay[..., 1][gt_edges] = 0.2
        overlay[..., 2][pe] = 0.2
        panels.append(overlay)
        titles.append("edges: GT(red) vs deep(blue)")

    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, arr, title in zip(axes.flat, panels, titles):
        if arr.ndim == 3:
            ax.imshow(np.clip(arr, 0, 1) if title == "RGB" else np.clip(arr, 0, 1))
        else:
            if title.startswith("|err|"):
                vmax_e = np.nanmax([np.nanmax(np.where(valid, np.abs(preds[k] - gt), np.nan))
                                    for k in preds]) if preds else 1.0
                im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=vmax_e)
            elif "edges" in title:
                ax.imshow(np.clip(arr, 0, 1))
            elif title == "GT validity domain":
                ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(arr, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{sample['name']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["bicubic", "deep", "classical", "hybrid"])
    ap.add_argument("--out", default="results/error_analysis")
    ap.add_argument("--atlas-count", type=int, default=6)
    ap.add_argument("--rgb-probe", action="store_true")
    ap.add_argument("--rgb-probe-samples", type=int, default=25)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = HFNYUValDataset(a.scale, DegradationConfig(), tar_paths=a.tars)
    n = min(a.max_samples, len(ds))
    fns = {name: load_method(name) for name in a.methods}
    print(f"error_analysis: {n} samples, methods {list(fns)}")

    rows, per_region_rows = [], []
    atlas_data = []
    for i in range(n):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        preds = {}
        rec = {"sample": s["name"]}
        for name, fn in fns.items():
            pred = np.asarray(fn(s["lr_depth"].copy(), s["hr_rgb"].copy(), s["lr_valid_mask"].copy(), s["scale"]), np.float32)
            preds[name] = pred
            m = compute_all_metrics(pred, gt, s["data_range"], valid)
            rec.update({f"{name}_{k}": m[k] for k in METRICS})
        regions = region_masks(gt, valid, s["hr_rgb"], s["data_range"])
        rec.update({
            "valid_fraction": float(valid.mean()),
            "depth_range_m": float(gt[valid].max() - gt[valid].min()) if valid.any() else float("nan"),
            "edge_density": float(depth_discontinuity_mask(gt, valid, s["data_range"]).mean()),
        })
        for rname, rmask in regions.items():
            for name, pred in preds.items():
                per_region_rows.append({
                    "sample": s["name"], "method": name, "region": rname,
                    "rmse": masked_rmse(pred, gt, rmask), "n_pixels": int(rmask.sum()),
                })
        rows.append(rec)
        atlas_data.append((s, preds, rec))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    with open(out / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "per_region_rmse.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "method", "region", "rmse", "n_pixels"])
        w.writeheader()
        w.writerows(per_region_rows)

    agg_region = {}
    for r in per_region_rows:
        key = (r["region"], r["method"])
        vals = agg_region.setdefault(key, [])
        if np.isfinite(r["rmse"]):
            vals.append(r["rmse"])
    with open(out / "region_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "method", "mean_region_rmse", "n_scenes"])
        for (region, method), vals in sorted(agg_region.items()):
            w.writerow([region, method, float(np.mean(vals)), len(vals)])

    deep_key = "deep_rmse" if "deep_rmse" in rows[0] else None
    bic_key = "bicubic_rmse" if "bicubic_rmse" in rows[0] else None
    rankings = {}
    for metric in ["rmse", "edge_f1", "boundary_rmse"]:
        key = f"deep_{metric}"
        if key in rows[0]:
            vals = [(r["sample"], r[key]) for r in rows]
            vals = [(s, v) for s, v in vals if np.isfinite(v)]
            worst = sorted(vals, key=lambda t: -t[1]) if metric != "edge_f1" else sorted(vals, key=lambda t: t[1])
            rankings[f"worst_deep_{metric}"] = worst[:10]
    if deep_key and bic_key:
        diffs = sorted(((r["sample"], r[deep_key] - r[bic_key]) for r in rows), key=lambda t: -t[1])
        rankings["largest_deep_vs_bicubic_regression"] = diffs[:10]
    if "hybrid_rmse" in rows[0] and deep_key:
        diffs = sorted(((r["sample"], r["hybrid_rmse"] - r[deep_key]) for r in rows), key=lambda t: -t[1])
        rankings["largest_hybrid_vs_deep_regression"] = diffs[:10]

    if deep_key:
        corr = {
            "rmse_vs_edge_density": spearman([r["deep_rmse"] for r in rows], [r["edge_density"] for r in rows]),
            "rmse_vs_valid_fraction": spearman([r["deep_rmse"] for r in rows], [r["valid_fraction"] for r in rows]),
            "rmse_vs_depth_range": spearman([r["deep_rmse"] for r in rows], [r["depth_range_m"] for r in rows]),
        }
    else:
        corr = {}
    with open(out / "rankings.json", "w") as f:
        json.dump({"rankings": {k: [[s, float(v)] for s, v in v10] for k, v10 in rankings.items()},
                   "spearman": corr}, f, indent=2)

    if deep_key:
        worst_samples = [s for s, _ in rankings.get("worst_deep_rmse", [])[: a.atlas_count]]
        for s, preds, _ in atlas_data:
            if s["name"] in worst_samples:
                save_atlas(s, preds, out / f"atlas_{s['name']}.png", s["data_range"],
                           s["valid_mask_original"])

    if a.rgb_probe:
        rgb_out = out / "rgb_probe"
        rgb_out.mkdir(exist_ok=True)
        rng = np.random.default_rng(0)
        probes = {}
        for i in range(min(a.rgb_probe_samples, n)):
            s = ds[i]
            gt, valid = s["depth_original"], s["valid_mask_original"]
            rgb = s["hr_rgb"]
            j = (i + 1) % n
            s_other = ds[j]
            variants = {
                "deep_original": rgb,
                "deep_grey": np.full_like(rgb, rgb.mean()),
                "deep_blurred": cv2.GaussianBlur(rgb, (0, 0), 4.0),
                "deep_shuffled": _block_shuffle(rgb, rng),
                "deep_other_sample": s_other["hr_rgb"],
            }
            for vname, vrgb in variants.items():
                pred = np.asarray(fns["deep"](s["lr_depth"].copy(), vrgb.copy(),
                                              s["lr_valid_mask"].copy(), s["scale"]), np.float32)
                m = compute_all_metrics(pred, gt, s["data_range"], valid)
                probes.setdefault(vname, []).append(m)
            if (i + 1) % 5 == 0:
                print(f"  rgb-probe {i + 1}/{min(a.rgb_probe_samples, n)}")
        summary = {}
        for vname, ms in probes.items():
            summary[vname] = {k: float(np.nanmean([m[k] for m in ms])) for k in METRICS}
        with open(rgb_out / "rgb_probe_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("rgb probe:", json.dumps(summary, indent=2))

    print(f"saved analysis to {out}")
    return 0


def _block_shuffle(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = rgb.shape
    bs = 32
    out = rgb.copy()
    blocks = [(y, x) for y in range(0, h - bs + 1, bs) for x in range(0, w - bs + 1, bs)]
    perm = rng.permutation(len(blocks))
    for (y, x), (y2, x2) in zip(blocks, [blocks[j] for j in perm]):
        out[y:y + bs, x:x + bs] = rgb[y2:y2 + bs, x2:x2 + bs]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
