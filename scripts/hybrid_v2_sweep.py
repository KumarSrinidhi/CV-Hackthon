# Team Owner: Kumar
# Workstream: Hybrid
# Module: scripts/hybrid_v2_sweep.py
# Description: Validation calibration of hybrid fusion weights per regime (plan.txt §13, B2).
"""Sweep confidence-weight combinations for hybrid fusion on validation data.

For each regime (clean / holes / noise), classical and robust-deep predictions are
computed once per sample; fusion + consistency enforcement is cheap, so a full weight
grid is evaluated. Selection uses validation ONLY. The winner feeds hybrid/pipeline.py.

    python scripts/hybrid_v2_sweep.py --samples 60 --out results/hybrid_v2_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from classical.pipeline import classical_hr_depth
from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig, degrade
from deep.adaptive import _DEFAULT_ROBUST
from deep.inference import deep_hr_depth
from hybrid.confidence import _normalize01, _resolve_valid
from hybrid.consistency import enforce_consistency

REGIMES = [
    ("clean", DegradationConfig()),
    ("holes", DegradationConfig(hole_ratio=0.1)),
    ("noise", DegradationConfig(noise_std=0.02)),
]


def masked_rmse(pred, gt, mask):
    m = mask & np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 50:
        return float("nan")
    e = pred[m].astype(np.float64) - gt[m].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def evidence_terms(dc, dd, lr, lrm, rgb):
    import cv2
    from scipy import ndimage as ndi
    from classical.adaptive_jbu import edge_evidence

    gx = cv2.Sobel(dc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dc, cv2.CV_32F, 0, 1, ksize=3)
    edge = _normalize01(cv2.magnitude(gx, gy))
    valid = _resolve_valid(lr, lrm)
    if valid.any():
        dist = ndi.distance_transform_edt(~valid)
        dist_hr = cv2.resize(dist.astype(np.float32), (dc.shape[1], dc.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        dist_score = _normalize01(dist_hr)
    else:
        dist_score = np.ones(dc.shape, np.float32)
    try:
        _, agree, _ = edge_evidence(lr, rgb)
        agreement = agree.astype(np.float32)
    except Exception:
        agreement = np.zeros(dc.shape, np.float32)
    return edge, dist_score, agreement


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--root", default="nyu_data/nyu_data")
    ap.add_argument("--robust-ckpt", default=str(_DEFAULT_ROBUST))
    ap.add_argument("--out", default="results/hybrid_v2_sweep")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid_w_edge = [0.0, 0.1, 0.2, 0.3]
    grid_w_dist = [0.0, 0.1, 0.2, 0.3, 0.4]
    grid_w_agree = [0.0, 0.1, 0.2]
    fixed_alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    report = {"per_regime": {}, "config": vars(a)}
    for regime, cfg in REGIMES:
        ds = NYUv2Dataset(root=a.root, split="val", scale=a.scale, degradation=cfg)
        n = min(a.samples, len(ds))
        rows = []
        for i in range(n):
            s = ds[i]
            gt, valid = s["depth_original"], s["valid_mask_original"]
            deg = degrade(gt, a.scale, cfg, valid_hr=valid, sample_index=i)
            lr, lrm = deg["lr_depth"], deg["lr_valid_mask"]
            rgb = s["hr_rgb"]
            dc = classical_hr_depth(lr, rgb, lrm, a.scale)
            dd = deep_hr_depth(lr, rgb, lrm, a.scale, ckpt_path=a.robust_ckpt)
            edge, dist_score, agreement = evidence_terms(dc, dd, lr, lrm, rgb)
            rec = {"gt": gt, "valid": valid, "dc": dc, "dd": dd,
                   "edge": edge, "dist": dist_score, "agree": agreement,
                   "lr": lr, "lrm": lrm}
            rows.append(rec)
            print(f"  {regime} {i + 1}/{n} computed", flush=True)

        results = {}
        for we, wd, wa in itertools.product(grid_w_edge, grid_w_dist, grid_w_agree):
            vals = []
            for r in rows:
                wc = we * r["edge"] + wd * (1.0 - r["dist"]) + wa * r["agree"]
                wc = np.clip(wc, 0.0, 1.0)
                fused = enforce_consistency(wc * r["dc"] + (1 - wc) * r["dd"], r["lr"], r["lrm"], a.scale)
                vals.append(masked_rmse(fused, r["gt"], r["valid"]))
            results[f"e{we}_d{wd}_a{wa}"] = float(np.nanmean(vals))
        for alpha in fixed_alphas:
            vals = []
            for r in rows:
                fused = enforce_consistency(alpha * r["dc"] + (1 - alpha) * r["dd"], r["lr"], r["lrm"], a.scale)
                vals.append(masked_rmse(fused, r["gt"], r["valid"]))
            results[f"fixed_alpha_{alpha}"] = float(np.nanmean(vals))
        vals = []
        for r in rows:
            better = (np.abs(r["dc"] - r["gt"]) <= np.abs(r["dd"] - r["gt"])).astype(np.float32)
            oracle = better * r["dc"] + (1 - better) * r["dd"]
            vals.append(masked_rmse(oracle, r["gt"], r["valid"]))
        results["oracle_per_pixel"] = float(np.nanmean(vals))
        results["deep_only"] = float(np.nanmean([masked_rmse(r["dd"], r["gt"], r["valid"]) for r in rows]))
        results["classical_only"] = float(np.nanmean([masked_rmse(r["dc"], r["gt"], r["valid"]) for r in rows]))

        best = min((v, k) for k, v in results.items() if not k.startswith("oracle"))
        report["per_regime"][regime] = {"results": results, "best": best[1], "best_rmse": best[0],
                                        "n_samples": n}
        print(f"{regime}: best={best[1]} rmse={best[0]:.5f} deep_only={results['deep_only']:.5f} "
              f"oracle={results['oracle_per_pixel']:.5f}", flush=True)

    out_json = out / "sweep_results.json"
    out_json.write_text(json.dumps(report, indent=2))
    print("saved", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
