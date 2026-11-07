# Team Owner: Kumar
# Workstream: Classical
# Module: scripts/task_b_hf_benchmark.py
# Description: Report-only verification of the improved classical branch on the
# 100-sample HF NYU clean benchmark (same samples/protocol as the team baseline).
"""Usage:
    python scripts/task_b_hf_benchmark.py --tars datasets/val-000000.tar --max-samples 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from classical.edge_upsampling import upsample_dense_depth
from classical.hole_filling import prepare_depth
from classical.jbu import to_float_rgb
from data.degradation import DegradationConfig
from data.preprocessing import fill_invalid_nearest
from eval.evaluate import _make_baseline, evaluate_methods, save_results
from scripts.eval_all3_hf import HFNYUValDataset
from scripts.task_b_classical import REGIMES, classical_baseline, classical_smooth, eval_row


def blend04(lr, lrm, rgb):
    rgb_f = to_float_rgb(np.asarray(rgb))
    smooth = classical_smooth(lr, lrm, rgb_f, 4)
    d, v2 = prepare_depth(lr, lrm)
    d = d if v2.all() else fill_invalid_nearest(np.where(v2, d, np.float32(0)), v2)
    dense = upsample_dense_depth(d, rgb_f, 4)
    return 0.6 * smooth + 0.4 * dense


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", default=["datasets/val-000000.tar"])
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--out", default="results/task_b_hf_clean100")
    a = ap.parse_args(argv)

    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=a.tars)
    bic = _make_baseline("bicubic")
    rows = {"bicubic": [], "classical": [], "classical_blend_g0.4": []}
    n = min(a.max_samples, len(ds))
    for i in range(n):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        lr, lrm, rgb = s["lr_depth"].copy(), s["lr_valid_mask"].copy(), s["hr_rgb"].copy()
        rows["bicubic"].append({"sample": s["name"], **eval_row(bic(lr, rgb, lrm, 4), gt, valid)})
        rows["classical"].append({"sample": s["name"], **eval_row(classical_baseline(lr, lrm, rgb, 4), gt, valid)})
        rows["classical_blend_g0.4"].append({"sample": s["name"], **eval_row(blend04(lr, lrm, rgb), gt, valid)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n}", flush=True)

    out = {}
    for name, r in rows.items():
        out[name] = {k: float(np.mean([x[k] for x in r])) for k in r[0] if k != "sample"}
        out[name]["n"] = len(r)
    lines = ["### HF NYU clean100 x4 (report-only verification)", "",
             "| method | RMSE | MAE | bnd RMSE | edge F1 |", "|---|---|---|---|---|"]
    for name, e in sorted(out.items(), key=lambda kv: kv[1]["rmse"]):
        lines.append(f"| {name} | {e['rmse']:.4f} | {e['mae']:.4f} | {e['boundary_rmse']:.4f} | {e['edge_f1']:.3f} |")
    print("\n".join(lines), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(__import__("json").dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
