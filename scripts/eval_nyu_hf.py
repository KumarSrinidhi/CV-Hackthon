# Team Owner: GPU verification run
# Module: scripts/eval_nyu_hf.py
# Description: Real-dataset GPU evaluation on the HuggingFace NYU-Depth-v2 val split
#              (sayakpaul/nyu_depth_v2, FastDepth 654-image labelled test set, Apache-2.0).
#              Same degradation + masked-metric protocol as eval/evaluate.py; the dataset
#              source differs from the SGNet paper bundle, so numbers are NOT directly
#              comparable to the paper (documented in docs/GPU_RUN_REPORT.md).
"""Usage:
    python scripts/eval_nyu_hf.py --tars datasets/val-000000.tar datasets/val-000001.tar
    python scripts/eval_nyu_hf.py --max-samples 30      # quick subset
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from PIL import Image

from data.degradation import downsample
from data.preprocessing import fill_invalid_nearest, to_float_rgb
from deep import inference as inf
from eval.metrics import compute_all_metrics

DATA_RANGE = 10.0  # NYU metres, fixed, same as eval/metrics.py
SCALE = 4
CV2 = True


def bicubic_up(lr: np.ndarray) -> np.ndarray:
    h, w = lr.shape[0] * SCALE, lr.shape[1] * SCALE
    img = Image.fromarray(np.ascontiguousarray(lr), mode="F")
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32)


def iter_h5(tar_paths: list[str]):
    for tp in tar_paths:
        try:
            with tarfile.open(tp, "r") as tar:
                for member in tar:
                    if not member.name.endswith(".h5"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    with h5py.File(io.BytesIO(f.read()), "r") as h5:
                        rgb = np.transpose(np.array(h5["rgb"]), (1, 2, 0))
                        depth = np.array(h5["depth"], dtype=np.float32)
                    yield member.name, rgb, depth
        except tarfile.ReadError:
            print(f"note: {tp} is truncated (partial download); stopping after its complete members")
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=654)
    ap.add_argument("--out", default="results/hf_nyu_val_x4")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "bicubic": lambda lr, rgb, valid: bicubic_up(lr),
        "deep": lambda lr, rgb, valid: inf.deep_hr_depth(lr, rgb, valid, SCALE, device=a.device),
    }
    dev = inf.get_device(a.device)
    print(f"device: {dev} | scale x{SCALE} | data_range {DATA_RANGE} | border 6 | tars: {a.tars}")

    rows, lat = [], {k: [] for k in methods}
    n_done = n_fail = 0
    t_start = time.perf_counter()
    for name, rgb, depth in iter_h5(a.tars):
        if n_done >= a.max_samples:
            break
        if depth.shape[0] % SCALE or depth.shape[1] % SCALE:
            continue
        rgb = to_float_rgb(rgb)
        valid_gt = np.isfinite(depth) & (depth > 0)
        if valid_gt.mean() < 0.95:
            continue
        # GT is dense: LR valid = all (matches clean bicubic protocol, operator_exact=True)
        lr = downsample(depth, SCALE, "bicubic")
        lr_valid = np.ones_like(lr, dtype=bool)

        row = {"sample": Path(name).stem}
        for mname, fn in methods.items():
            try:
                t0 = time.perf_counter()
                pred = fn(lr.copy(), rgb, lr_valid.copy())
                dt = (time.perf_counter() - t0) * 1000.0
                m = compute_all_metrics(pred, depth, DATA_RANGE, valid=valid_gt, border=6)
                m.update(latency_ms=dt, ok=True)
                if mname == "deep":
                    lat[mname].append(dt)
            except Exception as e:  # noqa: BLE001
                m = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                n_fail += 1
            row[mname] = m
        rows.append(row)
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done}/{a.max_samples} done ({time.perf_counter() - t_start:.0f}s)")

    # ------------------------------------------------------------------ write
    import csv

    keys = ["rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_f1", "latency_ms"]
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "method"] + keys)
        for r in rows:
            for mname in methods:
                m = r[mname]
                vals = [f"{m[k]:.6f}" if (m.get("ok") and k in m and np.isfinite(m.get(k, np.nan))) else "" for k in keys]
                w.writerow([r["sample"], mname] + vals)

    summary = {}
    for mname in methods:
        ok = [r[mname] for r in rows if r[mname].get("ok")]
        summary[mname] = {
            "n_ok": len(ok),
            "n_failed": sum(1 for r in rows if not r[mname].get("ok")),
            **{k: float(np.nanmean([m[k] for m in ok])) for k in keys},
        }
    meta = {
        "dataset": "huggingface:sayakpaul/nyu_depth_v2 (FastDepth val, 654 imgs)",
        "scale": SCALE,
        "data_range": DATA_RANGE,
        "border": 6,
        "degradation": "PIL bicubic x4 on GT (clean, operator exact)",
        "device": str(dev),
        "n_samples": n_done,
    }
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "meta": meta}, indent=2))
    with open(out_dir / "summary.md", "w") as f:
        f.write(f"### HF NYU val / x4 (n={n_done}, data_range=10m, border=6)\n\n")
        f.write("| method | RMSE | MAE | PSNR | SSIM | boundary RMSE | edge F1 | latency ms |\n|---|---|---|---|---|---|---|---|\n")
        for mname, s in summary.items():
            f.write(f"| {mname} | {s['rmse']:.4f} | {s['mae']:.4f} | {s['psnr']:.2f} | {s['ssim']:.4f} | "
                    f"{s['boundary_rmse']:.4f} | {s['edge_f1']:.3f} | {s['latency_ms']:.1f} |\n")
    print(json.dumps(summary, indent=2))
    print(f"results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
