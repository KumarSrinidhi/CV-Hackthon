"""One-off: paired bootstrap CI for classical blend_g0.4 vs bicubic and vs classical, val150 clean."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from classical.edge_upsampling import upsample_dense_depth
from classical.hole_filling import prepare_depth
from classical.jbu import to_float_rgb
from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from data.preprocessing import fill_invalid_nearest
from eval.metrics import rmse as rmse_fn
from scripts.task_b_classical import classical_smooth

ROOT = Path(__file__).resolve().parents[1]

ds = NYUv2Dataset(root=str(ROOT / "nyu_data" / "nyu_data"), split="val", scale=4,
                  degradation=DegradationConfig())
rows = {k: [] for k in ("bicubic", "classical", "blend")}
for i in range(150):
    s = ds[i]
    gt, valid = s["depth_original"], s["valid_mask_original"]
    lr, lrm, rgb = s["lr_depth"].copy(), s["lr_valid_mask"].copy(), s["hr_rgb"].copy()
    rgb_f = to_float_rgb(np.asarray(rgb))
    from eval.evaluate import _make_baseline
    bic = _make_baseline("bicubic")(lr, rgb, lrm, 4)
    cla = classical_smooth(lr, lrm, rgb_f, 4)
    d, v2 = prepare_depth(lr, lrm)
    d = d if v2.all() else fill_invalid_nearest(np.where(v2, d, np.float32(0)), v2)
    den = upsample_dense_depth(d, rgb_f, 4)
    blend = 0.6 * cla + 0.4 * den
    for name, p in (("bicubic", bic), ("classical", cla), ("blend", blend)):
        rows[name].append(rmse_fn(p, gt, valid, 6))
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/150", flush=True)

def paired(a, b):
    d = np.asarray(rows[a]) - np.asarray(rows[b])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(5000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

for a, b in (("blend", "bicubic"), ("blend", "classical"), ("classical", "bicubic")):
    m, lo, hi = paired(a, b)
    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
    print(f"{a} - {b}: mean {m:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  {sig}")
