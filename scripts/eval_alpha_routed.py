"""Compare classical, routed-deep, and hybrid on identical validation inputs; re-tune alpha.

Both checkpoint env-vars must be set:
  $env:SGNET_CKPT_X4        = <original .pth>
  $env:SGNET_CKPT_X4_ROBUST = <runs/x4_realistic/best.pth>

  python -m scripts.eval_alpha_routed ^
      --root "C:\\Users\\raksh\\Downloads\\nyu_data\\nyu_data" ^
      --start 0 --n 30 --scale 4

  # With degradation:
  python -m scripts.eval_alpha_routed --noise-std 0.01 --hole-ratio 0.1 --n-block-holes 2

  # Quick smoke-check (2 images) before a full sweep:
  python -m scripts.eval_alpha_routed --smoke --hole-ratio 0.3 --n-block-holes 2

Alpha sweep covers 0.0 -> 1.0 (inclusive).

EVALUATION DOMAINS
------------------
All masks are derived from a single gt_base that has the standard 6-pixel border
crop applied first (matching the official SGNet / eval.metrics protocol):

    gt_base = (vgt & isfinite(gt)) with border=6 zeroed

Primary (per-method own mask):
    mask = gt_base & isfinite(pred)
  If any gt_base pixel has a nonfinite prediction, RMSE is flagged FAIL/nan
  rather than silently dropping those pixels.  The nonfinite rate is reported
  explicitly for every method.

Shared-intersection (diagnostic only):
    shared = gt_base & isfinite(Dc) & isfinite(Dd)
  Used only to report coverage (total eval pixels, fraction retained).
  NOT used as the primary metric domain.

RAW BLEND
---------
    fused_raw = (1 - alpha) * Dd + alpha * Dc   (no substitution, no clipping)

Endpoints return copies to avoid 0*NaN arithmetic:
    alpha=0  =>  Dd.copy()
    alpha=1  =>  Dc.copy()
Intermediate alphas use the arithmetic blend.

fuse_fixed is reported as a SEPARATE stage ("fuse_prod") so its invalid-pixel
repair behaviour is visible and not confused with the raw blend.
Consistency correction is a third separate stage on top of raw blend.

WHY fuse_fixed(alpha=1) != standalone classical
-----------------------------------------------
fuse_fixed() applies an invalid-pixel substitution step before blending:

    valid_c = isfinite(Dc) & (Dc > 0)
    Dc_patched = where(valid_c, Dc, Dd)   # bad classical pixels <- deep values
    result = alpha * Dc_patched + (1-alpha) * Dd_patched

With heavy holes, classical_hr_depth produces zero-valued pixels at unfilled
hole locations (fill_small_holes only fills pixels with >=5 valid 3x3 neighbours;
large block holes remain as 0.0, and bicubic upsampling smears those zeros
outward). Those pixels satisfy Dc <= 0, so valid_c is False there, and
fuse_fixed replaces them with Dd values before the blend.

At alpha=1 the result is therefore Dd at every pixel where Dc<=0, not Dc.
The standalone classical eval includes those zero pixels (isfinite(0) is True)
and penalises them; fuse_fixed silently repairs them first.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from classical import classical_hr_depth
from data.dataset import NYUv2Dataset
from data.degradation import DegradationConfig
from deep.routing import deep_hr_depth_routed
from eval.metrics import boundary_rmse as _boundary_rmse
from eval.metrics import rmse as _rmse
from hybrid.consistency import enforce_consistency
from hybrid.fusion import fuse_fixed  # production fusion — reported separately

ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
_REPORT = ("rmse", "boundary_rmse")
BORDER = 6  # standard SGNet border crop, matches eval.metrics.DEFAULT_BORDER


# ------------------------------------------------------------------ blend

def _raw_blend(Dc: np.ndarray, Dd: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend with NO substitution.  Endpoints return copies to avoid 0*NaN."""
    if alpha == 0.0:
        return Dd.copy()
    if alpha == 1.0:
        return Dc.copy()
    return ((1.0 - alpha) * Dd + alpha * Dc).astype(np.float32)


def _consistency(arr: np.ndarray, lr, mask, scale: int) -> np.ndarray:
    return enforce_consistency(arr, lr, mask, scale, mode="bicubic", strength=0.25)


# ------------------------------------------------------------------ masks / metrics

def _gt_base_mask(gt, vgt, border: int = BORDER) -> np.ndarray:
    """Evaluation base mask: vgt & isfinite(gt) with a border-pixel strip zeroed.

    This is the single source of truth for the evaluation domain.  Every
    primary and diagnostic mask is derived from this array, so the border
    crop is applied exactly once and is identical for all methods and alphas.
    """
    m = np.asarray(vgt, dtype=bool) & np.isfinite(np.asarray(gt, dtype=np.float32))
    if border > 0:
        m[:border, :]  = False
        m[-border:, :] = False
        m[:, :border]  = False
        m[:, -border:] = False
    return m


def _method_mask(pred, gt_base: np.ndarray) -> np.ndarray:
    """Primary eval mask for one prediction: gt_base & isfinite(pred)."""
    return gt_base & np.isfinite(np.asarray(pred, dtype=np.float32))


def _nonfinite_rate(pred, gt_base: np.ndarray) -> float:
    """Fraction of gt_base pixels where pred is nonfinite."""
    n = int(gt_base.sum())
    if n == 0:
        return float("nan")
    return float((~np.isfinite(np.asarray(pred, dtype=np.float32)))[gt_base].mean())


def _metrics_own(pred, gt, dr, gt_base: np.ndarray) -> dict:
    """Primary metrics on the method's own mask.

    If any valid-GT pixel has a nonfinite prediction, rmse and boundary_rmse
    are returned as nan and 'nonfinite_fail' is set True.
    """
    p = np.asarray(pred, dtype=np.float32)
    nf_rate = _nonfinite_rate(p, gt_base)
    fail = nf_rate > 0.0
    own_m = _method_mask(p, gt_base)
    return {
        "rmse":           float("nan") if fail else _rmse(p, gt, mask=own_m),
        "boundary_rmse":  float("nan") if fail else _boundary_rmse(p, gt, dr, mask=own_m),
        "nonfinite_rate": nf_rate,
        "nonfinite_fail": fail,
    }


def _metrics_shared(pred, gt, dr, shared_m: np.ndarray) -> dict:
    """Diagnostic metrics on the shared-intersection mask."""
    p = np.asarray(pred, dtype=np.float32)
    return {
        "rmse":          _rmse(p, gt, mask=shared_m),
        "boundary_rmse": _boundary_rmse(p, gt, dr, mask=shared_m),
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), float) and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def _any_fail(rows, key="nonfinite_fail"):
    return any(r.get(key, False) for r in rows)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=None, help="NYU data root (default: datasets/nyu_data)")
    ap.add_argument("--start", type=int, default=0, help="first val index (default: 0)")
    ap.add_argument("--n", type=int, default=30, help="number of images (default: 30)")
    ap.add_argument("--scale", type=int, default=4, help="SR scale (default: 4)")
    ap.add_argument("--out", default="results/alpha_routed", help="output directory")
    ap.add_argument("--smoke", action="store_true",
                    help="run only 2 images as a quick sanity check before the full sweep")
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Gaussian noise std in depth units (default: 0.0 = clean)")
    ap.add_argument("--hole-ratio", type=float, default=0.0,
                    help="fraction of LR pixels randomly dropped (default: 0.0)")
    ap.add_argument("--n-block-holes", type=int, default=0,
                    help="number of rectangular block holes (default: 0)")
    ap.add_argument("--block-frac", type=float, default=0.15,
                    help="block side as fraction of LR side (default: 0.15)")
    a = ap.parse_args(argv)

    deg = DegradationConfig(
        seed=0,
        noise_std=a.noise_std,
        hole_ratio=a.hole_ratio,
        n_block_holes=a.n_block_holes,
        block_frac=a.block_frac,
    )
    ds = NYUv2Dataset(a.root, split="val", scale=a.scale, degradation=deg)
    n_images = 2 if a.smoke else min(a.n, len(ds) - a.start)
    end = a.start + n_images

    # Per-image rows — primary (own mask) and shared (diagnostic)
    classical_own, deep_own = [], []
    raw_own   = {alpha: [] for alpha in ALPHAS}
    prod_own  = {alpha: [] for alpha in ALPHAS}
    cons_own  = {alpha: [] for alpha in ALPHAS}

    classical_sh, deep_sh = [], []
    raw_sh    = {alpha: [] for alpha in ALPHAS}
    prod_sh   = {alpha: [] for alpha in ALPHAS}
    cons_sh   = {alpha: [] for alpha in ALPHAS}

    routes = Counter()
    alpha0_diffs, alpha1_diffs = [], []
    dc_nonpos_fracs = []

    # Coverage diagnostics
    total_eval_pixels = []      # post-border-crop eval pixel count per image
    shared_retained_fracs = []  # fraction of eval pixels in shared mask
    dc_nf_rates, dd_nf_rates = [], []

    # Accumulate (Dd, gt, gt_base_no_border) for the border-crop verification
    _verif_Dd:  list = []
    _verif_gt:  list = []
    _verif_vgt: list = []   # raw vgt before any border crop

    for i in range(a.start, end):
        s = ds[i]
        lr   = s["lr_depth"]
        rgb  = s["hr_rgb"]
        mask = s["lr_valid_mask"]   # original pre-fill LR mask, never modified
        gt   = s["depth_original"]
        vgt  = s["valid_mask_original"]
        dr   = s["data_range"]

        Dc = classical_hr_depth(lr, rgb, mask, a.scale)
        Dd, info = deep_hr_depth_routed(lr, rgb, mask, a.scale)
        routes[info["route"]] += 1

        # gt_base has the 6-pixel border zeroed; all masks below derive from it.
        gt_base  = _gt_base_mask(gt, vgt, border=BORDER)
        shared_m = gt_base & np.isfinite(Dc) & np.isfinite(Dd)

        n_eval = int(gt_base.sum())
        total_eval_pixels.append(n_eval)
        shared_retained_fracs.append(float(shared_m.sum() / n_eval) if n_eval else float("nan"))

        # Store raw arrays for the border-crop verification computed at the end.
        _verif_Dd.append(Dd)
        _verif_gt.append(gt)
        _verif_vgt.append(vgt)
        dc_nf_rates.append(_nonfinite_rate(Dc, gt_base))
        dd_nf_rates.append(_nonfinite_rate(Dd, gt_base))
        dc_nonpos_fracs.append(float((~(np.isfinite(Dc) & (Dc > 0))).mean()))

        classical_own.append(_metrics_own(Dc, gt, dr, gt_base))
        deep_own.append(_metrics_own(Dd, gt, dr, gt_base))
        classical_sh.append(_metrics_shared(Dc, gt, dr, shared_m))
        deep_sh.append(_metrics_shared(Dd, gt, dr, shared_m))

        for alpha in ALPHAS:
            raw  = _raw_blend(Dc, Dd, alpha)
            prod = fuse_fixed(Dc, Dd, alpha)
            cons = _consistency(raw, lr, mask, a.scale)

            raw_own[alpha].append(_metrics_own(raw, gt, dr, gt_base))
            prod_own[alpha].append(_metrics_own(prod, gt, dr, gt_base))
            cons_own[alpha].append(_metrics_own(cons, gt, dr, gt_base))

            raw_sh[alpha].append(_metrics_shared(raw, gt, dr, shared_m))
            prod_sh[alpha].append(_metrics_shared(prod, gt, dr, shared_m))
            cons_sh[alpha].append(_metrics_shared(cons, gt, dr, shared_m))

        # Endpoint invariants on copies — no 0*NaN risk
        alpha0_diffs.append(float(np.max(np.abs(_raw_blend(Dc, Dd, 0.0) - Dd))))
        alpha1_diffs.append(float(np.max(np.abs(_raw_blend(Dc, Dd, 1.0) - Dc))))

        print(
            f"  {i - a.start + 1}/{n_images}  route={info['route']}"
            f"  Dc_nonpos={dc_nonpos_fracs[-1]*100:.1f}%"
            f"  shared_retained={shared_retained_fracs[-1]*100:.1f}%",
            flush=True,
        )

    # ------------------------------------------------------------------ invariants
    max_a0 = float(np.max(alpha0_diffs))
    max_a1 = float(np.max(alpha1_diffs))
    a0_ok  = max_a0 < 1e-5
    a1_ok  = max_a1 < 1e-5

    best_raw  = min(ALPHAS, key=lambda al: _mean(raw_own[al],  "rmse"))
    best_cons = min(ALPHAS, key=lambda al: _mean(cons_own[al], "rmse"))

    # -------------------------------------------------------- border-crop verification
    # Recompute deep RMSE with border=6 (current protocol) and without (border=0)
    # using the same Dd predictions already collected.  This directly reproduces
    # the 4.844 cm (with border) vs 4.740 cm (without border) discrepancy.
    def _rmse_scalar(preds, gts, masks):
        """Mean RMSE across images given per-image (pred, gt, mask) triples."""
        vals = []
        for p, g, m in zip(preds, gts, masks):
            if not m.any():
                continue
            e = p[m].astype(np.float64) - g[m].astype(np.float64)
            vals.append(float(np.sqrt(np.mean(e * e))))
        return float(np.mean(vals)) if vals else float("nan")

    _masks_b6 = [
        _gt_base_mask(g, v, border=6) & np.isfinite(d)
        for d, g, v in zip(_verif_Dd, _verif_gt, _verif_vgt)
    ]
    _masks_b0 = [
        _gt_base_mask(g, v, border=0) & np.isfinite(d)
        for d, g, v in zip(_verif_Dd, _verif_gt, _verif_vgt)
    ]
    _deep_rmse_b6 = _rmse_scalar(_verif_Dd, _verif_gt, _masks_b6)
    _deep_rmse_b0 = _rmse_scalar(_verif_Dd, _verif_gt, _masks_b0)
    _mean_eval_b6 = float(np.mean([m.sum() for m in _masks_b6]))
    _mean_eval_b0 = float(np.mean([m.sum() for m in _masks_b0]))

    # ------------------------------------------------------------------ print
    def _fmt_rmse(rows, key="rmse"):
        v = _mean(rows, key)
        fail = _any_fail(rows)
        if fail:
            return "  FAIL/nan"
        return f"{v*100:9.3f}" if np.isfinite(v) else "      nan"

    def _fmt_nf(rows):
        v = _mean(rows, "nonfinite_rate")
        return f"{v*100:.2f}%" if np.isfinite(v) else "  nan"

    print(f"\n  -- border-crop verification (same Dd predictions, two border settings) --")
    print(f"  deep RMSE border=6 (current protocol): {_deep_rmse_b6*100:.3f} cm  "
          f"(mean eval px: {_mean_eval_b6:.0f})")
    print(f"  deep RMSE border=0 (no crop):          {_deep_rmse_b0*100:.3f} cm  "
          f"(mean eval px: {_mean_eval_b0:.0f})")
    print(f"  difference: {(_deep_rmse_b6 - _deep_rmse_b0)*100:+.3f} cm  "
          f"({'border=6 is higher' if _deep_rmse_b6 > _deep_rmse_b0 else 'border=6 is lower'})")

    col = 36
    print(f"\n{'method':{col}s}  {'RMSE(cm)':>9s}  {'bRMSE(cm)':>10s}  {'nf-rate':>8s}")
    print("-" * (col + 34))
    print(f"{'classical (own mask)':>{col}s}  "
          f"{_fmt_rmse(classical_own):>9s}  "
          f"{_fmt_rmse(classical_own,'boundary_rmse'):>10s}  "
          f"{_fmt_nf(classical_own):>8s}")
    print(f"{'routed_deep (own mask)':>{col}s}  "
          f"{_fmt_rmse(deep_own):>9s}  "
          f"{_fmt_rmse(deep_own,'boundary_rmse'):>10s}  "
          f"{_fmt_nf(deep_own):>8s}")

    print(f"\n  -- raw blend: (1-a)*Dd + a*Dc  [own mask per alpha] --")
    for alpha in ALPHAS:
        tag  = "  <-- best" if alpha == best_raw else ""
        note = ""
        if alpha == 0.0:
            note = f"  [== Dd.copy(), diff={max_a0:.1e}{'  OK' if a0_ok else '  FAIL'}]"
        elif alpha == 1.0:
            note = f"  [== Dc.copy(), diff={max_a1:.1e}{'  OK' if a1_ok else '  FAIL'}]"
        print(f"  raw  alpha={alpha:.2f}{'':14s}  "
              f"{_fmt_rmse(raw_own[alpha]):>9s}  "
              f"{_fmt_rmse(raw_own[alpha],'boundary_rmse'):>10s}  "
              f"{_fmt_nf(raw_own[alpha]):>8s}"
              f"{note}{tag}")

    print(f"\n  -- fuse_prod: fuse_fixed() [substitution active, own mask] --")
    for alpha in ALPHAS:
        note = ""
        if alpha == 1.0:
            delta = abs(_mean(prod_own[1.0], "rmse") - _mean(classical_own, "rmse"))
            note = f"  [vs classical: Δ={delta*100:.3f}cm]"
        elif alpha == 0.0:
            delta = abs(_mean(prod_own[0.0], "rmse") - _mean(deep_own, "rmse"))
            note = f"  [vs routed_deep: Δ={delta*100:.3f}cm]"
        print(f"  prod alpha={alpha:.2f}{'':14s}  "
              f"{_fmt_rmse(prod_own[alpha]):>9s}  "
              f"{_fmt_rmse(prod_own[alpha],'boundary_rmse'):>10s}  "
              f"{_fmt_nf(prod_own[alpha]):>8s}{note}")

    print(f"\n  -- raw blend + consistency (strength=0.25, own mask) --")
    for alpha in ALPHAS:
        tag = "  <-- best" if alpha == best_cons else ""
        print(f"  cons alpha={alpha:.2f}{'':14s}  "
              f"{_fmt_rmse(cons_own[alpha]):>9s}  "
              f"{_fmt_rmse(cons_own[alpha],'boundary_rmse'):>10s}  "
              f"{_fmt_nf(cons_own[alpha]):>8s}{tag}")

    # ------------------------------------------------------------------ diagnostic
    mean_eval  = float(np.mean(total_eval_pixels))
    mean_ret   = float(np.mean([v for v in shared_retained_fracs if np.isfinite(v)]))
    mean_dc_nf = float(np.mean([v for v in dc_nf_rates if np.isfinite(v)]))
    mean_dd_nf = float(np.mean([v for v in dd_nf_rates if np.isfinite(v)]))

    print(f"\n  -- shared-intersection diagnostic (NOT primary metric domain) --")
    print(f"  mean eval pixels per image (border={BORDER}): {mean_eval:.0f}")
    print(f"  mean Dc nonfinite rate over eval mask: {mean_dc_nf*100:.3f}%")
    print(f"  mean Dd nonfinite rate over eval mask: {mean_dd_nf*100:.3f}%")
    print(f"  mean shared-mask retention:            {mean_ret*100:.2f}% of eval pixels")
    print(f"  (shared mask = gt_base & isfinite(Dc) & isfinite(Dd))")
    print(f"  classical (shared):   RMSE={_mean(classical_sh,'rmse')*100:.3f}cm  "
          f"bRMSE={_mean(classical_sh,'boundary_rmse')*100:.3f}cm")
    print(f"  routed_deep (shared): RMSE={_mean(deep_sh,'rmse')*100:.3f}cm  "
          f"bRMSE={_mean(deep_sh,'boundary_rmse')*100:.3f}cm")

    print(f"\nroute distribution: {dict(routes)}  (n={n_images})")
    print(f"mean Dc non-positive fraction: {np.mean(dc_nonpos_fracs)*100:.2f}%")
    print(f"\nEndpoint invariants (raw blend, copy at endpoints):")
    print(f"  alpha=0 max|raw(0)-Dd|: {max_a0:.2e}  {'OK' if a0_ok else 'FAIL'}")
    print(f"  alpha=1 max|raw(1)-Dc|: {max_a1:.2e}  {'OK' if a1_ok else 'FAIL'}")
    if not a0_ok or not a1_ok:
        print("  WARNING: endpoint invariant violated.")

    print(f"\nBest alpha (raw blend, own mask):              {best_raw}")
    print(f"Best alpha (raw blend+consistency, own mask):  {best_cons}")

    if a.smoke:
        print("\n[smoke mode: 2 images only — re-run without --smoke for the full sweep]")
        return

    # ------------------------------------------------------------------ save
    def _ser_rows(rows_dict):
        return {
            str(alpha): {
                k: _mean(rows_dict[alpha], k) for k in _REPORT
            } | {
                "nonfinite_rate": _mean(rows_dict[alpha], "nonfinite_rate"),
                "any_fail": _any_fail(rows_dict[alpha]),
            }
            for alpha in ALPHAS
        }

    def _ser_sh(rows_dict):
        return {
            str(alpha): {k: _mean(rows_dict[alpha], k) for k in _REPORT}
            for alpha in ALPHAS
        }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "split": "val", "scale": a.scale, "start": a.start, "n": n_images,
            "border": BORDER,
            "mean_eval_pixels": mean_eval,
            "degradation": deg.as_dict(),
            "eval_domain": (
                f"gt_base = vgt & isfinite(gt) with {BORDER}-px border zeroed; "
                "primary = gt_base & isfinite(pred); "
                "shared = gt_base & isfinite(Dc) & isfinite(Dd) [diagnostic only]"
            ),
        },
        "primary": {
            "classical":   {k: _mean(classical_own, k) for k in _REPORT}
                         | {"nonfinite_rate": _mean(classical_own, "nonfinite_rate"),
                            "any_fail": _any_fail(classical_own)},
            "routed_deep": {k: _mean(deep_own, k) for k in _REPORT}
                         | {"nonfinite_rate": _mean(deep_own, "nonfinite_rate"),
                            "any_fail": _any_fail(deep_own)},
            "raw_blend":             _ser_rows(raw_own),
            "fuse_prod":             _ser_rows(prod_own),
            "raw_blend_consistency": _ser_rows(cons_own),
            "best_alpha_raw_blend":             best_raw,
            "best_alpha_raw_blend_consistency": best_cons,
        },
        "diagnostic_shared": {
            "mean_eval_pixels":           mean_eval,
            "mean_dc_nonfinite_rate":     mean_dc_nf,
            "mean_dd_nonfinite_rate":     mean_dd_nf,
            "mean_shared_retention_frac": mean_ret,
            "classical":   {k: _mean(classical_sh, k) for k in _REPORT},
            "routed_deep": {k: _mean(deep_sh, k) for k in _REPORT},
            "raw_blend":   _ser_sh(raw_sh),
            "fuse_prod":   _ser_sh(prod_sh),
            "raw_blend_consistency": _ser_sh(cons_sh),
        },
        "border_crop_verification": {
            "note": (
                "Same Dd predictions recomputed with two border settings to confirm "
                "that the 4.844 cm vs 4.740 cm discrepancy is caused solely by the "
                "border crop and not by any change to the deep branch."
            ),
            "deep_rmse_border6_cm":  round(_deep_rmse_b6 * 100, 4),
            "deep_rmse_border0_cm":  round(_deep_rmse_b0 * 100, 4),
            "difference_cm":         round((_deep_rmse_b6 - _deep_rmse_b0) * 100, 4),
            "mean_eval_pixels_b6":   _mean_eval_b6,
            "mean_eval_pixels_b0":   _mean_eval_b0,
        },
        "route_counts": dict(routes),
        "endpoint_invariants": {
            "alpha0_max_diff": max_a0, "alpha0_ok": a0_ok,
            "alpha1_max_diff": max_a1, "alpha1_ok": a1_ok,
        },
        "mean_dc_nonpos_fraction": float(np.mean(dc_nonpos_fracs)),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
