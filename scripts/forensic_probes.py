# Team Owner: Kumar
# Workstream: Deep Learning / Forensics
# Module: scripts/forensic_probes.py
# Description: Controlled root-cause probes A1, A3, B1, B3 (plan.txt §6).
"""Forensic probes:

    python scripts/forensic_probes.py a1 --tars datasets/val-000000.tar --out tmp/forensics
    python scripts/forensic_probes.py a3 --tars datasets/val-000000.tar --out tmp/forensics --samples 25
    python scripts/forensic_probes.py b1 --tars datasets/val-000000.tar --out tmp/forensics --samples 3
    python scripts/forensic_probes.py b3 --tars datasets/val-000000.tar --out tmp/forensics --samples 25

Probe A3 results are diagnostics; oracles are non-deployable upper-bound experiments
(plan.txt §6.A3). No irreducible-floor claim is made anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy import ndimage as ndi

from data.dataset import DepthSRDataset
from data.degradation import DegradationConfig, downsample
from data.preprocessing import fill_invalid_nearest, minmax_denormalize, minmax_normalize, minmax_range
from eval.metrics import compute_all_metrics, depth_discontinuity_mask
from scripts.eval_all3_hf import HFNYUValDataset


def _bicubic_up(lr: np.ndarray, scale: int) -> np.ndarray:
    h, w = lr.shape
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _radial_spectrum(img: np.ndarray) -> tuple:
    x = np.asarray(img, np.float64)
    x = x - x.mean()
    x = x * np.hanning(x.shape[0])[:, None] * np.hanning(x.shape[1])[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(x)))
    cy, cx = np.array(f.shape) // 2
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    tb = np.bincount(r.ravel(), f.ravel())
    cb = np.bincount(r.ravel())
    return tb / np.maximum(cb, 1)


# --------------------------------------------------------------------------- A1
def probe_a1(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # A1.1 empirical impulse response of the x4 PIL-bicubic downsampler
    N = 64
    impulse = np.zeros((N, N), np.float32)
    impulse[N // 2, N // 2] = 1.0
    lr = downsample(impulse, 4)
    ys, xs = np.nonzero(lr)
    report["impulse_support_lr_px"] = {
        "y_range": [int(ys.min()), int(ys.max())], "x_range": [int(xs.min()), int(xs.max())],
        "sum_of_weights": float(lr.sum()),
        "max_weight": float(lr.max()),
    }
    y0 = int(ys.min())
    report["impulse_kernel_rows"] = [[float(v) for v in row] for row in lr[y0:ys.max() + 1, int(xs.min()):int(xs.max()) + 1]]

    const = downsample(np.full((64, 64), 3.25, np.float32), 4)
    report["constant_preservation_max_err"] = float(np.abs(const - 3.25).max())

    # A1.2 sinusoid attenuation across frequency (cycles per HR pixel)
    # freqs are k/512 so the sinusoid is exactly periodic in HR (512 px) and LR (128 px)
    W = 512
    ks = [4, 8, 16, 24, 32, 40, 48, 52, 56, 64, 80, 96, 128, 160]
    atten, recon = {}, {}
    for k in ks:
        f = k / W
        t = np.arange(W, dtype=np.float64)
        sig = (1.0 + np.sin(2 * np.pi * f * t)).astype(np.float32)
        img = np.tile(sig, (W, 1))
        lr = downsample(img, 4)
        col = lr[0]
        # LR frequency of the primary tone is f*4 cyc/LR-px; fold aliases into [0, 0.5]
        f_fold = (f * 4.0) % 1.0
        f_fold = f_fold if f_fold <= 0.5 else 1.0 - f_fold
        spec = np.abs(np.fft.rfft(col - col.mean())) / len(col) * 2.0
        amp_lr = float(spec[int(round(f_fold * len(col)))])
        up = _bicubic_up(lr, 4)
        atten[f] = amp_lr
        recon[f] = float(np.sqrt(np.mean((up - img) ** 2)))
    report["sinusoid_lr_gain_by_freq"] = {str(f): v for f, v in atten.items()}
    report["sinusoid_roundtrip_rmse_by_freq"] = {str(f): v for f, v in recon.items()}

    # A1.2 step edge response
    step = np.zeros((128, 128), np.float32)
    step[:, 64:] = 1.0
    lr_step = downsample(step, 4)
    up_step = _bicubic_up(lr_step, 4)
    row = up_step[64]
    report["step_edge_overshoot"] = float(max(row.max() - 1.0, -row.min()))
    report["step_edge_transition_px"] = int(np.sum((row > 0.02) & (row < 0.98)))

    # A1.3 real scenes: radial power spectra HR vs LR (mapped to HR frequencies)
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    ratios = []
    for i in range(min(args.samples, len(ds))):
        s = ds[i]
        gt = s["depth_original"]
        lr = downsample(gt, 4)
        ps_hr = _radial_spectrum(gt)
        ps_lr = _radial_spectrum(lr)
        # map LR radius r to HR radius 4r; ratio array indexed by HR radius
        ratio = np.full(len(ps_hr), np.nan)
        top = min(len(ps_hr), 4 * len(ps_lr))
        for r in range(1, len(ps_lr) - 1):
            hr_r = 4 * r
            if hr_r < top:
                ratio[hr_r] = ps_lr[r] / max(ps_hr[hr_r], 1e-12)
        ratios.append(ratio)
    R = np.vstack(ratios)
    med = np.nanmedian(R, axis=0)
    # HR radius r corresponds to r / N cycles per pixel (N = radial FFT extent)
    n_px = ds[0]["depth_original"].shape[0]
    report["real_spectrum_gain_by_hr_freq"] = {
        f"r={r} ({r / (2 * n_px):.4f} cyc/px)": float(med[r])
        for r in [4, 8, 16, 32, 64, 96, 128, 160, 192] if r < len(med) and np.isfinite(med[r])
    }
    report["real_spectrum_note"] = (
        "LR spectrum mapped to HR radial frequency r_HR = 4 * r_LR; gain<1 means the "
        "operator attenuates that HR frequency band. Radial median over scenes."
    )

    with open(out / "a1_operator.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------------------- A3
def probe_a3(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    n = min(args.samples, len(ds))
    from deep.inference import deep_hr_depth

    res = {"a3a_identity": [], "a3b_roundtrip": [], "a3c1_gt_norm": [], "a3c2_spectral": []}

    for i in range(n):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        lr, lrm = s["lr_depth"], s["lr_valid_mask"]
        rng = s["data_range"]

        # A3a identity sanity: pred == gt through the metric plumbing
        m = compute_all_metrics(gt.astype(np.float32), gt, rng, valid)
        res["a3a_identity"].append(m)
        # preprocessing round trip on valid pixels
        work = fill_invalid_nearest(lr, lrm)
        lo, hi = minmax_range(work, lrm)
        rt = minmax_denormalize(minmax_normalize(work, lo, hi), lo, hi)
        res["a3a_identity"][-1]["fill_roundtrip_max_err_valid"] = float(np.abs(rt[lrm] - work[lrm]).max())
        res["a3a_identity"][-1]["identity_note"] = "pipeline sanity check only, NOT x4 capability"

        if i >= min(args.gpu_samples, n):
            continue

        # A3b operator round trip: bicubic-up vs SGNet on the same LR
        pred_bic = _bicubic_up(fill_invalid_nearest(lr, lrm), 4)
        pred_deep = deep_hr_depth(lr, s["hr_rgb"], lrm, 4)
        for name, pred in [("bicubic_up", pred_bic), ("sgnet", pred_deep)]:
            res["a3b_roundtrip"].append({"sample": s["name"], "recon": name,
                                         **compute_all_metrics(pred, gt, rng, valid)})

        # A3c-1 GT-range normalization ORACLE (non-deployable diagnostic)
        lo_gt = float(gt[valid].min())
        hi_gt = float(gt[valid].max())
        pred_oracle_norm = deep_hr_depth(lr, s["hr_rgb"], lrm, 4, norm_range=(lo_gt, hi_gt))
        res["a3c1_gt_norm"].append({"sample": s["name"],
                                    **compute_all_metrics(pred_oracle_norm, gt, rng, valid)})

        # A3c-2 spectral oracle (non-deployable diagnostic):
        #   pred_oracle = bicubic_up(S(gt)) + highpass(gt), lowpass sigma set so the
        #   Gaussian lowpass is -3 dB at the LR Nyquist (0.25 cyc/px): sigma = sqrt(ln2/(2 pi^2 f_c^2))
        f_c = 0.25
        sigma = float(np.sqrt(np.log(2) / (2 * np.pi**2 * f_c**2)))
        lowpass = cv2.GaussianBlur(gt, (0, 0), sigma)
        oracle = np.clip(_bicubic_up(downsample(gt, 4), 4) + (gt - lowpass), 0, None).astype(np.float32)
        res["a3c2_spectral"].append({"sample": s["name"], "sigma_px": sigma,
                                     **compute_all_metrics(oracle, gt, rng, valid)})
        if (i + 1) % 5 == 0:
            print(f"  a3 {i + 1}/{n}")

    def agg(rows, keys=("rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_precision", "edge_recall", "edge_f1")):
        if not rows:
            return {}
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}

    summary = {
        "a3a_identity_mean": agg([r for r in res["a3a_identity"]]),
        "a3a_max_fill_roundtrip_err": max(r["fill_roundtrip_max_err_valid"] for r in res["a3a_identity"]),
        "a3b_by_recon": {name: agg([r for r in res["a3b_roundtrip"] if r["recon"] == name])
                         for name in ("bicubic_up", "sgnet")},
        "a3c1_gt_norm_oracle_mean": agg(res["a3c1_gt_norm"]),
        "a3c1_deployable_reference_mean": agg([r for r in res["a3b_roundtrip"] if r["recon"] == "sgnet"]),
        "a3c2_spectral_oracle_mean": agg(res["a3c2_spectral"]),
        "labels": {
            "a3a": "pipeline sanity check, NOT x4 capability",
            "a3b": "controlled comparison under the bicubic operator, not a bound",
            "a3c": "GT-informed oracles, non-deployable upper-bound diagnostics",
            "a3d": "no irreducible-floor claim; see PERFORMANCE_ANALYSIS.md",
        },
    }
    with open(out / "a3_diagnostics.json", "w") as f:
        json.dump({"summary": summary, "rows": res}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- A2
def probe_a2(args) -> int:
    """Depth-dependence of noise parameterizations on real scenes.

    Models: absolute sigma (current default), relative sigma (sigma = c*z), and the
    Kinect-v1-style quadratic model sigma = c*z^2 (disparity-based structured light,
    Nguyen et al. 2012). NYU-v2 was captured with Kinect v1."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    rng = np.random.default_rng(0)
    bins = np.arange(0.5, 10.01, 0.5)

    def noise_map(z, model):
        if model == "abs_0.02":
            return np.full_like(z, 0.02)
        if model == "rel_0.02":
            return 0.02 * np.abs(z)
        if model == "quad_0.0015":
            return 0.0015 * z ** 2
        raise ValueError(model)

    curves = {}
    for model in ["abs_0.02", "rel_0.02", "quad_0.0015"]:
        per_bin = [[] for _ in bins]
        for i in range(min(args.samples, len(ds))):
            s = ds[i]
            gt, valid = s["depth_original"], s["valid_mask_original"]
            lr = s["lr_depth_clean"]
            sigma_lr = noise_map(lr, model)
            lr_noisy = np.maximum(lr + rng.normal(0, 1, lr.shape).astype(np.float32) * sigma_lr, 0)
            up = _bicubic_up(fill_invalid_nearest(lr_noisy, np.ones_like(lrm := s["lr_valid_mask"], bool)), 4)
            err2 = (up - gt) ** 2
            for b_idx, edge in enumerate(bins):
                m = valid & (np.abs(gt - edge) < 0.25)
                if m.sum() >= 50:
                    per_bin[b_idx].append(np.sqrt(np.mean(err2[m])))
        curves[model] = {f"{edge:.1f}m": float(np.nanmean(v)) if v else None
                         for edge, v in zip(bins, per_bin)}

    report = {
        "rmse_vs_depth_bin": curves,
        "note": ("Absolute noise under-covers far range for depth-dependent sensors; "
                 "relative and quadratic models grow with depth. NYU-v2's Kinect v1 "
                 "follows approximately the quadratic model."),
        "labels": {"usage": "informs the robustness-curriculum noise parameterization (plan.txt §8.5)"},
    }
    with open(out / "a2_noise.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(curves, indent=2))
    return 0



def _independent_edge_mask(d: np.ndarray, valid: np.ndarray, thr: float) -> np.ndarray:
    """Independently written edge detector: same definition, separate implementation."""
    v = valid & np.isfinite(d)
    e = np.zeros(d.shape, bool)
    dx = np.abs(d[:, 1:] - d[:, :-1]) > thr
    dy = np.abs(d[1:, :] - d[:-1, :]) > thr
    e[:, :-1] |= dx & v[:, :-1] & v[:, 1:]
    e[:, 1:] |= dx & v[:, :-1] & v[:, 1:]
    e[:-1, :] |= dy & v[:-1, :] & v[1:, :]
    e[1:, :] |= dy & v[:-1, :] & v[1:, :]
    return e


def _independent_dilate(m: np.ndarray, it: int) -> np.ndarray:
    r = ndi.uniform_filter(m.astype(np.float32), 2 * it + 1) > 0 if it else m
    return r


def probe_b1(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    from deep.inference import deep_hr_depth

    report = {"scenes": [], "threshold_sweep": [], "degenerate": {}}
    for i in range(min(args.samples, len(ds))):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        pred = deep_hr_depth(s["lr_depth"], s["hr_rgb"], s["lr_valid_mask"], 4)
        official = compute_all_metrics(pred, gt, s["data_range"], valid)

        m = valid & np.isfinite(pred) & np.isfinite(gt)
        m[:6], m[-6:], m[:, :6], m[:, -6:] = False, False, False, False
        e = (pred.astype(np.float64) - gt.astype(np.float64))[m]
        indep = {
            "rmse": float(np.sqrt(np.mean(e**2))),
            "mae": float(np.mean(np.abs(e))),
            "psnr": float(10 * np.log10(s["data_range"]**2 / np.mean(e**2))),
        }
        thr = 0.03 * s["data_range"]
        ge = _independent_edge_mask(gt, m, thr)
        pe = _independent_edge_mask(pred, m, thr)
        ge_d = _independent_dilate(ge, 1)
        pe_d = _independent_dilate(pe, 1)
        prec = float((pe & ge_d).sum() / pe.sum()) if pe.any() else 0.0
        rec = float((ge & pe_d).sum() / ge.sum()) if ge.any() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        band = _independent_dilate(ge, 2) & m
        indep["boundary_rmse"] = float(np.sqrt(np.mean((pred[band].astype(np.float64) - gt[band].astype(np.float64))**2)))
        indep["edge_precision"], indep["edge_recall"], indep["edge_f1"] = prec, rec, f1

        diffs = {k: float(indep[k] - official[k]) for k in
                 ["rmse", "mae", "psnr", "boundary_rmse", "edge_precision", "edge_recall", "edge_f1"]}
        report["scenes"].append({"sample": s["name"], "official": official,
                                 "independent": indep, "max_abs_diff": max(abs(v) for v in diffs.values()),
                                 "diffs": diffs})

    # threshold / tolerance sensitivity on the same scenes
    for rt in [0.01, 0.02, 0.03, 0.04, 0.05]:
        for tol in [0, 1, 2]:
            vals = []
            for i in range(min(args.samples, len(ds))):
                s = ds[i]
                pred = deep_hr_depth(s["lr_depth"], s["hr_rgb"], s["lr_valid_mask"], 4)
                mm = compute_all_metrics(pred, s["depth_original"], s["data_range"],
                                         s["valid_mask_original"], rel_thresh=rt, edge_tol=tol)
                vals.append(mm["edge_f1"])
            report["threshold_sweep"].append({"rel_thresh": rt, "edge_tol": tol,
                                              "mean_edge_f1": float(np.nanmean(vals))})

    # degenerate inputs
    gt = np.full((120, 160), 2.0, np.float32)
    v = np.ones_like(gt, bool)
    report["degenerate"]["constant_depth_edge_mask_empty"] = {
        "ge_count": int(depth_discontinuity_mask(gt, v, 10.0).sum()),
        "edge_f1_is_nan": bool(np.isnan(compute_all_metrics(gt, gt, 10.0, v)["edge_f1"])),
    }
    pred_nan = gt.copy()
    pred_nan[0, 0] = np.nan
    report["degenerate"]["nan_pred_masked_out_rmse_zero_elsewhere"] = compute_all_metrics(pred_nan, gt, 10.0, v)["rmse"]
    gt_e = np.zeros((120, 160), np.float32)
    gt_e[10:20, 10:20] = 5.0
    v_e = np.ones_like(gt_e, bool)
    report["degenerate"]["tiny_valid_after_border"] = compute_all_metrics(gt_e, gt_e, 10.0, v_e)["rmse"]

    with open(out / "b1_metric_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}, indent=2))
    for sc in report["scenes"]:
        print(f"  {sc['sample']}: max_abs_diff={sc['max_abs_diff']:.3e}")
    return 0


# --------------------------------------------------------------------------- B2
def probe_b2(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from classical.pipeline import classical_hr_depth
    from deep.inference import deep_hr_depth
    from hybrid.confidence import estimate_confidence
    from hybrid.consistency import enforce_consistency
    from data.degradation import degrade

    report = {"weight_sweep": {}, "pixel_preference": {}, "consistency": [], "oracle_selector": []}
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    n = min(args.samples, len(ds))
    ks = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    from data.degradation import forward_operator
    S = forward_operator(4, "bicubic")

    for regime, cfg in [("clean", DegradationConfig()),
                        ("holes", DegradationConfig(hole_ratio=0.1, seed=0))]:
        rows = {k: [] for k in ks}
        method_rmse = {"deep": [], "classical": [], "fused": []}
        worse_than_deep = []
        worse_than_classical = []
        oracle_rows = []
        cons_rows = []
        for i in range(n):
            s = ds[i]
            gt, valid = s["depth_original"], s["valid_mask_original"]
            deg = degrade(gt, 4, cfg, valid_hr=valid, sample_index=i)
            lr, lrm = deg["lr_depth"], deg["lr_valid_mask"]
            rgb = s["hr_rgb"]
            dc = classical_hr_depth(lr, rgb, lrm, 4)
            dd = deep_hr_depth(lr, rgb, lrm, 4)
            wc, wd = estimate_confidence(dc, dd, lr, lrm, rgb, 4)
            fused = enforce_consistency(wc * dc + wd * dd, lr, lrm, 4)
            method_rmse["deep"].append(masked_rmse(dd, gt, valid))
            method_rmse["classical"].append(masked_rmse(dc, gt, valid))
            method_rmse["fused"].append(masked_rmse(fused, gt, valid))
            for k in ks:
                wck = np.clip(wc * k, 0.0, 1.0)
                fk = enforce_consistency(wck * dc + (1 - wck) * dd, lr, lrm, 4)
                rows[k].append(masked_rmse(fk, gt, valid))
            worse_than_deep.append(float(np.mean((np.abs(fused - gt) > np.abs(dd - gt)) & valid)))
            worse_than_classical.append(float(np.mean((np.abs(fused - gt) > np.abs(dc - gt)) & valid)))
            better = (np.abs(dc - gt) <= np.abs(dd - gt)).astype(np.float32)
            oracle = better * dc + (1 - better) * dd
            oracle_rows.append(masked_rmse(oracle, gt, valid))
            before = float(np.sqrt(np.mean((S(fused) - lr)[lrm] ** 2)))
            after = float(np.sqrt(np.mean((S(enforce_consistency(fused, lr, lrm, 4)) - lr)[lrm] ** 2)))
            cons_rows.append({"sample": s["name"], "residual_before": before, "residual_after": after})
            if (i + 1) % 5 == 0:
                print(f"  b2 {regime} {i + 1}/{n}")
        report["weight_sweep"][regime] = {
            f"wc_scale={k}": {"mean_rmse": float(np.nanmean(rows[k])), "std": float(np.nanstd(rows[k]))}
            for k in ks
        }
        report["weight_sweep"][regime]["method_mean_rmse"] = {
            m: float(np.nanmean(v)) for m, v in method_rmse.items()
        }
        report["pixel_preference"][regime] = {
            "frac_pixels_fused_worse_than_deep": float(np.nanmean(worse_than_deep)),
            "frac_pixels_fused_worse_than_classical": float(np.nanmean(worse_than_classical)),
        }
        report["oracle_selector"].append({"regime": regime,
                                          "mean_rmse_oracle": float(np.nanmean(oracle_rows)),
                                          "note": "GT-informed per-pixel selector; diagnostic upper bound only"})
        report["consistency"] = cons_rows

    summary = {
        "weight_sweep": report["weight_sweep"],
        "pixel_preference": report["pixel_preference"],
        "oracle_selector": report["oracle_selector"],
        "consistency_residual": {
            "mean_before": float(np.nanmean([r["residual_before"] for r in report["consistency"]])),
            "mean_after": float(np.nanmean([r["residual_after"] for r in report["consistency"]])),
        },
        "labels": {"oracle_selector": "non-deployable diagnostic; never usable at inference"},
    }
    with open(out / "b2_fusion_audit.json", "w") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0



def probe_b3(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = HFNYUValDataset(4, DegradationConfig(), tar_paths=args.tars)
    from deep.inference import deep_hr_depth, profile_inference

    n = min(args.samples, len(ds))
    report = {"normalization_variants": {}, "holes_fill": [], "tiling": [], "precision": {}}

    for i in range(n):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        lr, lrm = s["lr_depth"], s["lr_valid_mask"]
        work = fill_invalid_nearest(lr, lrm)
        lo, hi = minmax_range(work, lrm)

        variants = {
            "valid_lr_minmax": deep_hr_depth(lr, s["hr_rgb"], lrm, 4),
            "pct_05_995": deep_hr_depth(lr, s["hr_rgb"], lrm, 4, norm_range=(
                float(np.percentile(work[lrm], 0.5)), float(np.percentile(work[lrm], 99.5)))),
            "fixed_0_10m": deep_hr_depth(lr, s["hr_rgb"], lrm, 4, norm_range=(0.0, 10.0)),
        }
        for vname, pred in variants.items():
            report["normalization_variants"].setdefault(vname, []).append(
                compute_all_metrics(pred, gt, s["data_range"], valid))
        if (i + 1) % 5 == 0:
            print(f"  b3 norm {i + 1}/{n}")

    # holes: error concentration near invalid LR pixels (deployable, clean+holes)
    cfg = DegradationConfig(hole_ratio=0.1, seed=0)
    from data.degradation import degrade
    for i in range(min(5, len(ds))):
        s = ds[i]
        gt, valid = s["depth_original"], s["valid_mask_original"]
        deg = degrade(gt, 4, cfg, valid_hr=valid, sample_index=i)
        lr, lrm = deg["lr_depth"], deg["lr_valid_mask"]
        pred = deep_hr_depth(lr, s["hr_rgb"], lrm, 4)
        invalid_hr = ~cv2.resize(lrm.astype(np.float32), (gt.shape[1], gt.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        near_hole = ndi.binary_dilation(invalid_hr, iterations=8) & ~invalid_hr & valid
        far = valid & ~ndi.binary_dilation(invalid_hr, iterations=8)
        report["holes_fill"].append({
            "sample": s["name"],
            "rmse_near_holes": masked_rmse(pred, gt, near_hole),
            "rmse_far_from_holes": masked_rmse(pred, gt, far),
            "rmse_overall": compute_all_metrics(pred, gt, s["data_range"], valid)["rmse"],
        })

    # tiling vs full on 2 samples
    for i in range(2):
        s = ds[i]
        pred_full = deep_hr_depth(s["lr_depth"], s["hr_rgb"], s["lr_valid_mask"], 4)
        pred_tiled = deep_hr_depth(s["lr_depth"], s["hr_rgb"], s["lr_valid_mask"], 4, tile_lr=32)
        d = np.abs(pred_full - pred_tiled)
        report["tiling"].append({
            "sample": s["name"], "tile_lr": 32,
            "max_abs_diff_m": float(d.max()),
            "rmse_diff_m": float(np.sqrt(np.mean((pred_tiled - s["depth_original"])**2))
                                 - np.sqrt(np.mean((pred_full - s["depth_original"])**2))),
        })

    # precision: attempt bf16 autocast on one forward
    try:
        import torch
        from deep.deep_model import get_model
        from data.preprocessing import to_float_rgb, check_geometry
        s = ds[0]
        lr, lrm = s["lr_depth"], s["lr_valid_mask"]
        work = fill_invalid_nearest(lr, lrm)
        lo, hi = minmax_range(work, lrm)
        d_norm = minmax_normalize(work, lo, hi)
        rgb = to_float_rgb(s["hr_rgb"])
        check_geometry(lr, rgb, 4)
        dev = torch.device("cuda")
        model, _ = get_model(4, dev)
        d_t = torch.from_numpy(d_norm)[None, None].to(dev)
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model((rgb_t, d_t))
        pred_bf16 = out[0, 0].float().cpu().numpy()
        pred_fp32 = deep_hr_depth(lr, s["hr_rgb"], lrm, 4)
        report["precision"]["bf16_autocast"] = {
            "ran": True,
            "max_abs_diff_vs_fp32_norm_space": float(np.abs(pred_bf16 - minmax_normalize(work, lo, hi)).max()),
        }
    except Exception as e:  # noqa: BLE001
        report["precision"]["bf16_autocast"] = {"ran": False, "error": f"{type(e).__name__}: {e}"}

    def agg(rows):
        return {k: float(np.nanmean([r[k] for r in rows])) for k in
                ("rmse", "mae", "psnr", "ssim", "boundary_rmse", "edge_f1")}

    summary = {
        "normalization_variants_mean": {k: agg(v) for k, v in report["normalization_variants"].items()},
        "holes_fill_mean": {
            "rmse_near_holes": float(np.nanmean([r["rmse_near_holes"] for r in report["holes_fill"]])),
            "rmse_far_from_holes": float(np.nanmean([r["rmse_far_from_holes"] for r in report["holes_fill"]])),
        },
        "tiling": report["tiling"],
        "precision": report["precision"],
        "labels": {"normalization": "deployable variants use inference-time information only; "
                                    "GT-range normalization is probe A3c-1"},
    }
    with open(out / "b3_inference_audit.json", "w") as f:
        json.dump({"summary": summary, "rows": report}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def masked_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 50:
        return float("nan")
    e = pred[m].astype(np.float64) - gt[m].astype(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["a1", "a2", "a3", "b1", "b2", "b3"])
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--out", default="tmp/forensics")
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--gpu-samples", type=int, default=25, help="A3: samples needing GPU inference")
    a = ap.parse_args(argv)
    return {"a1": probe_a1, "a2": probe_a2, "a3": probe_a3, "b1": probe_b1, "b2": probe_b2, "b3": probe_b3}[a.probe](a)


if __name__ == "__main__":
    raise SystemExit(main())
