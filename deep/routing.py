"""Input-quality-aware checkpoint routing for the deep branch.

Rule: original (clean-bicubic) checkpoint ONLY if the LR depth has no holes AND its noise
estimate is below a calibrated threshold; otherwise the robust fine-tuned checkpoint.

The robust checkpoint is optional. When it is not configured, the available
scale checkpoint is used for both input-quality routes.

Env:  SGNET_CKPT_X4 (original)   SGNET_CKPT_X4_ROBUST (fine-tuned)   [same for X8/X16]
Calibrate:  python -m deep.routing --calibrate --root <nyu_root>
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .inference import deep_hr_depth
from .settings import DeepModelUnavailableError

HOLE_THR = 0.005        # fraction of invalid LR pixels above which input is "incomplete"
NOISE_ABS_THR = 0.002   # metres; PLACEHOLDER -> set from --calibrate


def estimate_noise(lr_depth, valid_mask=None):
    """Returns (relative, absolute) robust noise sigma: MAD of (d - median3x3(d))."""
    d = np.asarray(lr_depth, dtype=np.float32)
    m = (d > 0) & np.isfinite(d) if valid_mask is None else (np.asarray(valid_mask, bool) & (d > 0))
    if m.sum() < 64:
        return float("inf"), float("inf")
    fill = np.where(m, d, np.median(d[m])).astype(np.float32)
    r = fill - cv2.medianBlur(fill, 3)
    inner = m.copy()
    inner[:2] = inner[-2:] = False
    inner[:, :2] = inner[:, -2:] = False
    r = r[inner]
    if r.size == 0:
        return float("inf"), float("inf")
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(sigma / max(float(np.median(d[m])), 1e-6)), float(sigma)


def assess_input(lr_depth, valid_mask=None) -> dict:
    d = np.asarray(lr_depth, dtype=np.float32)
    m = (d > 0) & np.isfinite(d) if valid_mask is None else (np.asarray(valid_mask, bool) & (d > 0))
    hole = 1.0 - float(m.mean())
    rel, ab = estimate_noise(d, m)
    clean = hole <= HOLE_THR and ab <= NOISE_ABS_THR
    return {"hole_ratio": hole, "noise_rel": rel, "noise_abs": ab, "route": "original" if clean else "robust"}


def _require_ckpt(env_var: str, scale: int, label: str) -> str:
    """Resolve an env-var checkpoint path and raise clearly if it is missing."""
    p = os.environ.get(env_var)
    if not p:
        raise DeepModelUnavailableError(
            f"Routed deep branch requires {env_var} (the {label} x{scale} checkpoint). "
            f"Set the environment variable to the path of the .pth file."
        )
    if not os.path.isfile(p):
        raise DeepModelUnavailableError(
            f"{env_var}={p!r} does not exist (the {label} x{scale} checkpoint)."
        )
    return p


def _ckpt_for(scale: int, route: str) -> str:
    """Return the best available checkpoint for the selected input-quality route."""
    if route == "robust" and os.environ.get(f"SGNET_CKPT_X{scale}_ROBUST"):
        return _require_ckpt(f"SGNET_CKPT_X{scale}_ROBUST", scale, "robust")

    explicit = os.environ.get(f"SGNET_CKPT_X{scale}")
    if explicit:
        return _require_ckpt(f"SGNET_CKPT_X{scale}", scale, "original")

    # Fall back to the standard scale discovery, which also supports the
    # repository's SGNet_X{scale}_R.pth files.
    from .deep_model import resolve_checkpoint

    return str(resolve_checkpoint(scale))


def deep_hr_depth_routed(lr_depth, hr_rgb, valid_mask=None, scale=4, **kw):
    """Route to original or robust checkpoint based on input quality.

    Pass the ORIGINAL LR validity mask (before any hole filling).
    Returns (prediction, route_info) where route_info is a per-call dict
    with keys: hole_ratio, noise_rel, noise_abs, route, ckpt_path.
    """
    info = assess_input(lr_depth, valid_mask)
    ckpt = _ckpt_for(scale, info["route"])
    info["ckpt_path"] = ckpt
    out = deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale, ckpt_path=ckpt, **kw)
    return out, info


def deep_hr_depth_auto(lr_depth, hr_rgb, valid_mask=None, scale=4, **kw):
    """Same signature/return as deep_hr_depth (bare array).

    Pass the ORIGINAL mask (before any hole filling). Routing metadata is
    returned as the second element of the tuple from deep_hr_depth_routed;
    use that function directly when you need per-request metadata.
    """
    out, _ = deep_hr_depth_routed(lr_depth, hr_rgb, valid_mask, scale, **kw)
    return out


def _calibrate(root, start=0, n=75, scale=4, split="val"):
    """Calibrate on val[start:start+n]; evaluate the router on a DISJOINT range."""
    from data.dataset import NYUv2Dataset
    from data.degradation import DegradationConfig
    conds = {"clean": {}, "noise0.005": {"noise_std": 0.005}, "noise0.01": {"noise_std": 0.01},
             "noise0.02": {"noise_std": 0.02}, "holes0.1": {"hole_ratio": 0.1}}
    for name, kw in conds.items():
        ds = NYUv2Dataset(root, split=split, scale=scale, degradation=DegradationConfig(seed=0, **kw))
        v = [assess_input(ds[i]["lr_depth"], ds[i]["lr_valid_mask"])["noise_abs"]
             for i in range(start, min(start + n, len(ds)))]
        print(f"{name:12s} noise_abs(m) p5={np.percentile(v,5):.5f}  p50={np.median(v):.5f}  p95={np.percentile(v,95):.5f}")
    print("Set NOISE_ABS_THR between clean p95 and noise0.01 p5 (noise0.005 is ~tie between models; ignore).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=75)
    a = ap.parse_args()
    if a.calibrate:
        _calibrate(a.root, a.start, a.n)