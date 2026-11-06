# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: deep/inference.py
# Description: Model forward pass execution, latency and GPU memory profiling, and deep_hr_depth wrapper implementation.
"""Public deep-branch API.

    deep_hr_depth(lr_depth, hr_rgb, valid_mask, scale=4) -> (H, W) float32 ndarray

Adapter between the frozen project contract (see data/preprocessing.py) and SGNet's
native contract (see deep/deep_model.py):

  1. validate geometry/scale, resolve the ORIGINAL validity mask (not modified)
  2. working copy: invalid LR pixels <- nearest valid value (SGNet has no mask input)
  3. per-image min-max normalisation over VALID pixels (the NYU training protocol)
  4. SGNet forward on (rgb, LR depth) in fp32 (cuFFT-in-fp16 only supports 2^n sizes)
  5. denormalise, clamp at 0, return float32 NumPy in the input depth unit

The output is the raw network prediction everywhere (also at originally-invalid
locations); the hybrid stage decides how much to trust it there.

CLI smoke test (Phase 1 gate):
    python -m deep.inference --scale 4 --height 480 --width 640
    python -m deep.inference --scale 4 --random-init     # no checkpoint, architecture/GPU check only
"""
from __future__ import annotations

import argparse
import time
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from data.preprocessing import (
    check_geometry,
    fill_invalid_nearest,
    minmax_denormalize,
    minmax_normalize,
    minmax_range,
    resolve_valid_mask,
    to_depth_float,
    to_float_rgb,
)

from .deep_model import get_model
from .settings import DeepModelUnavailableError, ensure_supported_scale

_MIN_TILE_LR = 16


# ------------------------------------------------------------------------------ device
def get_device(device: Optional[str] = None, require_cuda: bool = False) -> torch.device:
    """cuda if it *actually works*, else cpu (with a warning). The RTX 5050 is Blackwell
    (sm_120): it needs a PyTorch build with CUDA 12.8+ (pip index cu128 or newer)."""
    want = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda"):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            (torch.zeros(1, device=want) + 1).item()  # forces a real kernel launch
            return torch.device(want)
        except Exception as e:  # noqa: BLE001
            msg = (
                f"CUDA unusable ({e}). For an RTX 5050 (sm_120) install PyTorch built for "
                f"CUDA 12.8+: pip install torch torchvision --index-url "
                f"https://download.pytorch.org/whl/cu128"
            )
            if require_cuda or device:
                raise DeepModelUnavailableError(msg) from e
            warnings.warn(msg + " Falling back to CPU (slow).")
            return torch.device("cpu")
    return torch.device(want)


# ------------------------------------------------------------------------------ forward
def _forward(model, rgb_t: torch.Tensor, d_t: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out, _ = model((rgb_t, d_t))
    return out


def _forward_tiled(model, rgb_t, d_t, scale: int, tile_lr: int, margin_lr: int = 16) -> torch.Tensor:
    """Fallback for images that do not fit in VRAM. Tiles the LR grid, keeps a context
    margin, and stitches the tile cores. Not bit-identical to the full forward pass
    (SGNet has global FFT blocks), so results from this path are flagged by the caller."""
    _, _, h, w = d_t.shape
    out = torch.empty((1, 1, h * scale, w * scale), dtype=d_t.dtype, device=d_t.device)
    for y0 in range(0, h, tile_lr):
        for x0 in range(0, w, tile_lr):
            y1, x1 = min(h, y0 + tile_lr), min(w, x0 + tile_lr)
            ya, xa = max(0, y0 - margin_lr), max(0, x0 - margin_lr)
            yb, xb = min(h, y1 + margin_lr), min(w, x1 + margin_lr)
            o = _forward(
                model,
                rgb_t[:, :, ya * scale : yb * scale, xa * scale : xb * scale],
                d_t[:, :, ya:yb, xa:xb],
            )
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = o[
                :, :, (y0 - ya) * scale : (y1 - ya) * scale, (x0 - xa) * scale : (x1 - xa) * scale
            ]
    return out


def _run_model(model, rgb_t, d_t, scale: int, tile_lr: Optional[int]) -> Tuple[torch.Tensor, bool]:
    """Full-image pass; on CUDA OOM retry with shrinking tiles. Returns (out, used_tiling)."""
    if tile_lr:
        return _forward_tiled(model, rgb_t, d_t, scale, tile_lr), True
    try:
        return _forward(model, rgb_t, d_t), False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    h, w = d_t.shape[-2:]
    tile = max(_MIN_TILE_LR, (max(h, w) // 2 // 8) * 8)
    while True:
        try:
            out = _forward_tiled(model, rgb_t, d_t, scale, tile)
            warnings.warn(f"Deep branch ran in tiled mode (tile={tile} LR px): not identical to full-image inference.")
            return out, True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if tile <= _MIN_TILE_LR:
                raise DeepModelUnavailableError(
                    f"CUDA out of memory even with {_MIN_TILE_LR}px tiles for scale x{scale}."
                )
            tile = max(_MIN_TILE_LR, tile // 2)


# -------------------------------------------------------------------------------- API
def deep_hr_depth(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale: int = 4,
    *,
    norm_range: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    tile_lr: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    _allow_random_init: bool = False,
) -> np.ndarray:
    """Deep HR depth. See module docstring. Extra keyword arguments are optional:

    norm_range : (lo, hi) override for the min-max normalisation (default: valid LR min/max).
    device     : 'cuda' / 'cpu' (default: cuda if usable).
    tile_lr    : force tiled inference with this LR tile size (default: full image, tiled on OOM).
    """
    ensure_supported_scale(scale)
    lr = to_depth_float(lr_depth)
    rgb = to_float_rgb(hr_rgb)
    check_geometry(lr, rgb, scale)
    mask = resolve_valid_mask(lr, valid_mask)  # original mask, never altered

    work = fill_invalid_nearest(lr, mask)      # branch-specific copy
    lo, hi = minmax_range(work, mask, norm_range)
    d_norm = minmax_normalize(work, lo, hi)

    dev = get_device(device)
    model, _ = get_model(scale, dev, ckpt_path, allow_random_init=_allow_random_init)
    d_t = torch.from_numpy(d_norm)[None, None].to(dev)
    rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(dev)

    out, _ = _run_model(model, rgb_t, d_t, scale, tile_lr)
    pred = out[0, 0].float().cpu().numpy()
    if not np.isfinite(pred).all():
        raise DeepModelUnavailableError("SGNet produced NaN/Inf; check input range and checkpoint.")
    return np.maximum(minmax_denormalize(pred, lo, hi), 0.0).astype(np.float32)


# --------------------------------------------------------------------------- profiling
def profile_inference(
    lr_depth,
    hr_rgb,
    valid_mask=None,
    scale: int = 4,
    warmup: int = 2,
    runs: int = 10,
    device: Optional[str] = None,
    _allow_random_init: bool = False,
) -> dict:
    """End-to-end wrapper latency (incl. pre/post-processing and host<->device copies)
    and peak GPU memory. Memory fields are None on CPU."""
    dev = get_device(device)
    kw = dict(scale=scale, device=str(dev), _allow_random_init=_allow_random_init)
    for _ in range(max(0, warmup)):
        deep_hr_depth(lr_depth, hr_rgb, valid_mask, **kw)
    is_cuda = dev.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        deep_hr_depth(lr_depth, hr_rgb, valid_mask, **kw)
        if is_cuda:
            torch.cuda.synchronize(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {
        "device": str(dev),
        "scale": scale,
        "lr_shape": tuple(np.asarray(lr_depth).shape[:2]),
        "runs": len(times),
        "latency_ms_mean": float(t.mean()),
        "latency_ms_median": float(np.median(t)),
        "latency_ms_p95": float(np.percentile(t, 95)),
        "peak_mem_allocated_mb": (torch.cuda.max_memory_allocated(dev) / 2**20) if is_cuda else None,
        "peak_mem_reserved_mb": (torch.cuda.max_memory_reserved(dev) / 2**20) if is_cuda else None,
    }


def _synthetic_inputs(height: int, width: int, scale: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = height - height % scale, width - width % scale
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = 1.0 + 3.0 * (xx / W) + 0.5 * np.sin(yy / 17.0)
    depth[H // 4 : H // 2, W // 4 : W // 2] = 0.8  # a foreground block => a real depth edge
    rgb = rng.random((H, W, 3)).astype(np.float32)
    lr = depth[::scale, ::scale].copy()
    return lr.astype(np.float32), rgb, np.ones_like(lr, dtype=bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SGNet smoke test: real forward pass, latency, peak memory.")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random-init", action="store_true", help="skip checkpoint (meaningless output; GPU/arch test)")
    ap.add_argument("--require-cuda", action="store_true")
    a = ap.parse_args(argv)

    dev = get_device(a.device, require_cuda=a.require_cuda)
    print(f"torch {torch.__version__} | device: {dev}" + (f" ({torch.cuda.get_device_name(dev)}, "
          f"{torch.cuda.get_device_properties(dev).total_memory / 2**30:.1f} GiB)" if dev.type == "cuda" else ""))
    lr, rgb, mask = _synthetic_inputs(a.height, a.width, a.scale)
    model, info = get_model(a.scale, dev, a.ckpt, allow_random_init=a.random_init)
    print(f"model: SGNet x{a.scale}, {info['params_m']} M params, {info['size_mb_fp32']} MB fp32, "
          f"pretrained={info['pretrained']}, ckpt={info['ckpt']} (sha256[:12]={info['ckpt_sha256_12']})")
    out = deep_hr_depth(lr, rgb, mask, a.scale, device=str(dev), ckpt_path=a.ckpt, _allow_random_init=a.random_init)
    print(f"forward OK: LR {lr.shape} -> HR {out.shape}, dtype {out.dtype}, range [{out.min():.3f}, {out.max():.3f}]")
    stats = profile_inference(lr, rgb, mask, a.scale, runs=a.runs, device=str(dev), _allow_random_init=a.random_init)
    for k in ("latency_ms_mean", "latency_ms_median", "latency_ms_p95", "peak_mem_allocated_mb", "peak_mem_reserved_mb"):
        v = stats[k]
        print(f"  {k}: {'n/a (CPU)' if v is None else f'{v:.1f}'}")
    if a.random_init:
        print("NOTE: --random-init => numbers are valid for latency/memory only, not for quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
