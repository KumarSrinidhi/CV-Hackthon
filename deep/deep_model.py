# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: deep/deep_model.py
# Description: Official deep model architecture (SGNet), checkpoint verification, and model wrapper definition.
"""SGNet model construction and checkpoint handling.

Verified facts about the official implementation (yanzq95/SGNet, Apache-2.0)
---------------------------------------------------------------------------
* Native forward:  ``out, out_grad = net((rgb, lr_depth))``
    rgb       : (N, 3, H,   W  ) float32, RGB, range [0, 1]
    lr_depth  : (N, 1, H/s, W/s) float32, *low-resolution* (NOT pre-upsampled),
                normalised to ~[0, 1]
    out       : (N, 1, H, W) normalised HR depth (bicubic skip connection inside)
* No validity-mask channel exists. Invalid pixels must be handled outside the model.
* One checkpoint per scale: x4 (num_feats=48), x8 (40), x16 (40). x2: none released.
* H and W must be divisible by the scale (official loaders modcrop).
* Task: guided depth super-resolution (trained on synthetic NYU-v2, bicubic-down LR).
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .settings import (
    SGNET_VARIANTS,
    DeepModelUnavailableError,
    ckpt_dir,
    ensure_supported_scale,
)
from .sgnet import SGNet

_MODEL_CACHE: Dict[Tuple[str, str, int, int], torch.nn.Module] = {}


def build_sgnet(scale: int) -> SGNet:
    ensure_supported_scale(scale)
    return SGNet(num_feats=SGNET_VARIANTS[scale]["num_feats"], kernel_size=3, scale=scale)


def resolve_checkpoint(scale: int, ckpt_path: Optional[str] = None) -> Path:
    """Locate the checkpoint for ``scale``.

    Order: explicit argument -> env SGNET_CKPT_X{scale} -> files in ckpt_dir()
    matching ``*X{scale}*.pth`` (``_R`` re-uploaded files preferred).
    """
    import os

    ensure_supported_scale(scale)
    if ckpt_path:
        p = Path(ckpt_path)
        if not p.is_file():
            raise DeepModelUnavailableError(f"Checkpoint not found: {p}")
        return p
    env = os.environ.get(f"SGNET_CKPT_X{scale}")
    if env:
        p = Path(env)
        if not p.is_file():
            raise DeepModelUnavailableError(f"SGNET_CKPT_X{scale} points to a missing file: {p}")
        return p
    d = ckpt_dir()
    cands = sorted(d.glob(f"*[Xx]{scale}*.pth")) if d.is_dir() else []
    # a name like 'X4' must not match 'X16'/'X48': require non-digit after the number
    import re

    pat = re.compile(rf"[Xx]{scale}(?!\d)")
    cands = [c for c in cands if pat.search(c.name)]
    if not cands:
        raise DeepModelUnavailableError(
            f"No SGNet x{scale} checkpoint found in '{d}'. Download the official *_R.pth "
            f"files (link in the SGNet README) and place e.g. SGNet_X{scale}_R.pth there, "
            f"or set SGNET_CKPT_DIR / SGNET_CKPT_X{scale}."
        )
    cands.sort(key=lambda c: (0 if c.stem.endswith("_R") else 1, c.name))
    return cands[0]


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj:
        raise DeepModelUnavailableError("Checkpoint does not contain a state_dict.")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first:  # old pickled checkpoints
        warnings.warn(
            f"weights_only load failed for {path.name} ({type(first).__name__}); retrying "
            f"with full pickle. Only do this for the official file."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_sgnet(
    scale: int,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    allow_random_init: bool = False,
) -> Tuple[SGNet, dict]:
    """Build SGNet for ``scale`` and load verified weights. Returns (model, info).

    ``allow_random_init`` is only for architecture/GPU smoke tests: it skips the
    checkpoint and the resulting outputs are meaningless.
    """
    device = device or torch.device("cpu")
    model = build_sgnet(scale)
    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "scale": scale,
        "num_feats": SGNET_VARIANTS[scale]["num_feats"],
        "params_m": round(n_params / 1e6, 3),
        "size_mb_fp32": round(n_params * 4 / 2**20, 2),
        "pretrained": False,
        "ckpt": None,
        "ckpt_sha256_12": None,
    }
    if not allow_random_init:
        path = resolve_checkpoint(scale, ckpt_path)
        sd = _extract_state_dict(_torch_load(path))
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise DeepModelUnavailableError(
                f"Checkpoint {path.name} does not match SGNet x{scale} "
                f"(num_feats={info['num_feats']}). missing={res.missing_keys[:5]} "
                f"(+{max(0, len(res.missing_keys) - 5)}), unexpected={res.unexpected_keys[:5]} "
                f"(+{max(0, len(res.unexpected_keys) - 5)}). Use the *_R.pth files from the "
                f"official release."
            )
        info.update(
            pretrained=True,
            ckpt=str(path),
            ckpt_sha256_12=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        )
    model.eval().requires_grad_(False)
    return model.to(device), info


def get_model(
    scale: int,
    device: torch.device,
    ckpt_path: Optional[str] = None,
    allow_random_init: bool = False,
) -> Tuple[SGNet, dict]:
    """Cached ``load_sgnet`` (keyed by scale/device/ckpt mtime)."""
    if allow_random_init:
        key = ("random", str(device), scale, 0)
    else:
        path = resolve_checkpoint(scale, ckpt_path)
        key = (str(path), str(device), scale, int(path.stat().st_mtime_ns))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_sgnet(scale, ckpt_path, device, allow_random_init)
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def verify_checkpoints(scales=(4, 8, 16)) -> Dict[int, dict]:
    """Phase-0 feasibility check: which scales have a loadable checkpoint?"""
    out: Dict[int, dict] = {}
    for s in scales:
        try:
            _, info = load_sgnet(s)
            out[s] = {"status": "verified", **info}
        except Exception as e:  # noqa: BLE001
            out[s] = {"status": "unavailable", "reason": str(e)}
    return out
