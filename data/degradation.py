# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/degradation.py
# Description: Exact low-resolution degradation protocol (downsampling operator S) and synthetic sensor noise injection.
"""Low-resolution degradation protocol.

Default protocol = the official SGNet/DKN synthetic protocol:
    LR = PIL bicubic resize of the HR ground-truth depth to (W//s, H//s)
which is the *known forward operator* ``S`` (exposed via ``forward_operator``) that
Rishita's measurement-consistency step may use. Noise and holes are optional extras
for the "sparse / noisy / incomplete" experiments; once they are enabled, the
operator is no longer exactly S (documented in the returned dict).

Order of operations (always): downsample -> add noise -> punch holes.
Invalid LR pixels are stored as 0.0 AND flagged False in ``lr_valid_mask``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from .preprocessing import fill_invalid_nearest, to_depth_float

SUPPORTED_MODES = ("bicubic", "bilinear", "nearest")
_PIL = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}


@dataclass(frozen=True)
class DegradationConfig:
    mode: str = "bicubic"        # downsampling operator S
    noise_std: float = 0.0       # Gaussian noise std (depth units, or fraction if relative)
    noise_relative: bool = False # True -> std = noise_std * depth value
    hole_ratio: float = 0.0      # fraction of LR pixels dropped at random
    n_block_holes: int = 0       # number of rectangular holes
    block_frac: float = 0.15     # block side as a fraction of LR side
    seed: int = 0                # base seed; per-sample seed = seed + sample index

    @property
    def is_clean(self) -> bool:
        return self.noise_std == 0.0 and self.hole_ratio == 0.0 and self.n_block_holes == 0

    def as_dict(self) -> dict:
        return asdict(self)


def downsample(depth_hr: np.ndarray, scale: int, mode: str = "bicubic") -> np.ndarray:
    """Forward operator S: HR depth (H, W) -> LR depth (H//s, W//s), float32.

    Uses PIL on float32 ('F' mode) exactly like the official loaders."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_MODES}; got '{mode}'")
    d = to_depth_float(depth_hr)
    H, W = d.shape
    if scale < 1 or H % scale or W % scale:
        raise ValueError(f"depth {H}x{W} is not divisible by scale {scale}; modcrop first.")
    img = Image.fromarray(np.ascontiguousarray(d), mode="F")
    out = img.resize((W // scale, H // scale), _PIL[mode])
    return np.asarray(out, dtype=np.float32).copy()


def forward_operator(scale: int, mode: str = "bicubic") -> Callable[[np.ndarray], np.ndarray]:
    """Return S as a callable ``S(depth_hr) -> depth_lr`` (noise/hole free)."""
    return lambda depth_hr: downsample(depth_hr, scale, mode)


def _add_holes(valid: np.ndarray, rng: np.random.Generator, cfg: DegradationConfig) -> np.ndarray:
    valid = valid.copy()
    h, w = valid.shape
    if cfg.hole_ratio > 0:
        valid &= rng.random((h, w)) >= cfg.hole_ratio
    for _ in range(cfg.n_block_holes):
        bh = max(1, int(round(h * cfg.block_frac)))
        bw = max(1, int(round(w * cfg.block_frac)))
        y = int(rng.integers(0, max(1, h - bh + 1)))
        x = int(rng.integers(0, max(1, w - bw + 1)))
        valid[y : y + bh, x : x + bw] = False
    return valid


def degrade(
    depth_hr: np.ndarray,
    scale: int,
    cfg: Optional[DegradationConfig] = None,
    valid_hr: Optional[np.ndarray] = None,
    sample_index: int = 0,
) -> Dict[str, object]:
    """Create the LR measurement from an HR ground truth.

    Returns dict(lr_depth, lr_valid_mask, lr_depth_clean, operator_exact, config).
    ``depth_hr`` / ``valid_hr`` are never modified.
    """
    cfg = cfg or DegradationConfig()
    d = to_depth_float(depth_hr)
    if valid_hr is not None and not valid_hr.all():
        # GT with holes: downsample a nearest-filled copy; LR pixel valid if >=50% of
        # its HR footprint is valid.
        src = fill_invalid_nearest(d, valid_hr)
        h, w = d.shape[0] // scale, d.shape[1] // scale
        frac = cv2.resize(valid_hr.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        lr_valid = frac >= 0.5
    else:
        src = d
        lr_valid = np.ones((d.shape[0] // scale, d.shape[1] // scale), dtype=bool)

    lr_clean = downsample(src, scale, cfg.mode)
    rng = np.random.default_rng(int(cfg.seed) + int(sample_index))

    lr = lr_clean.copy()
    if cfg.noise_std > 0:
        std = cfg.noise_std * np.abs(lr) if cfg.noise_relative else cfg.noise_std
        lr = lr + rng.normal(0.0, 1.0, lr.shape).astype(np.float32) * std
        lr = np.maximum(lr, 0.0).astype(np.float32) if (lr_clean >= 0).all() else lr.astype(np.float32)
    lr_valid = _add_holes(lr_valid, rng, cfg)
    lr = np.where(lr_valid, lr, 0.0).astype(np.float32)

    return {
        "lr_depth": lr,
        "lr_valid_mask": lr_valid,
        "lr_depth_clean": lr_clean,
        "operator_exact": bool(cfg.is_clean and (valid_hr is None or valid_hr.all())),
        "config": cfg.as_dict(),
    }


def center_sample(depth: np.ndarray, scale: int) -> np.ndarray:
    """Sample the centre pixel of each ``scale`` x ``scale`` block (the real-depth
    centre-sampling protocol)."""
    d = np.asarray(depth, dtype=np.float32)
    return d[scale // 2 :: scale, scale // 2 :: scale].copy()


def holdout_mask(valid: np.ndarray, scales) -> np.ndarray:
    """Return ``valid`` with the centre-sampled pixels (for each scale) removed.

    Those pixels were fed to the model as input, so they must not be scored as
    unseen evaluation evidence."""
    mask = np.asarray(valid, dtype=bool).copy()
    for s in scales:
        mask[s // 2 :: s, s // 2 :: s] = False
    return mask
