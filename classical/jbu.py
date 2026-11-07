"""Joint bilateral upsampling (JBU) with RGB guidance and optional depth gating.

This is the classical edge-preserving upsampling kernel. It splats every valid
low-resolution depth sample into the high-resolution grid, gating the splat on RGB
colour similarity (and, optionally, on a depth-discontinuity term) so that depth is
never smeared across foreground/background boundaries and RGB texture is not copied
into depth.
"""

from __future__ import annotations

import numpy as np

from .hole_filling import prepare_depth


def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return float32 (H, W, 3) RGB in [0, 1]. Accepts uint8 / uint16 / float."""
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"RGB must be (H, W, 3); got shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("hr_rgb contains NaN/Inf.")
    mx = float(a.max()) if a.size else 0.0
    if mx > 1.0 + 1e-4:
        raise ValueError(f"hr_rgb values must be in [0, 1]; got max {mx:.1f}.")
    return np.clip(a, 0.0, 1.0)


def joint_bilateral_upsample(
    lr_depth,
    valid_mask=None,
    hr_rgb=None,
    *,
    radius: int = 1,
    color_thresh: float = 0.08,
    use_depth: bool = False,
    depth_thresh: float = 0.1,
) -> np.ndarray:
    """Upsample ``lr_depth`` to the resolution of ``hr_rgb``.

    Parameters
    ----------
    lr_depth   : (h, w) float32 depth; invalid pixels are <= 0, NaN or Inf.
    valid_mask : optional (h, w) bool mask (True = measured). Non-finite/<=0 is
                 always treated as invalid regardless of this mask.
    hr_rgb     : (H, W, 3) RGB guidance, uint8 or float32.
    radius     : LR-space splat radius (1 -> a 3x3 LR neighbourhood).
    color_thresh : hard colour gate on max channel difference in [0, 1].
    use_depth  : also gate on a depth-discontinuity term (adaptive JBU).
    depth_thresh : relative depth gate used when ``use_depth`` is True.

    Returns
    -------
    (H, W) float32 depth, same units as ``lr_depth``. Pixels that receive no
    support fall back to nearest-neighbour upsampling.
    """
    if hr_rgb is None:
        raise ValueError("hr_rgb is required for joint bilateral upsampling.")
    depth, valid = prepare_depth(lr_depth, valid_mask)
    rgb = to_float_rgb(hr_rgb)
    H, W = rgb.shape[:2]
    h, w = depth.shape
    if H % h or W % w:
        raise ValueError(f"RGB {H}x{W} is not an integer multiple of depth {h}x{w}.")
    scale = H // h

    nearest = np.repeat(np.repeat(depth, scale, axis=0), scale, axis=1)
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    ys, xs = np.where(valid)
    for y, x in zip(ys.tolist(), xs.tolist()):
        d = float(depth[y, x])
        cy = y * scale + scale // 2
        cx = x * scale + scale // 2
        ref = rgb[cy, cx]
        y0 = max(0, y * scale - radius * scale)
        y1 = min(H, (y + radius + 1) * scale)
        x0 = max(0, x * scale - radius * scale)
        x1 = min(W, (x + radius + 1) * scale)

        region = rgb[y0:y1, x0:x1]
        gate = np.abs(region - ref).max(axis=2) < color_thresh
        if use_depth:
            dregion = nearest[y0:y1, x0:x1]
            gate &= np.abs(dregion - d) <= depth_thresh * np.maximum(np.abs(dregion), abs(d))

        sub_acc = acc[y0:y1, x0:x1]
        sub_w = wsum[y0:y1, x0:x1]
        sub_acc[gate] += d
        sub_w[gate] += 1.0

    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), nearest)
    return out.astype(np.float32)
