"""Deterministic synthetic RGB-D scenes."""

import numpy as np


SCALES = (2, 4, 8)
KINDS = ("aligned", "textured", "misaligned", "large_hole", "thin", "noise")


def make_scene(kind, seed, scale):
    rng = np.random.default_rng(seed)
    height, width = 96, 128
    y, x = np.indices((height, width), dtype=np.float32)
    center = int(rng.integers(49, 77))
    edge = center + 5 * np.sin(y / 25 + rng.uniform(0, 2))
    foreground = x < edge
    if kind == "thin":
        foreground = np.abs(x - edge) < rng.uniform(1.0, 2.0)
    depth = np.where(foreground, 2 + 0.004 * y, 5 + 0.006 * x).astype(np.float32)
    rgb_mask = (x < edge + 6) if kind == "misaligned" else foreground
    intensity = np.where(rgb_mask, 0.24, 0.73)
    if kind in {"textured", "large_hole", "noise"}:
        intensity += 0.16 * np.sin(x * 0.9) * np.cos(y * 0.25)
    rgb = np.stack((intensity, intensity * 0.86, intensity * 0.74), axis=-1)
    rgb = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # Centre sampling of the clean HR depth, followed by noise and dropout.
    offset = scale // 2
    lr = depth[offset::scale, offset::scale].copy()
    noise = 0.16 if kind == "noise" else 0.035
    lr += rng.normal(0, noise, lr.shape).astype(np.float32)
    valid = rng.random(lr.shape) >= (0.13 if kind == "noise" else 0.05)
    if kind == "large_hole":
        cy, cx = np.array(lr.shape) // 2
        valid[cy - 2:cy + 3, cx - 2:cx + 3] = False
    lr[~valid] = 0
    return lr, rgb, valid, depth


SPLIT_SEEDS = {
    "tuning": range(300, 302),
    "validation": range(350, 352),
    "test": range(400, 403),
    "test_extension": range(500, 520),
}
