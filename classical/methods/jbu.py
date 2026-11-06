"""Joint bilateral upsampling from a local LR measurement neighborhood."""

import cv2
import numpy as np


def joint_bilateral_upsample(depth, valid, rgb, *, sigma_color=0.12,
                             sigma_space=0.85, radius=2, spatial_factor=None,
                             color_strength=None, protected_anchor=None, relative_jump=0.10):
    """Reconstruct HR depth with spatial and RGB weights in a bounded window."""
    height, width = rgb.shape[:2]
    rows, cols = depth.shape
    lr_rgb = cv2.resize(rgb, (cols, rows), interpolation=cv2.INTER_AREA)
    y = (np.arange(height, dtype=np.float32) + 0.5) * (rows / height) - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * (cols / width) - 0.5
    center_y, center_x = np.floor(y + 0.5).astype(int), np.floor(x + 0.5).astype(int)
    output = np.zeros((height, width), np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0)

    # Chunk rows to keep working memory bounded for large guidance images.
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        strip = rgb[start:stop]
        space = 2 * sigma_space**2
        if spatial_factor is not None:
            space *= spatial_factor[start:stop] ** 2
        color_scale = 1 / (2 * sigma_color**2)
        if color_strength is not None:
            color_scale *= color_strength[start:stop]
        anchor = protected_anchor[start:stop] if protected_anchor is not None else None
        total = np.zeros(strip.shape[:2], np.float32)
        weight_sum = np.zeros_like(total)
        spatial_total = np.zeros_like(total)
        spatial_sum = np.zeros_like(total)
        for dy in range(-radius, radius + 1):
            yy = center_y[start:stop] + dy
            iy = np.clip(yy, 0, rows - 1)
            wy = (y[start:stop] - yy.astype(np.float32)) ** 2
            for dx in range(-radius, radius + 1):
                xx = center_x + dx
                ix = np.clip(xx, 0, cols - 1)
                wx = (x - xx.astype(np.float32)) ** 2
                samples = np.ix_(iy, ix)
                support = valid[samples] & ((yy >= 0) & (yy < rows))[:, None]
                support &= ((xx >= 0) & (xx < cols))[None, :]
                color = strip - lr_rgb[samples]
                sample_depth = depth[samples]
                if anchor is not None:
                    jump = np.abs(sample_depth - anchor) > relative_jump * np.maximum(sample_depth, anchor)
                    support &= (anchor == 0) | ~jump
                distance = (wy[:, None] + wx[None, :]) / space
                spatial_weight = np.exp(-distance) * support
                spatial_total += spatial_weight * sample_depth
                spatial_sum += spatial_weight
                distance += np.sum(color**2, axis=2) * color_scale
                weight = np.exp(-distance) * support
                total += weight * sample_depth
                weight_sum += weight
        result = output[start:stop]
        guided = weight_sum > 1e-12
        np.divide(total, weight_sum, out=result, where=guided)
        # Strong color disagreement can suppress every measurement in a valid
        # neighborhood. Use the same support and anchor with spatial weights.
        np.divide(spatial_total, spatial_sum, out=result, where=~guided & (spatial_sum > 0))
    return output
