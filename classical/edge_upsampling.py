"""Edge-aware interpolation for dense and masked low-resolution depth."""

import cv2
import numpy as np
from PIL import Image


def _pixel_centers(length, output_length, scale):
    coordinates = (np.arange(output_length, dtype=np.float32) + 0.5) / scale - 0.5
    lower_unclipped = np.floor(coordinates).astype(np.int32)
    fraction = coordinates - lower_unclipped
    lower = np.clip(lower_unclipped, 0, length - 1)
    upper = np.clip(lower + 1, 0, length - 1)
    return lower, upper, fraction


def _corner_views(values, size, scale):
    """Return LR corner samples at HR pixel-center coordinates."""
    width, height = size
    y0, y1, _ = _pixel_centers(values.shape[0], height, scale)
    x0, x1, _ = _pixel_centers(values.shape[1], width, scale)
    return (
        values[y0[:, None], x0[None, :]],
        values[y0[:, None], x1[None, :]],
        values[y1[:, None], x0[None, :]],
        values[y1[:, None], x1[None, :]],
    )


def upsample_dense_depth(depth, rgb, scale, measured=None):
    """Sharpen depth-supported transitions, with optional masked back-projection."""
    h, w = depth.shape
    height, width = rgb.shape[:2]
    size = width, height
    _, _, x_fraction = _pixel_centers(w, width, scale)
    _, _, y_fraction = _pixel_centers(h, height, scale)
    x = x_fraction[None, :]
    y = y_fraction[:, None]
    samples = _corner_views(depth, size, scale)
    near = cv2.min(cv2.min(samples[0], samples[1]), cv2.min(samples[2], samples[3]))
    far = cv2.max(cv2.max(samples[0], samples[1]), cv2.max(samples[2], samples[3]))
    transition = np.clip(((far - near) / far - 0.017) / 0.075, 0, 1)

    # Narrow the spatial transition only where the depth samples disagree.
    sharp_x = np.clip((x - 0.5) * (0.6 * scale) + 0.5, 0, 1)
    sharp_y = np.clip((y - 0.5) * (0.6 * scale) + 0.5, 0, 1)
    # Keep a little support for the opposite side so a matching RGB region can
    # move a depth edge away from the center of the interpolation cell.
    u = x + 0.95 * transition * (sharp_x - x)
    v = y + 0.95 * transition * (sharp_y - y)
    spatial_weights = ((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v)
    guide = rgb[scale // 2::scale, scale // 2::scale]
    distances = []
    for reference in _corner_views(guide, size, scale):
        diff = rgb - reference
        distances.append(np.einsum("ijk,ijk->ij", diff, diff))

    # At a clear color boundary, preserve the observed depth on each side.
    matching_color = 0.0025
    confident = (
        (transition == 1)
        & (np.minimum.reduce(distances) < matching_color)
        & (np.maximum.reduce(distances) > 1)
    )
    weights = []
    total = np.zeros_like(near)
    for distance, spatial in zip(distances, spatial_weights):
        weight = spatial / (1 + transition * (distance * 200) ** 2)
        weight[confident & (distance >= matching_color)] = 0
        total += weight
        weights.append(weight)
    unsupported = total == 0
    for weight, spatial in zip(weights, spatial_weights):
        np.copyto(weight, spatial, where=unsupported)
    total[unsupported] = 1
    inverse_total = 1 / total
    confident &= ~unsupported

    sharp = sum(weight * sample for weight, sample in zip(weights, samples)) * inverse_total
    smooth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
    out = smooth + transition * (sharp - smooth)

    # Iterative bicubic back-projection recovers detail lost during downsampling.
    # Filled holes provide an initial estimate and do not constrain the update.
    depth_min, depth_max = depth.min(), depth.max()
    inverse_total[confident] = 0
    for _ in range(8):
        projected = np.asarray(Image.fromarray(out).resize((w, h), Image.Resampling.BICUBIC))
        residual = depth - projected
        if measured is not None:
            residual *= measured
        correction = sum(weight * sample for weight, sample in
                         zip(weights, _corner_views(residual, size, scale)))
        if float(np.max(np.abs(correction))) < 1e-4:
            break
        out = np.clip(out + 1.1 * inverse_total * correction, depth_min, depth_max)
    return out.astype(np.float32, copy=False)


def upsample_masked_depth(depth, measured, rgb, scale):
    """Fill local and enclosed gaps while leaving unsupported regions zero."""
    if not measured.any():
        return np.zeros(rgb.shape[:2], dtype=np.float32)

    from data.preprocessing import fill_invalid_nearest

    filled = fill_invalid_nearest(depth, measured)
    result = upsample_dense_depth(filled, rgb, scale, measured)

    # A gap surrounded by measurements can be estimated from those surfaces.
    # For open gaps, restrict support to the original one-cell neighborhood.
    row_before = np.logical_or.accumulate(measured, axis=1)
    row_after = np.logical_or.accumulate(measured[:, ::-1], axis=1)[:, ::-1]
    col_before = np.logical_or.accumulate(measured, axis=0)
    col_after = np.logical_or.accumulate(measured[::-1], axis=0)[::-1]
    enclosed = row_before & row_after & col_before & col_after
    nearby = cv2.dilate(measured.astype(np.uint8), np.ones((3, 3), np.uint8))
    support = cv2.resize(nearby | enclosed, result.shape[::-1], interpolation=cv2.INTER_NEAREST)
    result[support == 0] = 0
    return result
