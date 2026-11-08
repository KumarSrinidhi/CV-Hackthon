"""Prepare independent depth and support arrays for the classical branch."""

import cv2
import numpy as np


def prepare_depth(depth, valid_mask=None):
    """Return positive finite depth and its effective measurement mask."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Depth must be a nonempty (H, W) array.")
    valid = np.isfinite(values) & (values > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != values.shape:
            raise ValueError("Validity mask must match the LR depth shape.")
        valid &= np.isfinite(mask) & (mask != 0)
    return np.where(valid, values, np.float32(0)), valid


def fill_small_holes(depth, valid, relative_span=0.08):
    """Fill isolated holes with at least five agreeing 3x3 neighbors."""
    kernel = np.ones((3, 3), np.uint8)
    kwargs = {"borderType": cv2.BORDER_CONSTANT, "borderValue": 0}
    count = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    total = cv2.filter2D(depth, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    maximum = cv2.dilate(depth, kernel, **kwargs)
    minimum = cv2.erode(np.where(valid, depth, np.inf), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=float("inf"))
    fill = (~valid) & (count >= 5) & ((maximum - minimum) <= relative_span * maximum)
    result = depth.copy()
    result[fill] = total[fill] / count[fill]
    return result, valid | fill


def fill_holes_nearest(depth, valid):
    """True nearest-valid fill via distance transform.

    Each invalid pixel receives the depth of its nearest valid neighbour
    (Euclidean distance in pixel space).  Valid pixels are copied exactly;
    depth units and range are preserved by construction — no normalisation,
    no rescaling, no overshoot possible.

    ``valid`` is never modified; callers keep the original mask to distinguish
    in-hole from out-of-hole pixels when reporting errors.

    Implementation: cv2.distanceTransformWithLabels with DIST_LABEL_PIXEL
    treats every source pixel as its own label, so two adjacent valid pixels
    with different depths get different labels even when they belong to the
    same connected region.  Each label maps 1-to-1 to one source pixel, giving
    a correct per-pixel nearest-source lookup.

    DIST_LABEL_CCOMP must not be used here: it assigns one label per connected
    component, so all valid pixels that touch each other share a single label
    and the LUT would map every hole pixel in that Voronoi cell to whichever
    valid pixel happened to write last — an arbitrary, order-dependent result.
    """
    if valid.all():
        return depth.copy()
    if not valid.any():
        return depth.copy()
    # Valid pixels are sources (0); invalid pixels are unknown (255).
    src = np.where(valid, np.uint8(0), np.uint8(255))
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    # labels[y, x] is the 1-based sequential index of the nearest source pixel
    # in raster order.  Build a label -> depth lookup from the valid pixels.
    n_labels = int(labels.max()) + 1
    label_to_depth = np.zeros(n_labels, np.float32)
    ys, xs = np.where(valid)
    label_to_depth[labels[ys, xs]] = depth[ys, xs]
    result = label_to_depth[labels]
    result[valid] = depth[valid]   # safety: keep measured values bit-exact
    return result.astype(np.float32)


def fill_holes_telea(depth, valid):
    """Telea inpaint fill — kept for comparison; prefer fill_holes_nearest.

    Known limitations vs fill_holes_nearest:
    - cv2.inpaint requires float32 input in [0, 1]; depth is normalised before
      the call and rescaled after.  The Telea algorithm can produce values
      slightly outside [0, 1], so the rescaled result may overshoot
      [vmin, vmax] (clipped here, but the clip itself is a lossy operation).
    - Holes are zeroed before inpainting (norm[~valid] = 0), which biases
      boundary fill toward vmin when the hole abuts a low-depth region.
    - For large holes Telea diffuses smoothly but does not guarantee the
      nearest-valid value; it averages over a neighbourhood.
    """
    if valid.all():
        return depth.copy()
    if not valid.any():
        return depth.copy()
    vmin = float(depth[valid].min())
    vmax = float(depth[valid].max())
    span = vmax - vmin if vmax > vmin else 1.0
    norm = ((depth - vmin) / span).astype(np.float32)
    norm[~valid] = 0.0          # bias toward vmin at hole boundary
    mask_u8 = (~valid).astype(np.uint8)
    filled = cv2.inpaint(norm, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # Telea can overshoot [0, 1]; clip to preserve valid depth range.
    filled = np.clip(filled, 0.0, 1.0)
    result = (filled * span + vmin).astype(np.float32)
    result[valid] = depth[valid]
    return result


# Default filler used by the pipeline.
fill_holes_nn = fill_holes_nearest
