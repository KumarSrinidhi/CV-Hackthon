"""Render aligned RGB and depth comparisons with a shared depth color scale."""

import cv2
import numpy as np


def write_comparison(path, rgb, target, predictions):
    low, high = np.percentile(target[target > 0], [2, 98])

    def depth_colors(depth):
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        colors = cv2.applyColorMap((255 * normalized).astype(np.uint8), cv2.COLORMAP_TURBO)
        colors[depth <= 0] = 0
        return colors

    panels = [("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
              ("Reference", depth_colors(target))]
    panels.extend((method, depth_colors(depth)) for method, depth in predictions.items())
    labeled = []
    for label, image in panels:
        tile = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        labeled.append(tile)
    if not cv2.imwrite(str(path), np.vstack(labeled)):
        raise OSError(f"Could not write comparison image to {path}")
