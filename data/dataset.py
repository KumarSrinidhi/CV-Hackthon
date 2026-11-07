# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/dataset.py
# Description: Dataset loaders and reproducible synthetic RGB-D generation for benchmarking depth super-resolution.

from typing import Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
from data.degradation import create_lr_depth


def generate_synthetic_rgbd_scene(
    h: int = 240,
    w: int = 320,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a high-resolution RGB image and paired ground-truth depth map
    with distinct geometric planes, objects, and depth discontinuities.
    """
    np.random.seed(seed)

    # Background wall (depth plane around 3.5m)
    y, x = np.mgrid[0:h, 0:w]
    depth = 3.5 + 0.3 * (x / w) + 0.2 * (y / h)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [180, 190, 200]

    # Foreground object 1: Table / box (depth ~2.0m)
    box1_y1, box1_y2 = int(h * 0.4), int(h * 0.85)
    box1_x1, box1_x2 = int(w * 0.2), int(w * 0.6)
    depth[box1_y1:box1_y2, box1_x1:box1_x2] = 2.0 + 0.1 * ((y[box1_y1:box1_y2, box1_x1:box1_x2] - box1_y1) / (box1_y2 - box1_y1))
    rgb[box1_y1:box1_y2, box1_x1:box1_x2] = [80, 120, 180]

    # Foreground object 2: Sphere / cylinder (depth ~1.2m)
    center_y, center_x = int(h * 0.6), int(w * 0.75)
    radius = int(min(h, w) * 0.18)
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    sphere_mask = dist_sq <= radius ** 2
    depth[sphere_mask] = 1.2 + 0.4 * np.sqrt(np.maximum(0.0, 1.0 - dist_sq[sphere_mask] / (radius ** 2)))
    rgb[sphere_mask] = [200, 70, 70]

    return rgb, depth.astype(np.float32)


class RGBDDataset:
    """
    Dataset wrapper for evaluation and benchmarking.
    Can load real RGB-D images from disk or generate controlled synthetic evaluation sets.
    """

    def __init__(self, data_dir: Optional[str] = None, num_synthetic_samples: int = 10, scale: int = 4):
        self.data_dir = data_dir
        self.scale = scale
        self.samples: List[Dict[str, np.ndarray]] = []

        if data_dir is not None and os.path.isdir(data_dir):
            self._load_from_dir(data_dir)
        else:
            self._generate_synthetic_set(num_synthetic_samples)

    def _generate_synthetic_set(self, count: int):
        for i in range(count):
            rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=100 + i)
            lr_depth, valid_mask = create_lr_depth(gt_depth, scale=self.scale)
            self.samples.append({
                "id": f"synthetic_{i:03d}",
                "rgb": rgb,
                "gt_depth": gt_depth,
                "lr_depth": lr_depth,
                "valid_mask": valid_mask,
            })

    def _load_from_dir(self, directory: str):
        # Scan for paired images and depths
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.samples[idx]
