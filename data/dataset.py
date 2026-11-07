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
    Can load real RGB-D images from disk (e.g. depthsr_testset) or generate controlled synthetic evaluation sets.
    """

    def __init__(self, data_dir: Optional[str] = None, num_synthetic_samples: int = 10, scale: int = 4):
        self.data_dir = data_dir
        self.scale = scale
        self.samples: List[Dict[str, np.ndarray]] = []

        # Auto-detect official testset if no directory specified
        default_testset = os.path.abspath(os.path.join(os.path.dirname(__file__), "depthsr_testset"))
        if self.data_dir is None and os.path.isdir(default_testset):
            self.data_dir = default_testset

        if self.data_dir is not None and os.path.isdir(self.data_dir) and self.data_dir != "synthetic":
            self._load_from_dir(self.data_dir)
            if len(self.samples) == 0:
                self._generate_synthetic_set(num_synthetic_samples)
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
        # 1. Check for sample folders (depthsr_testset convention)
        subdirs = sorted([
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ])

        for sdir in subdirs:
            rgb_path = os.path.join(sdir, "rgb.png")
            gt_path = os.path.join(sdir, "depth_hr.npy")
            lr_depth_path = os.path.join(sdir, f"lr_depth_{self.scale}x.npy")
            lr_mask_path = os.path.join(sdir, f"lr_mask_{self.scale}x.npy")

            if os.path.exists(rgb_path) and os.path.exists(gt_path):
                bgr = cv2.imread(rgb_path)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                gt_depth = np.load(gt_path).astype(np.float32)

                if os.path.exists(lr_depth_path):
                    lr_depth = np.load(lr_depth_path).astype(np.float32)
                else:
                    lr_depth, _ = create_lr_depth(gt_depth, scale=self.scale)

                if os.path.exists(lr_mask_path):
                    valid_mask = np.load(lr_mask_path).astype(np.uint8)
                else:
                    valid_mask = (lr_depth > 0).astype(np.uint8)

                self.samples.append({
                    "id": os.path.basename(sdir),
                    "rgb": rgb,
                    "gt_depth": gt_depth,
                    "lr_depth": lr_depth,
                    "valid_mask": valid_mask,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.samples[idx]
