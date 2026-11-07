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
    with distinct geometric planes, objects, and depth discontinuities based on seed.
    """
    rng = np.random.RandomState(seed)

    # Background wall (depth plane around 3.5m - 4.5m)
    y, x = np.mgrid[0:h, 0:w]
    tilt_x = rng.uniform(0.1, 0.4)
    tilt_y = rng.uniform(0.1, 0.3)
    base_depth = rng.uniform(3.2, 4.2)
    depth = base_depth + tilt_x * (x / w) + tilt_y * (y / h)

    # Base background wall color
    bg_color = rng.randint(40, 200, size=3)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = bg_color

    # Foreground object 1: Table / box
    b_w = rng.uniform(0.25, 0.45)
    b_h = rng.uniform(0.3, 0.5)
    b_x = rng.uniform(0.1, 0.5)
    b_y = rng.uniform(0.35, 0.5)
    box1_y1, box1_y2 = int(h * b_y), int(h * min(0.95, b_y + b_h))
    box1_x1, box1_x2 = int(w * b_x), int(w * min(0.9, b_x + b_w))

    box_depth = rng.uniform(1.8, 2.6)
    depth[box1_y1:box1_y2, box1_x1:box1_x2] = box_depth + 0.1 * ((y[box1_y1:box1_y2, box1_x1:box1_x2] - box1_y1) / max(1, box1_y2 - box1_y1))
    box_color = rng.randint(40, 240, size=3)
    rgb[box1_y1:box1_y2, box1_x1:box1_x2] = box_color

    # Foreground object 2: Sphere / cylinder
    center_y = int(h * rng.uniform(0.45, 0.75))
    center_x = int(w * rng.uniform(0.6, 0.85))
    radius = int(min(h, w) * rng.uniform(0.12, 0.22))
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    sphere_mask = dist_sq <= radius ** 2
    sphere_depth = rng.uniform(1.0, 1.6)
    depth[sphere_mask] = sphere_depth + 0.4 * np.sqrt(np.maximum(0.0, 1.0 - dist_sq[sphere_mask] / (radius ** 2)))
    sphere_color = rng.randint(60, 255, size=3)
    rgb[sphere_mask] = sphere_color

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
