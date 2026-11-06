# Team Owner: Rakshitha
# Workstream: Data Pipeline
# Module: data/dataset.py
# Description: Dataset loaders for NYU-v2, Middlebury, and KITTI, including train/val/test splits and ground-truth management.
"""Dataset loaders (NumPy samples, no torch required).

Every sample is a dict with
    name                  str
    hr_rgb                float32 (H, W, 3) RGB in [0, 1]
    depth_original        float32 (H, W)   HR ground truth, native unit, NEVER modified
    valid_mask_original   bool    (H, W)   validity of the ground truth (made before any filling)
    lr_depth              float32 (h, w)   degraded measurement, invalid pixels = 0
    lr_valid_mask         bool    (h, w)   validity of lr_depth (authoritative)
    lr_depth_clean        float32 (h, w)   S(depth_original) without noise/holes
    scale                 int
    unit                  'm' (NYU) | '8bit' (Middlebury/Lu) | 'normalized' (NYU train, no minmax)
    data_range            float, FIXED range used for PSNR/SSIM/edge thresholds
    operator_exact        bool, True when lr_depth == S(depth_original) exactly

Splits (leakage-safe, see README_DEEP.md):
    NYU-v2  'train' : official SGNet train split (the checkpoints were trained on it)
            'val'   : first ~1/3 of the 449 official test images (contiguous block)
            'test'  : last block of the 449 test images, separated from 'val' by a gap
            'full'  : all 449 images (paper protocol; final reporting only)
    Middlebury / Lu : 'test' (= all images) only; tune on NYU 'val'.
    KITTI           : NOT supported (sparse depth completion != dense depth SR).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from deep.settings import dataset_dir

from .degradation import DegradationConfig, degrade
from .preprocessing import (
    align_rgb_to_depth,
    make_valid_mask,
    modcrop,
    to_depth_float,
    to_float_rgb,
)

NYU_DATA_RANGE_M = 10.0      # NYU-v2 depth is capped at 10 m
EIGHT_BIT_DATA_RANGE = 255.0


def make_split_indices(n: int, split: str, val_fraction: float = 1 / 3, gap: int = 8) -> List[int]:
    """Contiguous-block split of the NYU test images (no scene ids are available,
    so neighbouring frames are separated by ``gap`` unused images)."""
    if split == "full":
        return list(range(n))
    n_val = int(round(n * val_fraction))
    if split == "val":
        return list(range(0, n_val))
    if split == "test":
        return list(range(min(n, n_val + gap), n))
    raise ValueError(f"split must be 'val', 'test' or 'full' for test-derived splits; got '{split}'")


class DepthSRDataset:
    """Base class: subclasses implement ``_load_raw`` and set ``unit`` / ``data_range``."""

    name = "base"
    unit = "m"
    data_range = NYU_DATA_RANGE_M

    def __init__(
        self,
        scale: int,
        degradation: Optional[DegradationConfig] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.scale = int(scale)
        self.degradation = degradation or DegradationConfig()
        self.indices = list(indices) if indices is not None else None

    # -- subclass hooks --------------------------------------------------------
    def _num_raw(self) -> int:
        raise NotImplementedError

    def _load_raw(self, raw_idx: int):
        """-> (name, rgb float32 [0,1] HxWx3, depth float32 HxW, valid bool HxW)"""
        raise NotImplementedError

    # -- sequence protocol -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self._num_raw()

    def __getitem__(self, i: int) -> dict:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        raw_idx = self.indices[i] if self.indices is not None else i
        name, rgb, depth, valid = self._load_raw(raw_idx)
        s = self.scale
        rgb, depth, valid = modcrop(rgb, s), modcrop(depth, s), modcrop(valid, s)
        deg = degrade(depth, s, self.degradation, valid_hr=valid, sample_index=raw_idx)
        return {
            "name": name,
            "hr_rgb": np.ascontiguousarray(rgb, dtype=np.float32),
            "depth_original": np.ascontiguousarray(depth, dtype=np.float32),
            "valid_mask_original": np.ascontiguousarray(valid, dtype=bool),
            "lr_depth": deg["lr_depth"],
            "lr_valid_mask": deg["lr_valid_mask"],
            "lr_depth_clean": deg["lr_depth_clean"],
            "scale": s,
            "unit": self.unit,
            "data_range": float(self.data_range),
            "operator_exact": deg["operator_exact"],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ------------------------------------------------------------------------------ NYU-v2
class NYUv2Dataset(DepthSRDataset):
    """NYU-v2 in the layout of the official SGNet/DKN download (``nyu_data`` folder):

        train_depth_split.npy, train_images_split.npy,
        test_depth.npy, test_images_v2.npy, test_minmax.npy
    Depth in the .npy files is min-max normalised per image; ``test_minmax.npy``
    (shape 2 x N, rows = [max, min], metres) restores metres.
    """

    name = "nyu"
    unit = "m"
    data_range = NYU_DATA_RANGE_M

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "val",
        scale: int = 4,
        degradation: Optional[DegradationConfig] = None,
        val_fraction: float = 1 / 3,
        gap: int = 8,
    ):
        root = Path(root) if root else dataset_dir("nyu")
        self.root = root
        self.split = split
        if split == "train":
            dfile, ifile, mfile = "train_depth_split.npy", "train_images_split.npy", "train_minmax.npy"
        elif split in ("val", "test", "full"):
            dfile, ifile, mfile = "test_depth.npy", "test_images_v2.npy", "test_minmax.npy"
        else:
            raise ValueError(f"split must be one of train/val/test/full; got '{split}'")
        for f in (dfile, ifile):
            if not (root / f).is_file():
                raise FileNotFoundError(
                    f"{root / f} not found. Download the NYU-v2 npy bundle linked in the "
                    f"official SGNet README and extract it to '{root}' "
                    f"(or set NYU_ROOT / P4_DATA_ROOT)."
                )
        self._depths = np.load(root / dfile, mmap_mode="r")
        self._images = np.load(root / ifile, mmap_mode="r")
        if len(self._depths) != len(self._images):
            raise ValueError(f"{dfile} ({len(self._depths)}) and {ifile} ({len(self._images)}) differ in length")
        self._minmax = None
        if (root / mfile).is_file():
            mm = np.load(root / mfile).astype(np.float64)
            if mm.shape[0] != 2:
                mm = mm.T
            if mm.shape != (2, len(self._depths)):
                raise ValueError(f"{mfile} has shape {mm.shape}, expected (2, {len(self._depths)})")
            self._minmax = mm  # row 0 = max, row 1 = min (metres)
        elif split != "train":
            raise FileNotFoundError(f"{root / mfile} is required to convert test depth to metres.")
        if self._minmax is None:  # train split without minmax
            self.unit, self.data_range = "normalized", 1.0
        indices = None if split == "train" else make_split_indices(len(self._depths), split, val_fraction, gap)
        super().__init__(scale, degradation, indices)

    def _num_raw(self) -> int:
        return len(self._depths)

    def _load_raw(self, k: int):
        depth = np.asarray(self._depths[k], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze()
        if self._minmax is not None:
            hi, lo = self._minmax[0, k], self._minmax[1, k]
            depth = (depth * (hi - lo) + lo).astype(np.float32)
        img = np.asarray(self._images[k])
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
            img = img.transpose(1, 2, 0)
        rgb = align_rgb_to_depth(to_float_rgb(img), depth.shape)
        return f"nyu_{self.split}_{k:04d}", rgb, depth, make_valid_mask(depth)


# ------------------------------------------------------------------------- Middlebury / Lu
class MiddleburyDataset(DepthSRDataset):
    """Middlebury (30 images) / Lu (6 images) in the official folder layout:
    ``output_color_*.png`` + ``output_depth_*.png`` (8-bit). Native unit = grey levels
    0-255 (NOT metres); all pixels are evaluated like the official protocol."""

    unit = "8bit"
    data_range = EIGHT_BIT_DATA_RANGE

    def __init__(
        self,
        root: Optional[str] = None,
        scale: int = 4,
        degradation: Optional[DegradationConfig] = None,
        which: str = "middlebury",
    ):
        self.name = which
        root = Path(root) if root else dataset_dir(which)
        if not root.is_dir():
            raise FileNotFoundError(
                f"{root} not found. Download '{which}' from the link in the official SGNet "
                f"README (or set {'MIDDLEBURY_ROOT' if which == 'middlebury' else 'LU_ROOT'})."
            )
        self.root = root
        self._rgbs = sorted(p for p in root.iterdir() if "output_color" in p.name)
        self._gts = sorted(p for p in root.iterdir() if "output_depth" in p.name)
        if not self._gts or len(self._rgbs) != len(self._gts):
            raise FileNotFoundError(
                f"{root}: expected matching output_color_*/output_depth_* files, found "
                f"{len(self._rgbs)} colour and {len(self._gts)} depth files."
            )
        super().__init__(scale, degradation, None)

    def _num_raw(self) -> int:
        return len(self._gts)

    def _load_raw(self, k: int):
        gt = np.asarray(Image.open(self._gts[k]))
        if gt.ndim == 3:
            gt = gt[..., 0]
        depth = to_depth_float(gt)
        rgb = to_float_rgb(np.asarray(Image.open(self._rgbs[k]).convert("RGB")))
        rgb = align_rgb_to_depth(rgb, depth.shape)
        return self._gts[k].stem, rgb, depth, make_valid_mask(depth, min_valid=None)


# ------------------------------------------------------------------------------ factory
def get_dataset(
    name: str,
    split: str = "val",
    scale: int = 4,
    root: Optional[str] = None,
    degradation: Optional[DegradationConfig] = None,
) -> DepthSRDataset:
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("nyu", "nyuv2"):
        return NYUv2Dataset(root, split, scale, degradation)
    if key in ("middlebury", "lu"):
        if split not in ("test", "full"):
            raise ValueError(
                f"{name} has no validation split (30/6 images). Tune on NYU 'val' and use "
                f"split='test' here for final reporting."
            )
        return MiddleburyDataset(root, scale, degradation, which=key)
    if key == "kitti":
        raise NotImplementedError(
            "KITTI is NOT evaluated: it provides sparse LiDAR depth completion data (input and "
            "output share the RGB resolution, GT is semi-dense), which is not the dense "
            "scale-s depth super-resolution task SGNet was trained for. No SGNet-on-KITTI "
            "claim is made. See deep/README_DEEP.md."
        )
    raise ValueError(f"Unknown dataset '{name}'. Use one of: nyu, middlebury, lu.")


def read_rgb(path):
    """Read an image as RGB uint8 (BGR -> RGB)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def manifest_frames(csv_path):
    """Yield (group, frame, rgb, depth) rows from a CSV manifest.

    Columns: ``frame`` (id), ``rgb`` (image path), ``depth`` (.npy path), optional
    ``group``. Used by the real-depth evaluation CLI.
    """
    import csv

    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = row.get("group", row.get("dataset", "nyu"))
            frame = row["frame"] if "frame" in row else Path(row["depth"]).stem
            yield (group, frame, read_rgb(row["rgb"]), np.load(row["depth"]))
