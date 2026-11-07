# Team Owner: Tanush
# Workstream: Classical Computer Vision
# Module: classical/interpolation.py
# Description: Standard interpolation baselines: nearest-neighbor, bilinear, and bicubic upsampling.

from typing import Tuple, Union
import cv2
import numpy as np


def nearest_upsample(lr_depth: np.ndarray, target_size: Union[Tuple[int, int], int] = None, scale: int = 4) -> np.ndarray:
    """Nearest-neighbor upsampling baseline for depth maps."""
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    h, w = lr_depth.shape[:2]
    if target_size is not None:
        if isinstance(target_size, int):
            dsize = (w * target_size, h * target_size)
        else:
            dsize = (int(target_size[1]), int(target_size[0]))
    else:
        dsize = (w * scale, h * scale)
    return cv2.resize(lr_depth, dsize, interpolation=cv2.INTER_NEAREST).astype(np.float32)


def bilinear_upsample(lr_depth: np.ndarray, target_size: Union[Tuple[int, int], int] = None, scale: int = 4) -> np.ndarray:
    """Bilinear upsampling baseline for depth maps."""
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    h, w = lr_depth.shape[:2]
    if target_size is not None:
        if isinstance(target_size, int):
            dsize = (w * target_size, h * target_size)
        else:
            dsize = (int(target_size[1]), int(target_size[0]))
    else:
        dsize = (w * scale, h * scale)
    return cv2.resize(lr_depth, dsize, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def bicubic_upsample(lr_depth: np.ndarray, target_size: Union[Tuple[int, int], int] = None, scale: int = 4) -> np.ndarray:
    """Bicubic upsampling baseline for depth maps."""
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    h, w = lr_depth.shape[:2]
    if target_size is not None:
        if isinstance(target_size, int):
            dsize = (w * target_size, h * target_size)
        else:
            dsize = (int(target_size[1]), int(target_size[0]))
    else:
        dsize = (w * scale, h * scale)
    return cv2.resize(lr_depth, dsize, interpolation=cv2.INTER_CUBIC).astype(np.float32)
