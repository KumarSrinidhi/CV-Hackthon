# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: deep/inference.py
# Description: Official deep model inference wrapper conforming to shared project contracts.

from typing import Optional
import numpy as np
from deep.deep_model import DeepDepthSRModel

# Global model cache to avoid re-initializing weights per sample
_CACHED_MODEL: Optional[DeepDepthSRModel] = None


def get_deep_model(device: Optional[str] = None) -> DeepDepthSRModel:
    global _CACHED_MODEL
    if _CACHED_MODEL is None or (device is not None and str(_CACHED_MODEL.device) != device):
        _CACHED_MODEL = DeepDepthSRModel(device=device)
    return _CACHED_MODEL


def deep_hr_depth(
    lr_depth: np.ndarray,
    hr_rgb: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    scale: int = 4,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Produce high-resolution depth map from low-resolution depth and RGB guidance
    using the deep learning branch.

    Args:
        lr_depth: Low-resolution depth map of shape (H_lr, W_lr).
        hr_rgb: High-resolution RGB guidance image of shape (H_hr, W_hr, 3).
        valid_mask: Optional boolean mask of valid depth values.
        scale: Upsampling factor (e.g. 2, 4, 8).
        device: 'cuda' or 'cpu'. If None, auto-selects available device.

    Returns:
        High-resolution depth map of shape (H_hr, W_hr) as float32.
    """
    lr_depth = np.asarray(lr_depth, dtype=np.float32)
    hr_rgb = np.asarray(hr_rgb)

    model = get_deep_model(device=device)
    pred_depth = model.predict(lr_depth, hr_rgb, scale=scale)

    return pred_depth.astype(np.float32)
