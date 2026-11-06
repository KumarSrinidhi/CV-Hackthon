# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: deep/settings.py
# Description: Paths, supported SGNet variants and shared exceptions for the deep branch.
#              Everything is resolved lazily (environment variables are read at call
#              time) so tests and the UI can redirect checkpoints without re-importing.
"""Deep-branch settings.

Environment overrides (all optional)
------------------------------------
SGNET_CKPT_DIR     directory holding the official SGNet ``*.pth`` files
                   (default: <repo>/checkpoints/sgnet)
SGNET_CKPT_X4/X8/X16   explicit file path for one scale
P4_DATA_ROOT       parent of the dataset folders (default: <repo>/datasets)
NYU_ROOT / MIDDLEBURY_ROOT / LU_ROOT   explicit dataset folders
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Official SGNet variants (from the official README: train/test commands).
#   x4  -> --num_feats 48,  x8 / x16 -> --num_feats 40
# Scale 2 has no released checkpoint (architecture supports it, weights do not).
SGNET_VARIANTS = {
    4: {"num_feats": 48},
    8: {"num_feats": 40},
    16: {"num_feats": 40},
}
SUPPORTED_SCALES = tuple(sorted(SGNET_VARIANTS))


class DeepModelUnavailableError(RuntimeError):
    """Checkpoint missing/incompatible, CUDA unusable, or out of memory."""


class UnsupportedScaleError(ValueError):
    """Requested scale has no verified SGNet checkpoint."""


def ckpt_dir() -> Path:
    return Path(os.environ.get("SGNET_CKPT_DIR", REPO_ROOT / "checkpoints" / "sgnet"))


def data_root() -> Path:
    return Path(os.environ.get("P4_DATA_ROOT", REPO_ROOT / "datasets"))


def dataset_dir(name: str) -> Path:
    name = name.lower()
    env = {"nyu": "NYU_ROOT", "middlebury": "MIDDLEBURY_ROOT", "lu": "LU_ROOT"}[name]
    default = {"nyu": "nyu_data", "middlebury": "Middlebury", "lu": "Lu"}[name]
    return Path(os.environ.get(env, data_root() / default))


def ensure_supported_scale(scale: int) -> int:
    if scale not in SGNET_VARIANTS:
        raise UnsupportedScaleError(
            f"Deep branch (SGNet) has verified checkpoints only for scales "
            f"{SUPPORTED_SCALES}; got x{scale}. Use the classical branch for this "
            f"scale or train an SGNet x{scale} model yourself."
        )
    return int(scale)
