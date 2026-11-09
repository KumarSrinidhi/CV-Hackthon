"""Application configuration loader.

Reads ``configs/default_config.yaml`` (or the file named by the ``P4_CONFIG``
environment variable) and merges it over built-in defaults, so the app keeps
working if the file or PyYAML is missing. The built-in defaults use input-adaptive
fusion and measurement consistency.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

DEFAULTS = {
    "hybrid": {
        "classical_alpha": None,
        "use_consistency": True,
        "consistency_strength": 0.25,
        "consistency_iterations": 1,
        "use_boundary_refinement": False,
    },
    "app": {
        "default_method": "hybrid",
        "default_scale": 4,
        "scales": [2, 4, 8, 16],
        "deep_scales": [4, 8, 16],
        "data_range": 10.0,
        "output_dir": "tmp/app_output",
    },
}

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"


def _merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    """Return the merged configuration dict (a fresh copy on every call)."""
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path or os.environ.get("P4_CONFIG") or _DEFAULT_PATH)
    if not path.is_file():
        return cfg
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except ImportError:
        return cfg
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level of the config must be a mapping")
    cfg = _merge(cfg, loaded)
    _check(cfg, path)
    return cfg


def _check(cfg, path):
    h = cfg["hybrid"]
    alpha = h["classical_alpha"]
    if alpha is not None and not (isinstance(alpha, (int, float)) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"{path}: hybrid.classical_alpha must be null or in [0, 1], got {alpha!r}")
    s = h["consistency_strength"]
    if not (isinstance(s, (int, float)) and 0.0 <= s <= 1.0):
        raise ValueError(f"{path}: hybrid.consistency_strength must be in [0, 1], got {s!r}")
    it = h["consistency_iterations"]
    if isinstance(it, bool) or not isinstance(it, int) or it < 0:
        raise ValueError(f"{path}: hybrid.consistency_iterations must be a non-negative integer")


def hybrid_kwargs(cfg=None):
    """Keyword arguments for ``hybrid_hr_depth`` taken from the config."""
    h = (cfg or load_config())["hybrid"]
    return {
        "classical_weight": h["classical_alpha"],
        "use_consistency": bool(h["use_consistency"]),
        "consistency_strength": float(h["consistency_strength"]),
        "consistency_iterations": int(h["consistency_iterations"]),
        "use_refinement": bool(h["use_boundary_refinement"]),
    }
