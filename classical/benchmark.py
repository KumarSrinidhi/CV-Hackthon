"""Compatibility entry point for the classical benchmark."""

from .evaluation.benchmark import BASELINES, COMPARISONS, JUMPS, evaluate, main, mean, predict
from .evaluation.metrics import DEPTH_RANGE, quality, ssim
from .evaluation.synthetic_data import KINDS, SCALES, make_scene


if __name__ == "__main__":
    main()
