"""Adaptive JBU with boundary refinement enabled."""

from .adaptive_jbu import adaptive_jbu


def refined_jbu(depth, valid, rgb, settings):
    return adaptive_jbu(depth, valid, rgb, settings, refine=True)
