"""JBU with selected spatial and range settings."""

from .jbu import joint_bilateral_upsample


# Entries are (spatial sigma in LR pixels, RGB sigma, radius in LR pixels).
DEFAULT_SETTINGS = (0.85, 0.12, 2)
SCALE_SETTINGS = {2: (0.85, 0.08, 2), 4: (0.85, 0.12, 2), 8: (1.10, 0.12, 2)}


def tuned_jbu(depth, valid, rgb, settings):
    spatial, color, radius = settings
    return joint_bilateral_upsample(depth, valid, rgb, sigma_space=spatial,
                                    sigma_color=color, radius=radius)
