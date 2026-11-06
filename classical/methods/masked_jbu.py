"""Joint bilateral upsampling using only original valid samples."""

from .jbu import joint_bilateral_upsample


def masked_jbu(depth, valid, rgb):
    return joint_bilateral_upsample(depth, valid, rgb)
