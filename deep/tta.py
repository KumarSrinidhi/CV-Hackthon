"""Drop into deep/. Flip-TTA wrapper around deep_hr_depth (~2-4x latency, lowers RMSE/outliers).

    from deep.tta import deep_hr_depth_tta
"""
import numpy as np
from .inference import deep_hr_depth


def deep_hr_depth_tta(lr_depth, hr_rgb, valid_mask=None, scale=4, flips=("none", "h"), **kw):
    lr = np.asarray(lr_depth)
    rgb = np.asarray(hr_rgb)
    m = None if valid_mask is None else np.asarray(valid_mask)
    outs = []
    for f in flips:
        if f == "h":
            o = deep_hr_depth(lr[:, ::-1].copy(), rgb[:, ::-1].copy(),
                              None if m is None else m[:, ::-1].copy(), scale, **kw)[:, ::-1]
        elif f == "v":
            o = deep_hr_depth(lr[::-1].copy(), rgb[::-1].copy(),
                              None if m is None else m[::-1].copy(), scale, **kw)[::-1]
        else:
            o = deep_hr_depth(lr, rgb, m, scale, **kw)
        outs.append(o)
    # median is robust to the occasional bad pass; mean for 2 views
    return (np.mean(outs, 0) if len(outs) < 3 else np.median(outs, 0)).astype(np.float32)
