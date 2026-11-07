# Team Owner: Kumar
# Workstream: Application Testing
# Module: tests/test_app.py
# Description: Unit tests for UI visualization helpers, colorization, and pipeline demo runner.

import os
import numpy as np
import pytest
from app.visualization import colorize_depth, create_side_by_side_comparison
from app.ui import run_pipeline_demo


def test_colorize_depth():
    depth = np.random.uniform(0.5, 5.0, (50, 50)).astype(np.float32)
    rgb_vis = colorize_depth(depth)
    assert rgb_vis.shape == (50, 50, 3)
    assert rgb_vis.dtype == np.uint8


def test_side_by_side_comparison(tmp_path):
    h, w = 40, 60
    rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    lr_depth = np.random.uniform(1.0, 4.0, (h // 4, w // 4)).astype(np.float32)
    depth_map = np.random.uniform(1.0, 4.0, (h, w)).astype(np.float32)

    save_file = os.path.join(str(tmp_path), "test_comparison.png")
    fig = create_side_by_side_comparison(
        rgb, lr_depth, depth_map, depth_map, depth_map, depth_map, gt_depth=depth_map, save_path=save_file
    )
    assert os.path.isfile(save_file)
    assert os.path.getsize(save_file) > 0


def test_pipeline_demo_run(tmp_path):
    df = run_pipeline_demo(scale=2, output_dir=str(tmp_path))
    assert len(df) == 4
    assert "Method" in df.columns
    assert "RMSE" in df.columns
    assert os.path.isfile(os.path.join(str(tmp_path), "depth_sr_comparison.png"))
