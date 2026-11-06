"""Gradio UI for the P4 depth reconstruction application.

The Gradio dependency is imported lazily so the CLI, tests, and evaluation code do
not require it. Launch with:

    python app/ui.py
    # or
    python -c "from app.ui import launch; launch()"
"""

from __future__ import annotations

import cv2
import numpy as np

from app.visualization import (
    boundaries_overlay,
    confidence_overlay,
    depth_colormap,
)
from main import load_depth, reconstruct

_METHODS = ["hybrid", "classical", "deep", "jbu", "guided", "bicubic", "bilinear", "nearest"]


def _rgb_for_overlay(rgb):
    return np.asarray(rgb) if np.asarray(rgb).dtype == np.uint8 else (np.clip(np.asarray(rgb), 0, 1) * 255).astype(np.uint8)


def run_reconstruction(rgb, depth_file, mask_file, scale, method):
    """Core UI callback: returns (depth_colormap, boundaries, classical_conf, deep_conf)."""
    if rgb is None:
        raise ValueError("RGB guidance image is required.")
    rgb = np.asarray(rgb)
    if rgb.dtype == np.float32 and rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)

    if depth_file is None:
        raise ValueError("Low-resolution depth map is required.")
    lr = load_depth(depth_file.name if hasattr(depth_file, "name") else str(depth_file))
    mask = None
    if mask_file is not None:
        mask = load_depth(mask_file.name if hasattr(mask_file, "name") else str(mask_file))

    result = reconstruct(rgb, lr, mask, int(scale), method)
    depth = result["depth"]
    inter = result.get("intermediates", {})

    # The visualization helpers below follow OpenCV's BGR convention; Gradio
    # displays RGB, so convert before returning.
    colormap = cv2.cvtColor(depth_colormap(depth), cv2.COLOR_BGR2RGB)
    boundaries = boundaries_overlay(rgb, depth, color=(255, 0, 0))
    wc = inter.get("confidence_classical")
    wd = inter.get("confidence_deep")
    conf_c = confidence_overlay(rgb, wc) if wc is not None else colormap
    conf_d = confidence_overlay(rgb, wd) if wd is not None else colormap
    return colormap, boundaries, conf_c, conf_d


def build_app():
    import gradio as gr

    with gr.Blocks(title="P4 — Edge-Preserving Depth Upsampling") as demo:
        gr.Markdown("# P4 — Edge-Preserving Depth Map Upsampling & Refinement")
        gr.Markdown(
            "Upload an RGB guidance image and a low-resolution or sparse depth map. "
            "Choose a scale factor and method, then run reconstruction."
        )
        with gr.Row():
            rgb = gr.Image(label="RGB guidance", type="numpy")
            with gr.Column():
                depth = gr.File(label="LR depth (.npy or .png)")
                mask = gr.File(label="Validity mask (.npy, optional)")
                scale = gr.Dropdown([2, 4, 8], value=4, label="Upsampling factor")
                method = gr.Dropdown(_METHODS, value="hybrid", label="Method")
                run_btn = gr.Button("Reconstruct")

        with gr.Row():
            out_depth = gr.Image(label="Reconstructed depth")
            out_boundaries = gr.Image(label="Depth boundaries")
        with gr.Row():
            out_wc = gr.Image(label="Classical confidence")
            out_wd = gr.Image(label="Deep confidence")

        run_btn.click(
            run_reconstruction,
            inputs=[rgb, depth, mask, scale, method],
            outputs=[out_depth, out_boundaries, out_wc, out_wd],
        )
    return demo


def launch():
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    launch()
