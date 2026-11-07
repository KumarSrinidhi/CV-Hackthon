"""Gradio UI for the P4 depth reconstruction application.

The pipeline is always hybrid (classical + deep → fusion → consistency).
No method selection is exposed to the user.

Launch with:
    python -m app.ui
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from app.config import load_config
from app.visualization import (
    boundaries_overlay,
    confidence_overlay,
    depth_colormap,
    error_map,
    export_npy,
    export_png,
)
from main import comparison_image, load_depth, reconstruct


def _path(f):
    """Gradio gives a path string (>=4) or a tempfile-like object (older)."""
    return f.name if hasattr(f, "name") else str(f)


def process(rgb, depth_file, mask_file, scale, gt_file=None, make_comparison=False,
            export_dir=None, depth_scale=1.0):
    """Run the hybrid reconstruction and build every output the UI shows.

    Returns a dict with RGB uint8 images ``depth``, ``boundaries``,
    ``conf_classical``, ``conf_deep``, optional ``comparison`` / ``error``,
    a ``status`` string, the exported ``files`` and the raw ``result``.
    """
    if rgb is None:
        raise ValueError("RGB guidance image is required.")
    rgb = np.asarray(rgb)
    if rgb.dtype == np.float32 and rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    if depth_file is None:
        raise ValueError("Low-resolution depth map is required.")

    lr = load_depth(_path(depth_file), depth_scale)
    gt = load_depth(_path(gt_file), depth_scale) if gt_file is not None else None
    cfg = load_config()

    # Auto-fix geometry: if depth is the same size as RGB, downsample it to LR.
    H_rgb, W_rgb = rgb.shape[:2]
    h_lr, w_lr = lr.shape[:2]
    if (h_lr, w_lr) == (H_rgb, W_rgb):
        h_lr, w_lr = H_rgb // scale, W_rgb // scale
        lr = cv2.resize(lr, (w_lr, h_lr), interpolation=cv2.INTER_AREA)

    # Derive validity mask from depth (ignore any uploaded mask — it may be HR-sized).
    mask = None
    if mask_file is not None:
        raw_mask = load_depth(_path(mask_file))
        if raw_mask.shape == (h_lr, w_lr):
            mask = (raw_mask > 0).astype(bool)

    result = reconstruct(rgb, lr, mask, scale, cfg)
    depth = result["depth"]
    inter = result["intermediates"]

    colormap = cv2.cvtColor(depth_colormap(depth), cv2.COLOR_BGR2RGB)
    out = {
        "depth": colormap,
        "boundaries": boundaries_overlay(rgb, depth, color=(255, 0, 0)),
        "result": result,
        "files": [],
    }
    wc, wd = inter.get("confidence_classical"), inter.get("confidence_deep")
    out["conf_classical"] = confidence_overlay(rgb, wc) if wc is not None else colormap
    out["conf_deep"] = confidence_overlay(rgb, wd) if wd is not None else colormap

    lines = [f"Pipeline: hybrid  |  scale x{int(scale)}  |  output {depth.shape[1]}x{depth.shape[0]}"]
    deep_available = result["intermediates"].get("deep_available", False)
    if not result["fallback"]:
        h = cfg["hybrid"]
        a = h["classical_alpha"]
        if deep_available:
            fusion = f"fixed α={a} (classical {a}, deep {1 - a:.2f})" if a is not None else "adaptive confidence"
            cons = f"consistency strength {h['consistency_strength']}" if h["use_consistency"] else "no consistency"
            lines.append(f"Fusion: {fusion} + {cons}")
        else:
            lines.append("Deep branch: no checkpoint — classical pipeline (A1→A4) used")
    lines += [f"Note: {n}" for n in result["notes"]]

    if gt is not None:
        if gt.shape != depth.shape:
            raise ValueError(f"Ground truth is {gt.shape[::-1]} but the output is {depth.shape[::-1]}.")
        from eval.metrics import compute_all_metrics

        m = compute_all_metrics(depth, gt, data_range=cfg["app"]["data_range"])
        lines.append("RMSE {rmse:.4f}  MAE {mae:.4f}  PSNR {psnr:.2f}  SSIM {ssim:.4f}  "
                     "Boundary RMSE {boundary_rmse:.4f}  Edge F1 {edge_f1:.3f}".format(**m))
        out["error"] = cv2.cvtColor(error_map(depth, gt), cv2.COLOR_BGR2RGB)

    if make_comparison:
        out["comparison"] = comparison_image(rgb, result, gt, lr)

    export_dir = export_dir or tempfile.mkdtemp(prefix="p4_export_")
    base = os.path.join(export_dir, f"{result['method']}_x{int(scale)}")
    out["files"] = [export_png(depth, base + ".png"), export_npy(depth, base + ".npy")]
    out["status"] = "\n".join(lines)
    return out


def _gradio_callback(rgb, depth_file, mask_file, gt_file, scale, make_comparison, depth_scale):
    import gradio as gr

    try:
        o = process(rgb, depth_file, mask_file, scale, gt_file, make_comparison,
                    depth_scale=depth_scale or 1.0)
    except (ValueError, NotImplementedError, ImportError, FileNotFoundError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    return (o["depth"], o["boundaries"], o["conf_classical"], o["conf_deep"],
            o.get("comparison"), o.get("error"), o["status"], o["files"])


def build_app():
    import gradio as gr

    cfg = load_config()
    with gr.Blocks(title="P4 — Edge-Preserving Depth Upsampling") as demo:
        gr.Markdown("# P4 — Edge-Preserving Depth Map Upsampling & Refinement")
        gr.Markdown(
            "Upload an HR RGB guidance image and a low-resolution depth map. "
            "The RGB must be exactly *scale* times larger than the depth map in each dimension. "
            "The pipeline runs automatically: **Classical** (hole-fill → bicubic → JBU → boundary refinement) "
            "and **Deep** (SGNet) branches run in parallel, then fuse via **Hybrid** "
            "(fixed α=0.02 classical + 0.98 deep) with measurement-consistency correction."
        )
        with gr.Row():
            rgb_in = gr.Image(label="RGB guidance", type="numpy")
            with gr.Column():
                depth_in = gr.File(label="LR depth (.npy or .png)")
                mask_in = gr.File(label="Validity mask (.npy / .png, optional)")
                gt_in = gr.File(label="HR ground truth (.npy / .png, optional — enables metrics)")
                scale_in = gr.Dropdown(cfg["app"]["scales"], value=cfg["app"]["default_scale"],
                                       label="Upsampling factor")
                depth_scale_in = gr.Number(value=1.0, label="Depth multiplier (0.001 converts mm PNG to metres)")
                comparison_in = gr.Checkbox(value=True, label="Show side-by-side panel (LR depth, classical, deep, hybrid, GT)")
                run_btn = gr.Button("Reconstruct", variant="primary")

        status = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Row():
            out_depth = gr.Image(label="Hybrid output depth")
            out_boundaries = gr.Image(label="Depth boundaries")
        with gr.Row():
            out_wc = gr.Image(label="Classical branch confidence")
            out_wd = gr.Image(label="Deep branch confidence")
        out_comparison = gr.Image(label="Side-by-side comparison")
        out_error = gr.Image(label="Error map |pred - GT| (needs ground truth)")
        out_files = gr.File(label="Download (PNG 16-bit, NPY float32)", file_count="multiple",
                            interactive=False)

        run_btn.click(
            _gradio_callback,
            inputs=[rgb_in, depth_in, mask_in, gt_in, scale_in, comparison_in, depth_scale_in],
            outputs=[out_depth, out_boundaries, out_wc, out_wd, out_comparison, out_error,
                     status, out_files],
        )
    return demo


def launch(**kwargs):
    demo = build_app()
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
