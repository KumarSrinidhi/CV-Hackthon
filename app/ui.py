"""Gradio UI for the P4 depth reconstruction application.

The pipeline is always hybrid (classical + deep -> fusion -> consistency).
No method selection is exposed to the user.

Launch with:
    python -m app.ui
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from time import perf_counter
import warnings

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
from data.degradation import degrade, downsample
from data.preprocessing import make_valid_mask
from eval.metrics import DEFAULT_BORDER, compute_all_metrics, rmse
from main import comparison_image, load_depth, reconstruct


def _path(f):
    """Gradio gives a path string (>=4) or a tempfile-like object (older)."""
    return os.fspath(f) if isinstance(f, (str, os.PathLike)) else f.name


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
    scale = int(scale)
    input_notes = []
    uploaded_mask = (
        make_valid_mask(load_depth(_path(mask_file))) if mask_file is not None else None
    )

    # A full-size depth upload can be used to generate benchmark-style LR input.
    H_rgb, W_rgb = rgb.shape[:2]
    mask = None
    if lr.shape == (H_rgb, W_rgb) and scale > 1:
        if H_rgb % scale or W_rgb % scale:
            raise ValueError(f"Full-size depth and RGB dimensions must be divisible by x{scale}.")
        valid = make_valid_mask(lr)
        if uploaded_mask is not None and uploaded_mask.shape == lr.shape:
            valid &= uploaded_mask
            uploaded_mask = None
        sample = degrade(lr, scale, valid_hr=valid)
        lr, mask = sample["lr_depth"], sample["lr_valid_mask"]
        input_notes.append(f"Depth downsampled from {W_rgb}x{H_rgb} to {lr.shape[1]}x{lr.shape[0]} using bicubic.")

    h_lr, w_lr = lr.shape
    if uploaded_mask is not None:
        if uploaded_mask.shape != lr.shape:
            raise ValueError(f"Mask must match the {w_lr}x{h_lr} LR depth dimensions.")
        mask = uploaded_mask if mask is None else mask & uploaded_mask

    expected_rgb = (w_lr * scale, h_lr * scale)
    if (W_rgb, H_rgb) != expected_rgb:
        interpolation = cv2.INTER_AREA if (W_rgb, H_rgb) > expected_rgb else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, expected_rgb, interpolation=interpolation)
        input_notes.append(
            f"RGB resized from {W_rgb}x{H_rgb} to "
            f"{expected_rgb[0]}x{expected_rgb[1]} to match the LR depth at x{scale}"
        )

    started = perf_counter()
    result = reconstruct(rgb, lr, mask, scale, cfg)
    elapsed = perf_counter() - started
    depth = result["depth"]
    inter = result["intermediates"]

    colormap = cv2.cvtColor(depth_colormap(depth), cv2.COLOR_BGR2RGB)
    out = {
        "depth": colormap,
        "boundaries": boundaries_overlay(rgb, depth, color=(255, 0, 0)),
        "result": result,
        "files": [],
        "metrics": None,
    }
    wc, wd = inter.get("confidence_classical"), inter.get("confidence_deep")
    out["conf_classical"] = confidence_overlay(rgb, wc) if wc is not None else colormap
    out["conf_deep"] = confidence_overlay(rgb, wd) if wd is not None else colormap

    deep_available = inter.get("deep_available", False)
    pipeline = "hybrid" if deep_available else "classical fallback"
    lr_valid = make_valid_mask(lr)
    if mask is not None:
        lr_valid &= mask
    classical_share = float(wc.mean()) if deep_available else 1.0
    out["run_metrics"] = {
        "pipeline": pipeline,
        "output_size": f"{depth.shape[1]} × {depth.shape[0]}",
        "scale": scale,
        "lr_coverage": float(lr_valid.mean()),
        "output_coverage": float(make_valid_mask(depth).mean()),
        "measurement_rmse": rmse(downsample(depth, scale), lr, valid=lr_valid, border=0),
        "reconstruction_seconds": elapsed,
        "classical_share": classical_share,
        "deep_share": 1.0 - classical_share,
    }
    lines = [f"Pipeline: {pipeline}  |  scale x{int(scale)}  |  output {depth.shape[1]}x{depth.shape[0]}"]
    lines.extend(f"Input adjustment: {note}" for note in input_notes)
    if deep_available:
        a = inter["classical_weight"]
        if a is None:
            a = float(wc.mean())
            fusion = f"input-adaptive (classical {a:.1%}, deep {1 - a:.1%})"
        else:
            fusion = f"fixed (classical {a:.1%}, deep {1 - a:.1%})"
        cons = (
            f"consistency strength {inter['consistency_strength']:.3g}"
            if inter["consistency_enabled"] else "no consistency"
        )
        lines.append(f"Fusion: {fusion} + {cons}")
    lines += [f"Note: {n}" for n in result["notes"]]

    if gt is not None:
        if gt.shape != depth.shape:
            raise ValueError(f"Ground truth is {gt.shape[::-1]} but the output is {depth.shape[::-1]}.")
        gt_valid = make_valid_mask(gt)
        border = min(DEFAULT_BORDER, (min(gt.shape) - 1) // 2)
        m = compute_all_metrics(depth, gt, data_range=cfg["app"]["data_range"], valid=gt_valid, border=border)
        out["metrics"] = m
        lines.append("RMSE {rmse:.4f}  MAE {mae:.4f}  PSNR {psnr:.2f}  SSIM {ssim:.4f}  "
                     "Boundary RMSE {boundary_rmse:.4f}  Edge F1 {edge_f1:.3f}".format(**m))
        if deep_available:
            classical_rmse = rmse(inter["classical"], gt, valid=gt_valid, border=border)
            deep_rmse = rmse(inter["deep"], gt, valid=gt_valid, border=border)
            m.update(classical_rmse=classical_rmse, deep_rmse=deep_rmse)
            lines.append(f"Branch RMSE  |  classical {classical_rmse:.4f}  |  deep {deep_rmse:.4f}")
        out["error"] = cv2.cvtColor(error_map(depth, gt), cv2.COLOR_BGR2RGB)
        if gt_valid.any():
            reference = degrade(gt, scale, valid_hr=gt_valid)
            overlap = lr_valid & reference["lr_valid_mask"]
            if overlap.any():
                mismatch = rmse(lr, reference["lr_depth"], valid=overlap, border=0)
                m["reference_lr_rmse"] = mismatch
                lines.append(f"LR input vs bicubic-downsampled GT RMSE {mismatch:.4f} "
                             f"over {overlap.sum()}/{overlap.size} LR pixels")

    if make_comparison:
        out["comparison"] = comparison_image(rgb, result, gt, lr)

    export_dir = export_dir or tempfile.mkdtemp(prefix="p4_export_")
    base = os.path.join(export_dir, f"{result['method']}_x{int(scale)}")
    out["files"] = [export_png(depth, base + ".png"), export_npy(depth, base + ".npy")]
    out["status"] = "\n".join(lines)
    return out


# --------------------------------------------------------------------------- #
#  Frontend: styling, demo data, and layout
# --------------------------------------------------------------------------- #

_CUSTOM_CSS = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")


def _hero_html():
    return """
<header id="hero">
  <div class="hero-copy">
    <div class="workspace-label"><span></span>P4 DEPTH STUDIO</div>
    <h1>Depth reconstruction</h1>
    <p>Pair an RGB image with low-resolution depth, inspect the reconstruction,
    and compare it against your measurements.</p>
  </div>
  <div class="hero-art" aria-hidden="true">
    <svg viewBox="0 0 330 172" fill="none">
      <defs>
        <pattern id="depth-grid" width="18" height="18" patternUnits="userSpaceOnUse">
          <path d="M18 0H0v18" stroke="#aebaff" stroke-opacity=".17"/>
        </pattern>
        <linearGradient id="depth-line">
          <stop stop-color="#7585f6"/>
          <stop offset="1" stop-color="#baf4b0"/>
        </linearGradient>
      </defs>
      <rect width="330" height="172" fill="url(#depth-grid)"/>
      <g stroke="url(#depth-line)" stroke-width="1.4">
        <path d="M-7 130c52 4 66-76 116-76s56 66 106 66 64-77 123-77"/>
        <path d="M-7 142c52 4 66-76 116-76s56 66 106 66 64-77 123-77" opacity=".78"/>
        <path d="M-7 154c52 4 66-76 116-76s56 66 106 66 64-77 123-77" opacity=".56"/>
        <path d="M-7 166c52 4 66-76 116-76s56 66 106 66 64-77 123-77" opacity=".34"/>
      </g>
      <path d="M109 27v131" stroke="#d4dcff" stroke-opacity=".35" stroke-dasharray="3 5"/>
      <circle cx="109" cy="54" r="4" fill="#baf4b0"/>
    </svg>
  </div>
</header>
"""


def _metric_cards(metrics, fields):
    cards = []
    for key, label, fmt in fields:
        value = metrics.get(key)
        text = "—" if value is None else fmt.format(value)
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                text = "—"
            elif np.isposinf(value):
                text = "∞"
        cards.append(f'<div class="metric"><span>{label}</span><strong>{text}</strong></div>')
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def _metrics_html(run=None, accuracy=None):
    run = run or {}
    pipeline = run.get("pipeline")
    badge = {
        "hybrid": '<span class="run-badge hybrid">Hybrid</span>',
        "classical fallback": '<span class="run-badge fallback">Classical fallback</span>',
    }.get(pipeline, '<span class="run-badge">Ready</span>')
    run_cards = _metric_cards(run, (
        ("measurement_rmse", "LR fit RMSE", "{:.4f}"),
        ("lr_coverage", "Input coverage", "{:.1%}"),
        ("output_coverage", "Output coverage", "{:.1%}"),
        ("reconstruction_seconds", "Reconstruction time", "{:.2f} s"),
        ("output_size", "Output size", "{}"),
        ("scale", "Upsampling", "×{}"),
        ("classical_share", "Classical weight", "{:.1%}"),
        ("deep_share", "Deep weight", "{:.1%}"),
    ))
    accuracy_html = '<p class="metric-note">Add aligned ground truth to measure accuracy.</p>'
    if accuracy is not None:
        fields = [
            ("rmse", "RMSE", "{:.4f}"),
            ("mae", "MAE", "{:.4f}"),
            ("psnr", "PSNR", "{:.2f} dB"),
            ("ssim", "SSIM", "{:.4f}"),
            ("boundary_rmse", "Boundary RMSE", "{:.4f}"),
            ("edge_f1", "Edge F1", "{:.3f}"),
            ("classical_rmse", "Classical RMSE", "{:.4f}"),
            ("deep_rmse", "Deep RMSE", "{:.4f}"),
            ("reference_lr_rmse", "LR vs. ground truth RMSE", "{:.4f}"),
        ]
        accuracy_html = _metric_cards(accuracy, [field for field in fields if field[0] in accuracy])
    return (
        '<section class="metric-report" aria-label="Reconstruction metrics" aria-live="polite">'
        f'<div class="section-heading"><h3>Run diagnostics</h3>{badge}</div>'
        '<p class="metric-note">LR fit compares the bicubic-downsampled output with valid input depth.</p>'
        f'{run_cards}<h3 class="accuracy-title">Ground-truth accuracy</h3>{accuracy_html}</section>'
    )


def _synthetic_scene(n=256, variant=0):
    """Build a small synthetic RGB-D scene with crisp depth discontinuities."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    rgb = np.zeros((n, n, 3), np.float32)
    depth = np.full((n, n), 9.0, np.float32)  # back wall

    # background wall gradient
    rgb[..., 0] = 22 + 26 * yy
    rgb[..., 1] = 30 + 34 * yy
    rgb[..., 2] = 48 + 40 * xx

    # floor band
    floor = yy > 0.82
    depth[floor] = 4.2
    rgb[floor] = np.stack([70 + 40 * xx[floor], 46 + 30 * xx[floor], 40 + 20 * xx[floor]], axis=1)

    def box(x0, x1, y0, y1, z, color):
        m = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        depth[m] = z
        shade = 0.82 + 0.28 * (1.0 - (yy[m] - y0) / max(y1 - y0, 1e-3))
        rgb[m] = np.array(color, np.float32)[None, :] * shade[:, None]
        return m

    if variant == 0:
        box(0.12, 0.46, 0.30, 0.80, 3.0, (224, 120, 52))
        box(0.56, 0.90, 0.20, 0.72, 5.5, (150, 96, 232))
        ball = ((xx - 0.5) ** 2 + (yy - 0.70) ** 2) < 0.115 ** 2
    else:
        box(0.16, 0.52, 0.18, 0.66, 2.2, (52, 168, 224))
        box(0.60, 0.92, 0.34, 0.86, 6.4, (232, 96, 120))
        ball = ((xx - 0.34) ** 2 + (yy - 0.40) ** 2) < 0.13 ** 2

    depth[ball] = 1.8
    rgb[ball] = np.stack([60 + 80 * (1 - yy[ball]), 190 - 40 * yy[ball], 90 + 60 * xx[ball]], axis=1)

    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    # darken region borders so RGB edges align with depth edges
    edges = depth_boundaries_mask_np(depth)
    rgb_u8[edges] = (rgb_u8[edges] * 0.45).astype(np.uint8)
    return rgb_u8, depth.astype(np.float32)


def depth_boundaries_mask_np(depth, rel_thresh=0.08):
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0)
    edges = np.zeros(d.shape, bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        x, y = d[a], d[b]
        jump = valid[a] & valid[b] & (np.abs(x - y) > rel_thresh * np.maximum(x, y))
        edges[a] |= jump
        edges[b] |= jump
    return edges


def _make_examples(scale=4):
    """Write synthetic example files and return their input values."""
    ex_dir = os.path.join(tempfile.gettempdir(), "p4_examples")
    os.makedirs(ex_dir, exist_ok=True)
    rows = []
    for variant in (0, 1):
        rgb, gt = _synthetic_scene(256, variant)
        lr = downsample(gt, scale)
        if variant == 1:  # punch sparse holes to demo hole-filling
            hole = np.zeros(lr.shape, bool)
            rng = np.random.default_rng(7)
            hole[rng.integers(0, lr.shape[0], 120), rng.integers(0, lr.shape[1], 120)] = True
            lr = lr.copy()
            lr[hole] = 0.0
        rgb_p = os.path.join(ex_dir, f"scene{variant}_rgb.png")
        lr_p = os.path.join(ex_dir, f"scene{variant}_x{scale}_lr.npy")
        gt_p = os.path.join(ex_dir, f"scene{variant}_gt.npy")
        cv2.imwrite(rgb_p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(lr_p, lr.astype(np.float32))
        np.save(gt_p, gt.astype(np.float32))
        rows.append([rgb_p, lr_p, scale, gt_p])
    return rows


def _gradio_callback(rgb, depth_file, mask_file, gt_file, scale, make_comparison, depth_scale):
    import gradio as gr

    try:
        o = process(rgb, depth_file, mask_file, scale, gt_file, make_comparison,
                    depth_scale=depth_scale or 1.0)
    except (ValueError, NotImplementedError, ImportError, FileNotFoundError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    status_md = "```\n" + o["status"] + "\n```"
    metrics_html = _metrics_html(o["run_metrics"], o["metrics"])
    return (
        status_md,
        metrics_html,
        o["depth"],
        o["boundaries"],
        o["conf_classical"],
        o["conf_deep"],
        o.get("comparison"),
        o.get("error"),
        o["files"],
        gr.update(visible=o["result"]["intermediates"].get("deep_available", False)),
        gr.update(visible=o.get("comparison") is not None),
        gr.update(visible=o.get("error") is not None),
        gr.update(selected="depth"),
    )


def build_app():
    import gradio as gr

    cfg = load_config()
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f0f3fa",
        body_background_fill_dark="#f0f3fa",
        body_text_color="#19223a",
        body_text_color_dark="#19223a",
        body_text_color_subdued="#526079",
        body_text_color_subdued_dark="#526079",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f7f8fd",
        background_fill_secondary_dark="#f7f8fd",
        block_background_fill="#f7f8fd",
        block_background_fill_dark="#f7f8fd",
        block_border_color="#dce2ef",
        block_border_color_dark="#dce2ef",
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_text_color="#3c4a66",
        block_label_text_color_dark="#3c4a66",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#dce2ef",
        input_border_color_dark="#dce2ef",
        button_primary_background_fill="#4936bf",
        button_primary_background_fill_dark="#4936bf",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#edeaff",
        button_secondary_background_fill_dark="#edeaff",
        button_secondary_text_color="#4837a4",
        button_secondary_text_color_dark="#4837a4",
    )

    warnings.filterwarnings("ignore", message="The parameters have been moved")
    with gr.Blocks(theme=theme, css=_CUSTOM_CSS, title="P4 Depth Studio") as demo:
        gr.HTML(_hero_html())

        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=265, elem_classes=["studio-card"]):
                gr.HTML("<div class='panel-heading'><span>01</span><h2>Input setup</h2></div>")
                rgb_in = gr.Image(label="High-resolution RGB", type="numpy", height=200)
                depth_in = gr.File(label="Low-resolution depth (.npy or .png)")
                scale_in = gr.Radio(
                    choices=cfg["app"]["scales"], value=cfg["app"]["default_scale"],
                    label="Upsampling factor", info="×2 runs classical reconstruction.",
                )
                with gr.Accordion("Reference and advanced settings", open=False):
                    mask_in = gr.File(label="Validity mask (.npy / .png)")
                    gt_in = gr.File(label="Aligned ground-truth depth (.npy / .png)")
                    depth_scale_in = gr.Number(
                        value=1.0, label="Depth multiplier",
                        info="0.001 converts a millimetre PNG to metres",
                    )
                    comparison_in = gr.Checkbox(
                        value=True, label="Generate side-by-side comparison",
                    )
                run_btn = gr.Button("Reconstruct depth", elem_id="cta", variant="primary")

                gr.HTML('<p class="sample-label">Sample scenes with ground truth</p>')
                with gr.Row():
                    for label, sample in zip(
                        ("Dense scene", "Scene with holes"), _make_examples(cfg["app"]["default_scale"])
                    ):
                        def load_sample(values=sample):
                            return *values, None, 1.0

                        gr.Button(label, size="sm").click(
                            load_sample, inputs=[],
                            outputs=[rgb_in, depth_in, scale_in, gt_in, mask_in, depth_scale_in],
                            queue=False, show_progress="hidden",
                        )

            with gr.Column(scale=2, min_width=285, elem_classes=["studio-card"]):
                gr.HTML("<div class='panel-heading'><span>02</span><h2>Reconstruction</h2></div>")
                metrics_html = gr.HTML(_metrics_html())
                with gr.Accordion("Run details", open=False):
                    status = gr.Markdown("Run a reconstruction to see the pipeline and input adjustments.",
                                         elem_classes=["status-box"])

                with gr.Tabs(selected="depth") as preview_tabs:
                    with gr.Tab("Depth", id="depth"):
                        out_depth = gr.Image(label="Reconstructed depth", elem_classes=["stage"], height=400,
                                             interactive=False)
                    with gr.Tab("Boundaries"):
                        out_boundaries = gr.Image(label="Depth boundaries", elem_classes=["stage"], height=400)
                    with gr.Tab("Branch weights", visible=False) as weights_tab:
                        with gr.Row():
                            out_wc = gr.Image(label="Classical confidence", elem_classes=["stage"], height=320)
                            out_wd = gr.Image(label="Deep confidence", elem_classes=["stage"], height=320)
                    with gr.Tab("Comparison", visible=False) as comparison_tab:
                        out_comparison = gr.Image(label="Side-by-side", elem_classes=["stage"], height=380)
                    with gr.Tab("Error map", visible=False) as error_tab:
                        out_error = gr.Image(label="Absolute depth error", elem_classes=["stage"], height=380)

                out_files = gr.File(label="Download depth (NPY values or preview PNG)", file_count="multiple",
                                    interactive=False)

        run_btn.click(
            _gradio_callback,
            inputs=[rgb_in, depth_in, mask_in, gt_in, scale_in, comparison_in, depth_scale_in],
            outputs=[status, metrics_html, out_depth, out_boundaries, out_wc, out_wd,
                     out_comparison, out_error, out_files, weights_tab, comparison_tab, error_tab, preview_tabs],
        )
    return demo


def launch(**kwargs):
    demo = build_app()
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
