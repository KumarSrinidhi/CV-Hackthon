"""Gradio UI for the P4 depth reconstruction application.

The pipeline is always hybrid (classical + deep -> fusion -> consistency).
No method selection is exposed to the user.

Launch with:
    python -m app.ui
"""

from __future__ import annotations

import os
import tempfile
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
    scale = int(scale)
    input_note = None

    # Auto-fix geometry: if depth is the same size as RGB, downsample it to LR.
    H_rgb, W_rgb = rgb.shape[:2]
    h_lr, w_lr = lr.shape[:2]
    if (h_lr, w_lr) == (H_rgb, W_rgb):
        h_lr, w_lr = H_rgb // scale, W_rgb // scale
        lr = cv2.resize(lr, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    expected_rgb = (w_lr * scale, h_lr * scale)
    if (W_rgb, H_rgb) != expected_rgb:
        interpolation = cv2.INTER_AREA if (W_rgb, H_rgb) > expected_rgb else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, expected_rgb, interpolation=interpolation)
        input_note = (
            f"RGB resized from {W_rgb}x{H_rgb} to "
            f"{expected_rgb[0]}x{expected_rgb[1]} to match the LR depth at x{scale}"
        )

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
        "metrics": None,
    }
    wc, wd = inter.get("confidence_classical"), inter.get("confidence_deep")
    out["conf_classical"] = confidence_overlay(rgb, wc) if wc is not None else colormap
    out["conf_deep"] = confidence_overlay(rgb, wd) if wd is not None else colormap

    lines = [f"Pipeline: hybrid  |  scale x{int(scale)}  |  output {depth.shape[1]}x{depth.shape[0]}"]
    if input_note:
        lines.append(f"Input adjustment: {input_note}")
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
        out["metrics"] = m
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


# --------------------------------------------------------------------------- #
#  Frontend: styling, demo data, and layout
# --------------------------------------------------------------------------- #

_CUSTOM_CSS = """
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 0 18px 48px !important;
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
  color: #e6ecff !important;
}
body, .gradio-container {
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(99,102,241,0.22), transparent 60%),
    radial-gradient(900px 520px at 98% 6%, rgba(56,189,248,0.16), transparent 55%),
    linear-gradient(180deg, #0a0e1f 0%, #0b1124 42%, #0a0e1f 100%) !important;
}
#hero { padding: 26px 6px 12px; }
#hero .eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 12px; letter-spacing: 2.2px; text-transform: uppercase;
  color: #a5b4fc; font-weight: 600;
  padding: 6px 12px; border-radius: 999px;
  background: rgba(99,102,241,0.14); border: 1px solid rgba(129,140,248,0.35);
}
#hero h1 {
  font-size: clamp(30px, 5vw, 52px); line-height: 1.04; margin: 16px 0 10px;
  font-weight: 800; letter-spacing: -1px;
  background: linear-gradient(120deg, #e9edff 0%, #c7d2fe 32%, #7dd3fc 68%, #a5b4fc 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
#hero .sub { font-size: 17px; line-height: 1.6; color: #b7c2e6; max-width: 820px; }
.badges { display: flex; flex-wrap: wrap; gap: 9px; margin: 18px 2px 4px; }
.badge {
  font-size: 12.5px; color: #cdd7ff; font-weight: 500;
  padding: 7px 13px; border-radius: 11px;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
}
.badge b { color: #fff; }
.card {
  background: rgba(17, 24, 46, 0.62) !important;
  border: 1px solid rgba(129, 140, 248, 0.18) !important;
  border-radius: 20px !important;
  padding: 22px 22px 24px !important;
  box-shadow: 0 18px 50px rgba(2, 6, 23, 0.45) !important;
  backdrop-filter: blur(9px);
}
.card-title {
  font-size: 13px; letter-spacing: 1.6px; text-transform: uppercase;
  color: #93c5fd; font-weight: 700; margin: 0 0 14px;
}
.card-title .dot { color: #6366f1; }
#cta {
  background: linear-gradient(120deg, #6366f1 0%, #4f46e5 45%, #0ea5e9 100%) !important;
  color: #fff !important; font-weight: 700 !important; font-size: 17px !important;
  border: none !important; border-radius: 14px !important; height: 52px !important;
  box-shadow: 0 12px 30px rgba(79, 70, 229, 0.45) !important;
  transition: transform .12s ease, box-shadow .12s ease;
}
#cta:hover { transform: translateY(-2px); box-shadow: 0 18px 40px rgba(79,70,229,0.55) !important; }
.status-box {
  background: rgba(9, 14, 32, 0.66) !important;
  border: 1px solid rgba(129, 140, 248, 0.22) !important;
  border-radius: 14px !important; padding: 16px 18px !important;
  color: #c9d4f5 !important; font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 13px !important; line-height: 1.7 !important;
}
.metric-strip { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 8px; }
.metric {
  flex: 1 1 128px; min-width: 128px;
  background: linear-gradient(160deg, rgba(99,102,241,0.16), rgba(56,189,248,0.10));
  border: 1px solid rgba(129,140,248,0.28); border-radius: 14px; padding: 13px 15px;
}
.metric .val { font-size: 23px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.metric .name { font-size: 11px; letter-spacing: 1.1px; text-transform: uppercase; color: #9fb0dd; margin-top: 3px; }
.tab-nav { gap: 6px; }
.tab-item {
  background: rgba(255,255,255,0.04) !important; border-radius: 11px !important;
  color: #c2cdf3 !important; font-weight: 600 !important;
}
.stage {
  border-radius: 16px; overflow: hidden; border: 1px solid rgba(129,140,248,0.18);
  background: #0a0f22;
}
footer { margin-top: 26px; color: #8ea0cd; font-size: 13px; text-align: center; line-height: 1.8; }
footer .pill {
  display:inline-block; margin: 4px; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
}
a { color: #7dd3fc !important; }
"""


def _hero_html():
    return """
<div id="hero">
  <span class="eyebrow">◆ Computer Vision · Depth Super-Resolution</span>
  <h1>Edge-Preserving Depth Map<br>Upsampling &amp; Refinement</h1>
  <p class="sub">
    Turn a sparse or low-resolution depth map into a dense, high-resolution one that
    <b>keeps object boundaries sharp</b>. Upload a high-res RGB image and a low-res depth
    map — the hybrid pipeline (classical + deep fusion + measurement consistency) runs
    automatically and visualises every stage.
  </p>
  <div class="badges">
    <span class="badge"><b>Classical</b> · JBU + guided + boundary refinement</span>
    <span class="badge"><b>Deep</b> · SGNet guided super-resolution</span>
    <span class="badge"><b>Hybrid</b> · confidence fusion + consistency</span>
    <span class="badge"><b>Metrics</b> · RMSE · PSNR · SSIM · Edge-F1</span>
  </div>
</div>
"""


def _metrics_html(metrics):
    if not metrics:
        return ""
    order = [
        ("rmse", "RMSE", "{:.4f}"),
        ("mae", "MAE", "{:.4f}"),
        ("psnr", "PSNR", "{:.2f}"),
        ("ssim", "SSIM", "{:.4f}"),
        ("boundary_rmse", "Boundary RMSE", "{:.4f}"),
        ("edge_f1", "Edge F1", "{:.3f}"),
    ]
    cards = []
    for key, name, fmt in order:
        if key in metrics:
            val = metrics[key]
            txt = fmt.format(val) if val == val else "—"
            cards.append(f"<div class='metric'><div class='val'>{txt}</div><div class='name'>{name}</div></div>")
    return f"<div class='metric-strip'>{''.join(cards)}</div>" if cards else ""


def _footer_html():
    return """
<footer>
  <div>
    <span class="pill">A1 hole-fill</span>→<span class="pill">A2 bicubic</span>→
    <span class="pill">A3 joint bilateral</span>→<span class="pill">A4 boundary refinement</span>
  </div>
  <div style="margin-top:8px">
    Invalid depth (≤ 0, NaN, Inf) is never treated as a measurement ·
    scales x2/x4/x8/x16 · deep branch supports x4/x8/x16
  </div>
</footer>
"""


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
    """Write synthetic example files to disk and return gr.Examples rows."""
    ex_dir = os.path.join(tempfile.gettempdir(), "p4_examples")
    os.makedirs(ex_dir, exist_ok=True)
    rows = []
    for variant in (0, 1):
        rgb, gt = _synthetic_scene(256, variant)
        h, w = gt.shape
        lr = cv2.resize(gt, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        if variant == 1:  # punch sparse holes to demo hole-filling
            hole = np.zeros(lr.shape, bool)
            rng = np.random.default_rng(7)
            hole[rng.integers(0, lr.shape[0], 120), rng.integers(0, lr.shape[1], 120)] = True
            lr = lr.copy()
            lr[hole] = 0.0
        rgb_p = os.path.join(ex_dir, f"scene{variant}_rgb.png")
        lr_p = os.path.join(ex_dir, f"scene{variant}_lr.npy")
        gt_p = os.path.join(ex_dir, f"scene{variant}_gt.npy")
        cv2.imwrite(rgb_p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(lr_p, lr.astype(np.float32))
        np.save(gt_p, gt.astype(np.float32))
        rows.append([rgb_p, lr_p, scale])
    return rows


def _gradio_callback(rgb, depth_file, mask_file, gt_file, scale, make_comparison, depth_scale):
    import gradio as gr

    try:
        o = process(rgb, depth_file, mask_file, scale, gt_file, make_comparison,
                    depth_scale=depth_scale or 1.0)
    except (ValueError, NotImplementedError, ImportError, FileNotFoundError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    status_md = "```\n" + o["status"] + "\n```"
    metrics_html = _metrics_html(o.get("metrics"))
    return (
        status_md,
        metrics_html,
        o["depth"],
        o["boundaries"],
        o["conf_classical"],
        o["conf_deep"],
        gr.update(value=o.get("comparison"), visible=o.get("comparison") is not None),
        gr.update(value=o.get("error"), visible=o.get("error") is not None),
        o["files"],
    )


def build_app():
    import gradio as gr

    cfg = load_config()
    theme = gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
    )

    warnings.filterwarnings("ignore", message="The parameters have been moved")
    with gr.Blocks(theme=theme, css=_CUSTOM_CSS, title="P4 — Edge-Preserving Depth Upsampling") as demo:
        gr.HTML(_hero_html())

        with gr.Row(equal_height=True):
            # ---------------- Inputs ----------------
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.HTML("<div class='card-title'><span class='dot'>◆</span> Inputs</div>")
                rgb_in = gr.Image(label="RGB guidance (high-res)", type="numpy", height=260)
                depth_in = gr.File(label="Low-res depth map (.npy or .png)")
                with gr.Accordion("Optional & advanced", open=False):
                    mask_in = gr.File(label="Validity mask (.npy / .png)")
                    gt_in = gr.File(label="HR ground truth (.npy / .png) — enables metrics")
                    scale_in = gr.Radio(
                        choices=cfg["app"]["scales"], value=cfg["app"]["default_scale"],
                        label="Upsampling factor", info="x2 is classical-only; deep supports x4/x8/x16",
                    )
                    depth_scale_in = gr.Number(
                        value=1.0, label="Depth multiplier",
                        info="0.001 converts a millimetre PNG to metres",
                    )
                    comparison_in = gr.Checkbox(
                        value=True, label="Build side-by-side panel (LR · classical · deep · hybrid · GT)",
                    )
                run_btn = gr.Button("▲  Reconstruct depth", elem_id="cta", variant="primary")

                gr.Examples(
                    examples=_make_examples(cfg["app"]["default_scale"]),
                    inputs=[rgb_in, depth_in, scale_in],
                    label="Try an example",
                    examples_per_page=4,
                )

            # ---------------- Outputs ----------------
            with gr.Column(scale=2, elem_classes=["card"]):
                gr.HTML("<div class='card-title'><span class='dot'>◆</span> Results</div>")
                metrics_html = gr.HTML("")
                status = gr.Markdown("Upload an RGB image and a depth map, then hit **Reconstruct**.",
                                     elem_classes=["status-box"])

                with gr.Tabs(elem_classes=["tab-nav"]):
                    with gr.Tab("Depth", elem_classes=["tab-item"]):
                        out_depth = gr.Image(label="Hybrid output depth", elem_classes=["stage"], height=420)
                    with gr.Tab("Boundaries", elem_classes=["tab-item"]):
                        out_boundaries = gr.Image(label="Depth boundaries", elem_classes=["stage"], height=420)
                    with gr.Tab("Branch confidence", elem_classes=["tab-item"]):
                        with gr.Row():
                            out_wc = gr.Image(label="Classical confidence", elem_classes=["stage"], height=320)
                            out_wd = gr.Image(label="Deep confidence", elem_classes=["stage"], height=320)
                    with gr.Tab("Comparison", elem_classes=["tab-item"]):
                        out_comparison = gr.Image(label="Side-by-side", elem_classes=["stage"], height=380, visible=True)
                    with gr.Tab("Error map", elem_classes=["tab-item"]):
                        out_error = gr.Image(label="|pred − GT|", elem_classes=["stage"], height=380, visible=False)

                out_files = gr.File(label="Download (PNG 16-bit · NPY float32)", file_count="multiple",
                                    interactive=False)

        gr.HTML(_footer_html())

        run_btn.click(
            _gradio_callback,
            inputs=[rgb_in, depth_in, mask_in, gt_in, scale_in, comparison_in, depth_scale_in],
            outputs=[status, metrics_html, out_depth, out_boundaries, out_wc, out_wd,
                     out_comparison, out_error, out_files],
        )
    return demo


def launch(**kwargs):
    demo = build_app()
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
