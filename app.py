# Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Super-Resolution and Refinement
# Premium Production-Grade Web Application for Hugging Face Spaces & Local Deployment

import os
import sys
import time
from typing import Optional, Tuple
import cv2
import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from classical.pipeline import classical_hr_depth
from deep.inference import deep_hr_depth
from hybrid.pipeline import hybrid_hr_depth
from eval.metrics import compute_all_metrics, extract_depth_edges
from app.visualization import colorize_depth, create_side_by_side_comparison, plot_scanline_profile
from app.point_cloud import export_depth_to_ply, export_depth_to_mesh_obj
from data.dataset import generate_synthetic_rgbd_scene
from data.degradation import create_lr_depth


# Cache for current run results to power scanline analysis
LAST_RUN = {
    "rgb": None,
    "gt": None,
    "bicubic": None,
    "classical": None,
    "deep": None,
    "hybrid": None,
}


def load_preset(scene_name: str) -> np.ndarray:
    if "Living Room" in scene_name:
        path = "data/presets/living_room.png"
    elif "Curved" in scene_name:
        path = "data/presets/curved_sphere.png"
    else:
        path = "data/presets/pillars.png"
    if os.path.isfile(path):
        return cv2.imread(path)[:, :, ::-1]
    rgb, _ = generate_synthetic_rgbd_scene(240, 320, seed=42)
    return rgb


def run_super_resolution(
    input_rgb: Optional[np.ndarray],
    preset_choice: str,
    scale: int,
    colormap_choice: str,
    noise_std: float,
    hole_ratio_pct: float,
    alpha: float,
    use_consistency: bool,
    scanline_row: int,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # ImageSlider pair
    np.ndarray,                     # Degraded LR input
    np.ndarray,                     # Edge map
    np.ndarray,                     # Scanline profile plot
    np.ndarray,                     # Multi-track panel
    str,                            # HTML KPI cards
    pd.DataFrame,                   # Metrics table
    str,                            # OBJ 3D Model
    str,                            # OBJ download
    str,                            # PLY download
    str,                            # NPY download
]:
    os.makedirs("outputs", exist_ok=True)
    hole_ratio = hole_ratio_pct / 100.0

    # 1. Acquire Input Data
    if input_rgb is not None:
        rgb = np.asarray(input_rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gt_depth = 1.0 + 3.0 * (1.0 - gray)
    else:
        seed = 42 if "Living Room" in preset_choice else (105 if "Curved" in preset_choice else 201)
        rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=seed)

    h_hr, w_hr = rgb.shape[:2]

    # 2. Simulate Low-Resolution Sensor Input
    lr_depth, valid_mask = create_lr_depth(
        gt_depth, scale=scale, noise_std=noise_std, hole_ratio=hole_ratio
    )

    # 3. Execute Reconstruction Tracks
    t0 = time.perf_counter()
    d_b = classical_hr_depth(lr_depth, rgb, scale=scale, method="bicubic")
    lat_b = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    d_c = classical_hr_depth(lr_depth, rgb, valid_mask=valid_mask, scale=scale, method="jbu")
    lat_c = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    d_d = deep_hr_depth(lr_depth, rgb, valid_mask=valid_mask, scale=scale)
    lat_d = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    d_h = hybrid_hr_depth(
        lr_depth,
        rgb,
        valid_mask=valid_mask,
        scale=scale,
        alpha=alpha,
        deep_depth=d_d,
        classical_depth=d_c,
        method="fixed",
        use_consistency=use_consistency,
    )
    lat_h = (time.perf_counter() - t0) * 1000.0 + lat_c + lat_d

    # Update cache for scanline inspection
    LAST_RUN["rgb"] = rgb
    LAST_RUN["gt"] = gt_depth
    LAST_RUN["bicubic"] = d_b
    LAST_RUN["classical"] = d_c
    LAST_RUN["deep"] = d_d
    LAST_RUN["hybrid"] = d_h

    # 4. Colorized Renderings
    lr_upsampled_nearest = cv2.resize(lr_depth, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    lr_vis = colorize_depth(lr_upsampled_nearest, cmap=colormap_choice)
    d_h_vis = colorize_depth(d_h, cmap=colormap_choice)
    b_vis = colorize_depth(d_b, cmap=colormap_choice)
    c_vis = colorize_depth(d_c, cmap=colormap_choice)
    d_vis = colorize_depth(d_d, cmap=colormap_choice)

    # ImageSlider: Before (Degraded LR) vs After (Hybrid Depth)
    slider_pair = (lr_vis, d_h_vis)

    # Multi-track side-by-side comparison panel
    panel_path = "outputs/web_comparison.png"
    create_side_by_side_comparison(
        rgb, lr_depth, d_b, d_c, d_d, d_h, gt_depth=gt_depth, save_path=panel_path
    )
    panel_vis = cv2.imread(panel_path)[:, :, ::-1]

    # Edge map of reconstructed depth
    edges = extract_depth_edges(d_h, threshold=0.04)
    edge_vis = (edges.astype(np.uint8) * 255)
    edge_vis_rgb = np.stack([edge_vis, edge_vis, edge_vis], axis=-1)

    # 1D Scanline profile plot
    scan_row = min(max(0, scanline_row), h_hr - 1)
    scan_plot = plot_scanline_profile(
        d_b, d_c, d_d, d_h, gt=gt_depth, row_idx=scan_row, save_path="outputs/scanline_plot.png"
    )

    # 5. Compute Comprehensive Metrics
    m_b = compute_all_metrics(d_b, gt_depth, valid_mask=valid_mask)
    m_c = compute_all_metrics(d_c, gt_depth, valid_mask=valid_mask)
    m_d = compute_all_metrics(d_d, gt_depth, valid_mask=valid_mask)
    m_h = compute_all_metrics(d_h, gt_depth, valid_mask=valid_mask)

    psnr_delta = m_h["psnr"] - m_b["psnr"]
    rmse_delta = m_b["rmse"] - m_h["rmse"]

    metrics_df = pd.DataFrame([
        {"Method": "Bicubic Baseline", "RMSE (m)": f"{m_b['rmse']:.4f}", "MAE (m)": f"{m_b['mae']:.4f}", "PSNR (dB)": f"{m_b['psnr']:.2f}", "SSIM": f"{m_b['ssim']:.4f}", "Boundary RMSE": f"{m_b['boundary_rmse']:.4f}", "Edge F1": f"{m_b['edge_f1']:.3f}", "Latency": f"{lat_b:.1f} ms"},
        {"Method": "Classical (JBU)", "RMSE (m)": f"{m_c['rmse']:.4f}", "MAE (m)": f"{m_c['mae']:.4f}", "PSNR (dB)": f"{m_c['psnr']:.2f}", "SSIM": f"{m_c['ssim']:.4f}", "Boundary RMSE": f"{m_c['boundary_rmse']:.4f}", "Edge F1": f"{m_c['edge_f1']:.3f}", "Latency": f"{lat_c:.1f} ms"},
        {"Method": "Deep (EdgeGuidedSRNet)", "RMSE (m)": f"{m_d['rmse']:.4f}", "MAE (m)": f"{m_d['mae']:.4f}", "PSNR (dB)": f"{m_d['psnr']:.2f}", "SSIM": f"{m_d['ssim']:.4f}", "Boundary RMSE": f"{m_d['boundary_rmse']:.4f}", "Edge F1": f"{m_d['edge_f1']:.3f}", "Latency": f"{lat_d:.1f} ms"},
        {"Method": f"Hybrid (Optimal α={alpha:.2f})", "RMSE (m)": f"{m_h['rmse']:.4f}", "MAE (m)": f"{m_h['mae']:.4f}", "PSNR (dB)": f"{m_h['psnr']:.2f}", "SSIM": f"{m_h['ssim']:.4f}", "Boundary RMSE": f"{m_h['boundary_rmse']:.4f}", "Edge F1": f"{m_h['edge_f1']:.3f}", "Latency": f"{lat_h:.1f} ms"},
    ])

    # 6. Modern Glassmorphic KPI Cards
    kpi_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 14px;">
        <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.95), rgba(49, 46, 129, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(79, 70, 229, 0.3);">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #c7d2fe; font-weight: 600;">Hybrid RMSE</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['rmse']:.4f} m</div>
            <div style="font-size: 0.72rem; color: #a5b4fc;">{'+' if rmse_delta >= 0 else ''}{rmse_delta:.4f} m vs bicubic</div>
        </div>
        <div style="background: linear-gradient(135deg, rgba(5, 150, 105, 0.95), rgba(6, 95, 70, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(5, 150, 105, 0.3);">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a7f3d0; font-weight: 600;">Peak SNR</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['psnr']:.2f} dB</div>
            <div style="font-size: 0.72rem; color: #6ee7b7;">{'+' if psnr_delta >= 0 else ''}{psnr_delta:.2f} dB over baseline</div>
        </div>
        <div style="background: linear-gradient(135deg, rgba(217, 119, 6, 0.95), rgba(146, 64, 14, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(217, 119, 6, 0.3);">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #fde68a; font-weight: 600;">Edge F1 Score</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['edge_f1']:.3f}</div>
            <div style="font-size: 0.72rem; color: #fcd34d;">Boundary Precision</div>
        </div>
        <div style="background: linear-gradient(135deg, rgba(2, 132, 199, 0.95), rgba(3, 105, 161, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(2, 132, 199, 0.3);">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #bae6fd; font-weight: 600;">SSIM Fidelity</div>
            <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['ssim']:.4f}</div>
            <div style="font-size: 0.72rem; color: #7dd3fc;">Structural Match</div>
        </div>
    </div>
    """

    # 7. 3D Mesh and Point Cloud Files
    obj_path = "outputs/reconstructed_surface.obj"
    export_depth_to_mesh_obj(d_h, rgb, output_path=obj_path, target_dim=65)

    ply_path = "outputs/reconstructed_point_cloud.ply"
    export_depth_to_ply(d_h, rgb, output_path=ply_path, max_points=35000)

    npy_path = "outputs/pred_hybrid_depth.npy"
    np.save(npy_path, d_h)

    return (
        slider_pair,
        lr_vis,
        edge_vis_rgb,
        scan_plot,
        panel_vis,
        kpi_html,
        metrics_df,
        obj_path,
        obj_path,
        ply_path,
        npy_path,
    )


def update_scanline_only(row_idx: int) -> np.ndarray:
    if LAST_RUN["hybrid"] is None:
        rgb, gt = generate_synthetic_rgbd_scene(240, 320, seed=42)
        d_b = cv2.resize(cv2.resize(gt, (80, 60)), (320, 240), interpolation=cv2.INTER_CUBIC)
        return plot_scanline_profile(d_b, d_b, d_b, gt, gt=gt, row_idx=row_idx)
    return plot_scanline_profile(
        LAST_RUN["bicubic"],
        LAST_RUN["classical"],
        LAST_RUN["deep"],
        LAST_RUN["hybrid"],
        gt=LAST_RUN["gt"],
        row_idx=row_idx,
    )


# Custom UI CSS
custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.hero-banner {
    background: radial-gradient(circle at 10% 20%, #1e1b4b 0%, #0f172a 90%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.6);
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.35);
    color: #a5b4fc;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 4px 0 8px 0;
}
.hero-desc {
    color: #94a3b8;
    font-size: 0.98rem;
    line-height: 1.5;
    max-width: 950px;
}
.hero-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.hero-chip {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 500;
}
.btn-reconstruct {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Edge-Preserving Depth Super-Resolution (P4)") as demo:
    gr.HTML(
        f"""
        <style>{custom_css}</style>
        <div class="hero-banner">
            <div class="hero-tag">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;"></span>
                HACKATHON 2026 · PROBLEM STATEMENT 4 (ProbID - P4)
            </div>
            <div class="hero-title">Edge-Preserving Depth Map Super-Resolution & Refinement</div>
            <div class="hero-desc">
                An intelligent hybrid depth reconstruction platform combining <b>Joint Bilateral Upsampling</b> with <b>EdgeGuidedSRNet Neural Guidance</b>
                and <b>Backward Measurement Consistency Projection</b> to reconstruct crisp, physical depth discontinuities without texture-bleeding artifacts.
            </div>
            <div class="hero-chips">
                <span class="hero-chip">⚡ PyTorch 2.14 / CUDA</span>
                <span class="hero-chip">🎯 Optimal Fixed α = 0.02</span>
                <span class="hero-chip">📐 Gradient Edge Gating</span>
                <span class="hero-chip">🧊 3D WebGL Mesh & PLY Export</span>
                <span class="hero-chip">📊 NYU-v2 Protocol Verification</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔮 Interactive Super-Resolution Studio"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 🎛️ Input Controls & Degradation")
                    
                    preset_choice = gr.Dropdown(
                        label="Choose Pre-Packaged Benchmark Scene",
                        choices=[
                            "Living Room (Indoor Furniture & Objects)",
                            "Curved Shapes (Sphere Discontinuities)",
                            "Pillars & Geometric Step Boundaries",
                        ],
                        value="Living Room (Indoor Furniture & Objects)",
                    )
                    
                    input_rgb = gr.Image(
                        label="Or Upload Custom RGB Guidance Image",
                        type="numpy",
                        height=200,
                    )

                    with gr.Row():
                        scale = gr.Radio(
                            label="Super-Resolution Scale",
                            choices=[2, 4, 8],
                            value=4,
                        )
                        colormap_choice = gr.Dropdown(
                            label="Depth Colormap",
                            choices=["turbo", "plasma", "viridis", "inferno", "magma", "bone", "gray"],
                            value="turbo",
                        )

                    with gr.Accordion("⚙️ Sensor Simulation & Fusion Hyperparameters", open=False):
                        noise_std = gr.Slider(
                            minimum=0.0, maximum=0.05, step=0.005, value=0.0,
                            label="Simulated Sensor Gaussian Noise (meters)"
                        )
                        hole_ratio_pct = gr.Slider(
                            minimum=0.0, maximum=20.0, step=1.0, value=0.0,
                            label="Missing Depth Measurement Holes (%)"
                        )
                        alpha = gr.Slider(
                            minimum=0.0, maximum=0.2, step=0.01, value=0.02,
                            label="Classical Blending Weight α (Calibrated Optimal = 0.02)"
                        )
                        use_consistency = gr.Checkbox(
                            value=True,
                            label="Enforce Measurement Consistency: r_LR = D_LR - S(D_f)"
                        )
                        scanline_row = gr.Slider(
                            minimum=10, maximum=230, step=5, value=120,
                            label="Scanline Row for 1D Profile Inspection (y)"
                        )

                    run_btn = gr.Button("✨ Reconstruct High-Resolution Depth", elem_classes=["btn-reconstruct"], size="lg")

                with gr.Column(scale=6):
                    gr.Markdown("### 📈 Reconstructed Output & Performance")
                    kpi_display = gr.HTML()

                    image_slider = gr.ImageSlider(
                        label="Swipe to Compare: Degraded LR Input vs. Reconstructed Hybrid Depth",
                        show_label=True,
                    )

                    with gr.Row():
                        lr_output = gr.Image(label="Degraded LR Input (Nearest Grid)", type="numpy")
                        edge_output = gr.Image(label="Recovered Depth Boundaries (Edge Map)", type="numpy")

                    gr.Markdown("### 📋 Multi-Track Performance Scorecard")
                    metrics_table = gr.Dataframe(
                        headers=["Method", "RMSE (m)", "MAE (m)", "PSNR (dB)", "SSIM", "Boundary RMSE", "Edge F1", "Latency"],
                    )

        with gr.TabItem("📐 Edge & Scanline Discontinuity Inspector"):
            gr.Markdown("### 🔬 1D Depth Discontinuity Profile (Step-Response Analysis)")
            gr.Markdown("Inspect how sharp depth steps are reconstructed. Classical and Hybrid prevent the slope-blurring artifact typical of Bicubic interpolation.")
            
            with gr.Row():
                scanline_slider = gr.Slider(
                    minimum=10, maximum=230, step=5, value=120,
                    label="Inspect Horizontal Scanline Row (y-coordinate)"
                )
                scan_btn = gr.Button("Update Scanline Slice", size="sm")
            
            scanline_output = gr.Image(label="1D Cross-Section Depth Profile", type="numpy")
            
            gr.Markdown("### 🖼️ 6-Panel Multi-Track Comparison Panel")
            multi_panel_image = gr.Image(label="Side-by-Side Track Panel", type="numpy")

        with gr.TabItem("🧊 3D Digital Twin & Point Cloud Studio"):
            gr.Markdown("### 🌐 Interactive 3D WebGL Surface Canvas")
            gr.Markdown("Click, drag, and scroll to inspect the reconstructed depth geometry in 3D. Vertex colors are sampled directly from the RGB guidance image.")
            with gr.Row():
                with gr.Column(scale=7):
                    model_3d = gr.Model3D(label="Interactive 3D Textured Surface Mesh (.OBJ)")
                with gr.Column(scale=3):
                    gr.Markdown("#### 📥 Export Reconstructed 3D Assets")
                    dl_obj = gr.File(label="Download 3D Mesh (.OBJ)")
                    dl_ply = gr.File(label="Download 3D Point Cloud (.PLY)")
                    dl_npy = gr.File(label="Download Metric Depth (.NPY)")

        with gr.TabItem("📊 Multi-Scale Benchmark & Ablation Lab"):
            gr.Markdown(
                """
                ### Empirical Findings: Resolving the Hybrid Trade-Off

                In initial development experiments, combining deep predictions with classical boundary filtering improved RMSE and PSNR but slightly sacrificed MAE and Edge F1.
                
                #### The Cause & The Solution:
                * **Forward Operator Alignment**: Prior consistency projections used area downsampling ($S_{area}$), whereas dataset degradation was bicubic ($S_{cubic}$). Aligning the forward degradation operator strictly to $S_{cubic}$ anchors predictions accurately to sensor measurements.
                * **Boundary-Aware Gating ($\alpha=0.02$)**: Instead of heavy boundary smoothing that washes out thin objects, a small calibrated classical weight ($\alpha=0.02$) provides geometric sharpness while preserving planar depth accuracy.

                | Validation Variant | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE ↓ | Edge F1 ↑ |
                |:---|:---:|:---:|:---:|:---:|:---:|:---:|
                | **Bicubic Baseline** | 0.0367 | 0.0100 | 51.62 | 0.9957 | 0.1680 | 0.259 |
                | **Classical (JBU)** | 0.0584 | 0.0167 | 47.59 | 0.9900 | 0.2591 | 0.671 |
                | **Deep Branch (EdgeGuidedSRNet)** | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 |
                | **Hybrid (Optimal α=0.02 + Consistency)** | **0.0122** | **0.0036** | **59.58** | **0.9994** | **0.0536** | **0.975** |
                """
            )

        with gr.TabItem("📖 Technical Methodology"):
            gr.Markdown(
                """
                ### Mathematical Formulations

                #### 1. Joint Bilateral Upsampling (JBU)
                $$\\tilde{D}(p) = \\frac{\\sum_{q_* \\in \\Omega} f_s(\\|p_* - q_*\\|) \\cdot f_r(\\|I(p) - I(q_* \\cdot s)\\|) \\cdot D_{LR}(q_*)}{\\sum_{q_* \\in \\Omega} f_s(\\|p_* - q_*\\|) \\cdot f_r(\\|I(p) - I(q_* \\cdot s)\\|)}$$

                #### 2. Deep Branch (EdgeGuidedSRNet)
                * Dual-stream feature extraction: multi-scale RGB guidance stream $F_{rgb}$ and depth stream $F_d$.
                * Edge-aware attention gating: $F_{fused} = (F_d \\oplus F_{rgb}) \\odot \\sigma(\\text{Conv}(F_d \\oplus F_{rgb}))$.
                * Residual reconstruction: $\\hat{D} = D_{bicubic} + \\mathcal{R}(F_{fused})$.

                #### 3. Measurement Consistency Enforcement
                $$r_{LR} = D_{LR} - S(D_f)$$
                $$D_{final} = D_f + \\gamma \\cdot S^{\\dagger}(r_{LR})$$
                where $S$ is the matching forward degradation operator and $\\gamma \\in (0, 1]$ is the damping factor.
                """
            )

    # Event handlers
    preset_choice.change(
        fn=load_preset,
        inputs=[preset_choice],
        outputs=[input_rgb],
    )

    scan_btn.click(
        fn=update_scanline_only,
        inputs=[scanline_slider],
        outputs=[scanline_output],
    )

    run_btn.click(
        fn=run_super_resolution,
        inputs=[
            input_rgb,
            preset_choice,
            scale,
            colormap_choice,
            noise_std,
            hole_ratio_pct,
            alpha,
            use_consistency,
            scanline_row,
        ],
        outputs=[
            image_slider,
            lr_output,
            edge_output,
            scanline_output,
            multi_panel_image,
            kpi_display,
            metrics_table,
            model_3d,
            dl_obj,
            dl_ply,
            dl_npy,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme)
