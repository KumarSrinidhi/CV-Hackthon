# Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Super-Resolution and Refinement
# Premium Production-Grade Web Application for Hugging Face Spaces & Local Deployment

import os
import re
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


# Global cache for current run results to power scanline analysis
LAST_RUN = {
    "rgb": None,
    "gt": None,
    "bicubic": None,
    "classical": None,
    "deep": None,
    "hybrid": None,
    "lr_depth": None,
    "valid_mask": None,
}

TESTSET_PRESETS = [
    "⭐ [Official Testset] Sample 00 · Indoor Living Space (512×512)",
    "🏢 [Official Testset] Sample 01 · Office Interior (512×512)",
    "🏫 [Official Testset] Sample 02 · Classroom Hall (512×512)",
    "📚 [Official Testset] Sample 03 · Living Room Shelves (512×512)",
    "🖥️ [Official Testset] Sample 04 · Dual-Monitor Desk Station (512×512)",
    "🛋️ [Official Testset] Sample 05 · Lounge Area (512×512)",
    "🔴 [Geometric Preset] Curved 3D Shaded Sphere (240×320)",
    "🏛️ [Geometric Preset] Architectural 3-Pillar Steps (240×320)",
    "🛋️ [Geometric Preset] Stylized Living Room Scene (240×320)",
    "📁 [Custom Upload] Upload Your Own RGB & Depth Files",
]


def get_sample_data(
    preset_choice: str,
    scale: int = 4,
    input_rgb: Optional[np.ndarray] = None,
    input_depth_file=None,
    noise_std: float = 0.0,
    hole_ratio: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Acquire paired RGB, GT Depth, LR Depth, and Valid Mask for the chosen scene.
    """
    testset_dir = os.path.join(os.path.dirname(__file__), "data", "depthsr_testset")

    # 1. Official Testset Samples
    if "Official Testset" in preset_choice:
        m = re.search(r"sample\s*(\d+)", preset_choice.lower())
        sample_name = f"sample_{int(m.group(1)):02d}" if m else "sample_00"
        sdir = os.path.join(testset_dir, sample_name)
        if os.path.isdir(sdir):
            rgb_bgr = cv2.imread(os.path.join(sdir, "rgb.png"))
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            gt_depth = np.load(os.path.join(sdir, "depth_hr.npy")).astype(np.float32)

            lr_path = os.path.join(sdir, f"lr_depth_{scale}x.npy")
            mask_path = os.path.join(sdir, f"lr_mask_{scale}x.npy")

            if os.path.exists(lr_path):
                lr_depth = np.load(lr_path).astype(np.float32)
            else:
                lr_depth, _ = create_lr_depth(gt_depth, scale=scale)

            if os.path.exists(mask_path):
                valid_mask = np.load(mask_path).astype(np.uint8)
            else:
                valid_mask = (lr_depth > 0).astype(np.uint8)

            # Optional extra noise / holes if user explicitly requested in sliders
            if noise_std > 0.0:
                lr_depth = lr_depth + np.random.normal(0, noise_std, lr_depth.shape).astype(np.float32)
            if hole_ratio > 0.0:
                hole_mask = np.random.rand(*lr_depth.shape) >= hole_ratio
                valid_mask = valid_mask & hole_mask.astype(np.uint8)
                lr_depth[valid_mask == 0] = 0.0

            return rgb, gt_depth, lr_depth, valid_mask

    # 2. Custom User Upload
    if "Custom Upload" in preset_choice and input_rgb is not None:
        rgb = np.asarray(input_rgb, dtype=np.uint8)
        if input_depth_file is not None:
            path = input_depth_file.name if hasattr(input_depth_file, "name") else str(input_depth_file)
            if path.endswith(".npy"):
                loaded = np.load(path).astype(np.float32)
            else:
                loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            if loaded.shape[0] < rgb.shape[0]:
                lr_depth = loaded
                valid_mask = (lr_depth > 0).astype(np.uint8)
                gt_depth = None
                return rgb, gt_depth, lr_depth, valid_mask
            else:
                gt_depth = loaded
                lr_depth, valid_mask = create_lr_depth(gt_depth, scale=scale, noise_std=noise_std, hole_ratio=hole_ratio)
                return rgb, gt_depth, lr_depth, valid_mask
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gt_depth = 1.0 + 3.0 * (1.0 - gray)
            lr_depth, valid_mask = create_lr_depth(gt_depth, scale=scale, noise_std=noise_std, hole_ratio=hole_ratio)
            return rgb, gt_depth, lr_depth, valid_mask

    # 3. Geometric Benchmark Presets
    if "Curved" in preset_choice:
        path_rgb = os.path.join(os.path.dirname(__file__), "data", "presets", "curved_sphere.png")
        path_depth = os.path.join(os.path.dirname(__file__), "data", "presets", "curved_sphere_depth.npy")
    elif "Pillars" in preset_choice:
        path_rgb = os.path.join(os.path.dirname(__file__), "data", "presets", "pillars.png")
        path_depth = os.path.join(os.path.dirname(__file__), "data", "presets", "pillars_depth.npy")
    else:
        path_rgb = os.path.join(os.path.dirname(__file__), "data", "presets", "living_room.png")
        path_depth = os.path.join(os.path.dirname(__file__), "data", "presets", "living_room_depth.npy")

    if os.path.exists(path_rgb) and os.path.exists(path_depth):
        rgb_bgr = cv2.imread(path_rgb)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        gt_depth = np.load(path_depth).astype(np.float32)
    else:
        seed = 201 if "Pillars" in preset_choice else (105 if "Curved" in preset_choice else 42)
        rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=seed)

    lr_depth, valid_mask = create_lr_depth(gt_depth, scale=scale, noise_std=noise_std, hole_ratio=hole_ratio)
    return rgb, gt_depth, lr_depth, valid_mask


def on_preset_change(choice: str, scale: int) -> np.ndarray:
    rgb, _, _, _ = get_sample_data(choice, scale=scale)
    return rgb


def run_super_resolution(
    preset_choice: str,
    input_rgb: Optional[np.ndarray],
    input_depth_file,
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
    preset_rgb, preset_gt, preset_lr, preset_mask = get_sample_data(
        preset_choice,
        scale=scale,
        noise_std=noise_std,
        hole_ratio=hole_ratio,
    )

    is_custom = ("Custom Upload" in preset_choice) or (
        input_rgb is not None and not np.array_equal(input_rgb, preset_rgb)
    )

    if is_custom and input_rgb is not None:
        rgb, gt_depth, lr_depth, valid_mask = get_sample_data(
            "Custom Upload",
            scale=scale,
            input_rgb=input_rgb,
            input_depth_file=input_depth_file,
            noise_std=noise_std,
            hole_ratio=hole_ratio,
        )
    else:
        rgb, gt_depth, lr_depth, valid_mask = preset_rgb, preset_gt, preset_lr, preset_mask

    h_hr, w_hr = rgb.shape[:2]

    # 2. Execute Reconstruction Tracks
    t0 = time.perf_counter()
    d_b = classical_hr_depth(lr_depth, rgb, valid_mask=valid_mask, scale=scale, method="bicubic")
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
    LAST_RUN["lr_depth"] = lr_depth
    LAST_RUN["valid_mask"] = valid_mask

    # 3. Colorized Renderings
    lr_upsampled_nearest = cv2.resize(lr_depth, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    lr_vis = colorize_depth(lr_upsampled_nearest, cmap=colormap_choice)
    d_h_vis = colorize_depth(d_h, cmap=colormap_choice)

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

    # 4. Evaluation Metrics
    if gt_depth is not None:
        m_b = compute_all_metrics(d_b, gt_depth, valid_mask=valid_mask)
        m_c = compute_all_metrics(d_c, gt_depth, valid_mask=valid_mask)
        m_d = compute_all_metrics(d_d, gt_depth, valid_mask=valid_mask)
        m_h = compute_all_metrics(d_h, gt_depth, valid_mask=valid_mask)

        psnr_gain = m_h["psnr"] - m_b["psnr"]
        rmse_gain = m_b["rmse"] - m_h["rmse"]

        metrics_df = pd.DataFrame([
            {"Method": "Bicubic Baseline", "RMSE (m)": f"{m_b['rmse']:.4f}", "MAE (m)": f"{m_b['mae']:.4f}", "PSNR (dB)": f"{m_b['psnr']:.2f}", "SSIM": f"{m_b['ssim']:.4f}", "Boundary RMSE": f"{m_b['boundary_rmse']:.4f}", "Edge F1": f"{m_b['edge_f1']:.3f}", "Latency": f"{lat_b:.1f} ms"},
            {"Method": "Classical (JBU)", "RMSE (m)": f"{m_c['rmse']:.4f}", "MAE (m)": f"{m_c['mae']:.4f}", "PSNR (dB)": f"{m_c['psnr']:.2f}", "SSIM": f"{m_c['ssim']:.4f}", "Boundary RMSE": f"{m_c['boundary_rmse']:.4f}", "Edge F1": f"{m_c['edge_f1']:.3f}", "Latency": f"{lat_c:.1f} ms"},
            {"Method": "Deep (EdgeGuidedSRNet)", "RMSE (m)": f"{m_d['rmse']:.4f}", "MAE (m)": f"{m_d['mae']:.4f}", "PSNR (dB)": f"{m_d['psnr']:.2f}", "SSIM": f"{m_d['ssim']:.4f}", "Boundary RMSE": f"{m_d['boundary_rmse']:.4f}", "Edge F1": f"{m_d['edge_f1']:.3f}", "Latency": f"{lat_d:.1f} ms"},
            {"Method": f"Hybrid (Optimal α={alpha:.2f} + Consistency)", "RMSE (m)": f"{m_h['rmse']:.4f}", "MAE (m)": f"{m_h['mae']:.4f}", "PSNR (dB)": f"{m_h['psnr']:.2f}", "SSIM": f"{m_h['ssim']:.4f}", "Boundary RMSE": f"{m_h['boundary_rmse']:.4f}", "Edge F1": f"{m_h['edge_f1']:.3f}", "Latency": f"{lat_h:.1f} ms"},
        ])

        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 12px;">
            <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.95), rgba(49, 46, 129, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(79, 70, 229, 0.3);">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #c7d2fe; font-weight: 600;">Hybrid RMSE</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['rmse']:.4f} m</div>
                <div style="font-size: 0.72rem; color: #a5b4fc;">{'+' if rmse_gain >= 0 else ''}{rmse_gain:.4f} m vs bicubic</div>
            </div>
            <div style="background: linear-gradient(135deg, rgba(5, 150, 105, 0.95), rgba(6, 95, 70, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(5, 150, 105, 0.3);">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #a7f3d0; font-weight: 600;">Peak SNR</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['psnr']:.2f} dB</div>
                <div style="font-size: 0.72rem; color: #6ee7b7;">{'+' if psnr_gain >= 0 else ''}{psnr_gain:.2f} dB gain</div>
            </div>
            <div style="background: linear-gradient(135deg, rgba(217, 119, 6, 0.95), rgba(146, 64, 14, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(217, 119, 6, 0.3);">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #fde68a; font-weight: 600;">Edge F1 Score</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['edge_f1']:.3f}</div>
                <div style="font-size: 0.72rem; color: #fcd34d;">Boundary Precision</div>
            </div>
            <div style="background: linear-gradient(135deg, rgba(2, 132, 199, 0.95), rgba(7, 89, 133, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 8px 16px -4px rgba(2, 132, 199, 0.3);">
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #bae6fd; font-weight: 600;">SSIM Fidelity</div>
                <div style="font-size: 1.55rem; font-weight: 800; color: #ffffff; margin: 4px 0;">{m_h['ssim']:.4f}</div>
                <div style="font-size: 0.72rem; color: #7dd3fc;">Structural Match</div>
            </div>
        </div>
        """
    else:
        metrics_df = pd.DataFrame([
            {"Method": "Reconstructed", "RMSE (m)": "N/A (Custom Input)", "MAE (m)": "N/A", "PSNR (dB)": "N/A", "SSIM": "N/A", "Boundary RMSE": "N/A", "Edge F1": "N/A", "Latency": f"{lat_h:.1f} ms"}
        ])
        kpi_html = """
        <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 12px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">✓ Custom Depth Reconstructed Successfully</div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Ground truth evaluation is only available on paired benchmark scenes.</div>
        </div>
        """

    # 5. Export 3D Assets (.OBJ, .PLY, .NPY)
    obj_path = "outputs/reconstructed_surface.obj"
    ply_path = "outputs/reconstructed_pointcloud.ply"
    npy_path = "outputs/reconstructed_depth.npy"

    export_depth_to_mesh_obj(d_h, rgb, output_path=obj_path, target_dim=80)
    export_depth_to_ply(d_h, rgb, output_path=ply_path, max_points=35000)
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
        return np.zeros((200, 400, 3), dtype=np.uint8)
    return plot_scanline_profile(
        LAST_RUN["bicubic"],
        LAST_RUN["classical"],
        LAST_RUN["deep"],
        LAST_RUN["hybrid"],
        gt=LAST_RUN["gt"],
        row_idx=row_idx,
        save_path="outputs/scanline_plot.png",
    )


# Custom CSS Theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    background-color: #030712 !important;
}

.hero-banner {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 65%, #020617 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.25);
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 80%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
    max-width: 860px;
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.hero-chip {
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    padding: 3px 10px;
    border-radius: 6px;
}

.btn-reconstruct {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.btn-reconstruct:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.6) !important;
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
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                HACKATHON 2026 · PROBLEM STATEMENT 4 (ProbID - P4)
            </div>
            <div class="hero-title">Edge-Preserving Depth Map Super-Resolution & Refinement</div>
            <div class="hero-desc">
                An intelligent hybrid depth reconstruction platform fusing <b>Joint Bilateral Upsampling</b> with <b>EdgeGuidedSRNet Neural Guidance</b>
                and <b>Backward Measurement Consistency</b> to restore crisp physical depth discontinuities without texture-bleeding artifacts.
            </div>
            <div class="hero-chips">
                <span class="hero-chip">⚡ PyTorch 2.14 / CUDA</span>
                <span class="hero-chip">🎯 Optimal Fixed α = 0.50</span>
                <span class="hero-chip">📐 Gradient Edge Discontinuity Gating</span>
                <span class="hero-chip">🧊 3D WebGL Mesh & Point Cloud (.OBJ / .PLY)</span>
                <span class="hero-chip">📊 Official Benchmark Testset Integrated</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔮 Interactive Super-Resolution Studio"):
            with gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=4):
                    gr.Markdown("### 🎛️ Input Data & Controls")

                    preset_choice = gr.Dropdown(
                        label="Select Scene or Official Testset Sample",
                        choices=TESTSET_PRESETS,
                        value=TESTSET_PRESETS[0],
                    )

                    input_rgb = gr.Image(
                        label="RGB Guidance Image Preview",
                        type="numpy",
                        height=210,
                    )

                    input_depth_file = gr.File(
                        label="Optional: Custom Depth Map (.npy or .png)",
                        file_types=[".npy", ".png", ".jpg"],
                    )

                    with gr.Row():
                        scale = gr.Radio(
                            label="Scale Factor",
                            choices=[2, 4, 8],
                            value=4,
                        )
                        colormap_choice = gr.Dropdown(
                            label="Colormap Palette",
                            choices=["turbo", "plasma", "viridis", "inferno", "magma", "bone", "gray"],
                            value="turbo",
                        )

                    with gr.Accordion("⚙️ Sensor Simulation & Fusion Hyperparameters", open=False):
                        noise_std = gr.Slider(
                            minimum=0.0, maximum=0.05, step=0.005, value=0.0,
                            label="Simulated Sensor Gaussian Noise (meters)"
                        )
                        hole_ratio_pct = gr.Slider(
                            minimum=0.0, maximum=25.0, step=1.0, value=0.0,
                            label="Missing Depth Measurement Holes (%)"
                        )
                        alpha = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.50,
                            label="Classical Blending Weight α (Calibrated Optimal = 0.50)"
                        )
                        use_consistency = gr.Checkbox(
                            value=True,
                            label="Enforce Measurement Consistency: r_LR = D_LR - S(D_f)"
                        )
                        scanline_row = gr.Slider(
                            minimum=10, maximum=500, step=5, value=250,
                            label="Scanline Row for 1D Edge Profile Inspection (y)"
                        )

                    run_btn = gr.Button("🚀 Reconstruct High-Resolution Depth", elem_classes=["btn-reconstruct"], size="lg")

                # Right Column: Reconstructed Results
                with gr.Column(scale=6):
                    gr.Markdown("### 📈 Reconstructed Output & Performance")
                    kpi_display = gr.HTML()

                    image_slider = gr.ImageSlider(
                        label="Swipe to Compare: Degraded LR Sensor Input (Left) vs. Reconstructed Hybrid Depth (Right)",
                        show_label=True,
                        height=420,
                    )

                    with gr.Row():
                        lr_output = gr.Image(label="Degraded LR Sensor Input (Nearest Grid)", type="numpy")
                        edge_output = gr.Image(label="Recovered Depth Boundaries (Edge Map)", type="numpy")

                    gr.Markdown("### 📋 Multi-Track Performance Scorecard")
                    metrics_table = gr.Dataframe(
                        headers=["Method", "RMSE (m)", "MAE (m)", "PSNR (dB)", "SSIM", "Boundary RMSE", "Edge F1", "Latency"],
                    )

        with gr.TabItem("📐 Edge & Scanline Discontinuity Inspector"):
            gr.Markdown("### 🔬 1D Depth Discontinuity Profile (Step-Response Analysis)")
            gr.Markdown("Inspect how sharp depth steps are reconstructed. Hybrid fusion and JBU eliminate the slope-blurring artifact typical of Bicubic interpolation.")

            with gr.Row():
                scanline_slider = gr.Slider(
                    minimum=10, maximum=500, step=5, value=250,
                    label="Active Inspection Row (Y-coordinate)",
                )
                refresh_scanline_btn = gr.Button("🔄 Refresh Scanline", size="sm")

            scanline_output = gr.Image(label="1D Depth Step Response Profile", type="numpy")

        with gr.TabItem("🧊 3D Digital Twin & Point Cloud Studio"):
            gr.Markdown("### 🌐 Interactive 3D WebGL Surface Mesh Viewer")
            gr.Markdown("Orbit, rotate, and zoom into the 3D surface mesh to inspect sharp object boundaries in physical 3D space.")

            with gr.Row():
                with gr.Column(scale=6):
                    mesh_3d_viewer = gr.Model3D(label="3D Surface Mesh (WebGL)", height=450)
                with gr.Column(scale=4):
                    gr.Markdown("#### 📥 Download 3D Assets")
                    download_obj_btn = gr.File(label="Download 3D Mesh (.OBJ)")
                    download_ply_btn = gr.File(label="Download Colored Point Cloud (.PLY)")
                    download_npy_btn = gr.File(label="Download Raw HR Depth Array (.NPY)")

        with gr.TabItem("📊 Multi-Scale Benchmark & Ablation Lab"):
            gr.Markdown("### 🧪 Full Multi-Track Progression & Visual Breakdown")
            gr.Markdown("Side-by-side progression across RGB Guidance, Degraded Input, Bicubic, Classical, Deep, Hybrid, Ground Truth, and Residual Heatmap.")
            showcase_panel = gr.Image(label="Full 8-Panel Multi-Track Comparison Breakdown", type="numpy")

    # Wire Preset Dropdown Change Event
    preset_choice.change(
        fn=on_preset_change,
        inputs=[preset_choice, scale],
        outputs=[input_rgb],
    )

    # Wire Main Execution Event
    run_btn.click(
        fn=run_super_resolution,
        inputs=[
            preset_choice,
            input_rgb,
            input_depth_file,
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
            showcase_panel,
            kpi_display,
            metrics_table,
            mesh_3d_viewer,
            download_obj_btn,
            download_ply_btn,
            download_npy_btn,
        ],
    )

    # Wire Scanline Slider Change
    scanline_slider.change(
        fn=update_scanline_only,
        inputs=[scanline_slider],
        outputs=[scanline_output],
    )
    refresh_scanline_btn.click(
        fn=update_scanline_only,
        inputs=[scanline_slider],
        outputs=[scanline_output],
    )

    # Auto-load initial sample on startup
    demo.load(
        fn=lambda: on_preset_change(TESTSET_PRESETS[0], 4),
        inputs=[],
        outputs=[input_rgb],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=theme,
    )
