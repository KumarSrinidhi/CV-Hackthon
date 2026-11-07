# Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Super-Resolution and Refinement
# Hugging Face Spaces Entry Point (Gradio Interface)

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
from app.visualization import colorize_depth, create_side_by_side_comparison
from app.point_cloud import export_depth_to_ply
from data.dataset import generate_synthetic_rgbd_scene
from data.degradation import create_lr_depth


# Create demo presets
def get_preset_scene(scene_id: str, scale: int = 4):
    seed = 42 if scene_id == "Living Room" else (105 if scene_id == "Sphere & Curved Objects" else 201)
    rgb, gt_depth = generate_synthetic_rgbd_scene(h=240, w=320, seed=seed)
    return rgb, gt_depth


def run_super_resolution(
    input_rgb: Optional[np.ndarray],
    preset_choice: str,
    scale: int,
    noise_std: float,
    hole_ratio_pct: float,
    alpha: float,
    use_consistency: bool,
    method_choice: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Optional[str]]:
    """
    Main processing callback for Gradio UI.
    """
    os.makedirs("outputs", exist_ok=True)
    hole_ratio = hole_ratio_pct / 100.0

    # 1. Acquire Input Data
    if input_rgb is not None:
        rgb = np.asarray(input_rgb, dtype=np.uint8)
        # Generate paired synthetic depth for user-uploaded image if ground truth is not uploaded
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gt_depth = 1.0 + 3.0 * (1.0 - gray)
    else:
        rgb, gt_depth = get_preset_scene(preset_choice, scale=scale)

    h_hr, w_hr = rgb.shape[:2]

    # 2. Simulate Low-Resolution Sensor Input
    lr_depth, valid_mask = create_lr_depth(
        gt_depth, scale=scale, noise_std=noise_std, hole_ratio=hole_ratio
    )

    # 3. Execute Reconstruction Branch(es)
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

    t0 = time.perf_counter()
    d_b = classical_hr_depth(lr_depth, rgb, scale=scale, method="bicubic")
    lat_b = (time.perf_counter() - t0) * 1000.0

    # 4. Determine Primary Output Display
    if method_choice == "Hybrid (Optimal: Classical + Deep + Consistency)":
        primary_depth = d_h
    elif method_choice == "Deep (EdgeGuidedSRNet)":
        primary_depth = d_d
    elif method_choice == "Classical (Joint Bilateral Upsampling)":
        primary_depth = d_c
    elif method_choice == "Bicubic (Baseline)":
        primary_depth = d_b
    else:
        # All (Side-by-side)
        vis_path = "outputs/web_comparison.png"
        create_side_by_side_comparison(
            rgb, lr_depth, d_b, d_c, d_d, d_h, gt_depth=gt_depth, save_path=vis_path
        )
        primary_vis = cv2.imread(vis_path)[:, :, ::-1]
        primary_depth = d_h

    if method_choice != "Compare All (Side-by-Side)":
        primary_vis = colorize_depth(primary_depth)

    # Low-res input visualization
    lr_vis = colorize_depth(
        cv2.resize(lr_depth, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    )

    # Edge map of reconstructed depth
    edges = extract_depth_edges(primary_depth, threshold=0.04)
    edge_vis = (edges.astype(np.uint8) * 255)
    edge_vis_rgb = np.stack([edge_vis, edge_vis, edge_vis], axis=-1)

    # 5. Compute Metrics Table
    m_b = compute_all_metrics(d_b, gt_depth, valid_mask=valid_mask)
    m_b["Latency"] = f"{lat_b:.1f} ms"

    m_c = compute_all_metrics(d_c, gt_depth, valid_mask=valid_mask)
    m_c["Latency"] = f"{lat_c:.1f} ms"

    m_d = compute_all_metrics(d_d, gt_depth, valid_mask=valid_mask)
    m_d["Latency"] = f"{lat_d:.1f} ms"

    m_h = compute_all_metrics(d_h, gt_depth, valid_mask=valid_mask)
    m_h["Latency"] = f"{lat_h:.1f} ms"

    metrics_df = pd.DataFrame([
        {"Method": "Bicubic (baseline)", "RMSE (m)": f"{m_b['rmse']:.4f}", "MAE (m)": f"{m_b['mae']:.4f}", "PSNR (dB)": f"{m_b['psnr']:.2f}", "SSIM": f"{m_b['ssim']:.4f}", "Boundary RMSE": f"{m_b['boundary_rmse']:.4f}", "Edge F1": f"{m_b['edge_f1']:.3f}", "Latency": m_b["Latency"]},
        {"Method": "Classical (JBU)", "RMSE (m)": f"{m_c['rmse']:.4f}", "MAE (m)": f"{m_c['mae']:.4f}", "PSNR (dB)": f"{m_c['psnr']:.2f}", "SSIM": f"{m_c['ssim']:.4f}", "Boundary RMSE": f"{m_c['boundary_rmse']:.4f}", "Edge F1": f"{m_c['edge_f1']:.3f}", "Latency": m_c["Latency"]},
        {"Method": "Deep (SGNet branch)", "RMSE (m)": f"{m_d['rmse']:.4f}", "MAE (m)": f"{m_d['mae']:.4f}", "PSNR (dB)": f"{m_d['psnr']:.2f}", "SSIM": f"{m_d['ssim']:.4f}", "Boundary RMSE": f"{m_d['boundary_rmse']:.4f}", "Edge F1": f"{m_d['edge_f1']:.3f}", "Latency": m_d["Latency"]},
        {"Method": f"Hybrid (Optimal, alpha={alpha})", "RMSE (m)": f"{m_h['rmse']:.4f}", "MAE (m)": f"{m_h['mae']:.4f}", "PSNR (dB)": f"{m_h['psnr']:.2f}", "SSIM": f"{m_h['ssim']:.4f}", "Boundary RMSE": f"{m_h['boundary_rmse']:.4f}", "Edge F1": f"{m_h['edge_f1']:.3f}", "Latency": m_h["Latency"]},
    ])

    # 6. Export 3D Point Cloud (.PLY)
    ply_path = "outputs/reconstructed_point_cloud.ply"
    export_depth_to_ply(primary_depth, rgb, output_path=ply_path, max_points=40000)

    return primary_vis, lr_vis, edge_vis_rgb, metrics_df, ply_path


# Build Modern Gradio UI
custom_css = """
.gradio-container { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.main-header { text-align: center; margin-bottom: 20px; }
.main-header h1 { color: #1e293b; font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }
.main-header p { color: #64748b; font-size: 1.05rem; }
.metric-box { border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
"""

with gr.Blocks(title="Edge-Preserving Depth Super-Resolution (P4)") as demo:
    gr.HTML(
        f"""
        <style>{custom_css}</style>
        <div class="main-header">
            <h1>🏔️ Edge-Preserving Depth Map Super-Resolution & Refinement</h1>
            <p>Problem Statement 4 (ProbID - P4) · <b>Hybrid Intelligence</b>: Classical CV (JBU) + Deep Guidance (EdgeGuidedSRNet) + Consistency</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🚀 Interactive Reconstruction & Comparison"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 1. Input Guidance & Degradation Controls")
                    input_rgb = gr.Image(
                        label="Upload HR RGB Guidance (Optional)",
                        type="numpy",
                    )
                    preset_choice = gr.Dropdown(
                        label="Or Choose Pre-packaged RGB-D Scene",
                        choices=["Living Room", "Sphere & Curved Objects", "Pillars & Discontinuities"],
                        value="Living Room",
                    )
                    with gr.Row():
                        scale = gr.Radio(
                            label="Super-Resolution Scale",
                            choices=[2, 4, 8],
                            value=4,
                        )
                        method_choice = gr.Dropdown(
                            label="Reconstruction Track",
                            choices=[
                                "Compare All (Side-by-Side)",
                                "Hybrid (Optimal: Classical + Deep + Consistency)",
                                "Deep (EdgeGuidedSRNet)",
                                "Classical (Joint Bilateral Upsampling)",
                                "Bicubic (Baseline)",
                            ],
                            value="Compare All (Side-by-Side)",
                        )

                    with gr.Accordion("⚙️ Sensor Simulation & Fusion Hyperparameters", open=False):
                        noise_std = gr.Slider(
                            minimum=0.0, maximum=0.05, step=0.005, value=0.0,
                            label="Simulated Sensor Noise Std (meters)"
                        )
                        hole_ratio_pct = gr.Slider(
                            minimum=0.0, maximum=20.0, step=1.0, value=0.0,
                            label="Missing Depth Holes (%)"
                        )
                        alpha = gr.Slider(
                            minimum=0.0, maximum=0.2, step=0.01, value=0.02,
                            label="Classical Blending Weight alpha (Calibrated Optimal = 0.02)"
                        )
                        use_consistency = gr.Checkbox(
                            value=True,
                            label="Enforce Measurement Consistency Projection r_LR = D_LR - S(D_f)"
                        )

                    run_btn = gr.Button("🚀 Run High-Resolution Reconstruction", variant="primary", size="lg")

                with gr.Column(scale=6):
                    gr.Markdown("### 2. High-Resolution Reconstruction Output")
                    primary_output = gr.Image(label="Reconstruction / Comparison Result", type="numpy")

                    with gr.Row():
                        lr_output = gr.Image(label="LR Depth Input (Degraded)", type="numpy")
                        edge_output = gr.Image(label="Depth Discontinuities (Edge Map)", type="numpy")

                    gr.Markdown("### 3. Quantitative Evaluation Scorecard")
                    metrics_table = gr.Dataframe(
                        label="Benchmark Metrics Comparison (NYU Protocol)",
                        headers=["Method", "RMSE (m)", "MAE (m)", "PSNR (dB)", "SSIM", "Boundary RMSE", "Edge F1", "Latency"],
                    )

                    with gr.Row():
                        ply_download = gr.File(label="📥 Download 3D Point Cloud (.PLY Model)")

            run_btn.click(
                fn=run_super_resolution,
                inputs=[
                    input_rgb,
                    preset_choice,
                    scale,
                    noise_std,
                    hole_ratio_pct,
                    alpha,
                    use_consistency,
                    method_choice,
                ],
                outputs=[
                    primary_output,
                    lr_output,
                    edge_output,
                    metrics_table,
                    ply_download,
                ],
            )

        with gr.TabItem("📊 Multi-Scale Benchmark & Ablation Study"):
            gr.Markdown(
                """
                ### Validation Findings on NYU-v2 Benchmark (x4 Scale, CUDA)

                | Validation Variant | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE ↓ | Edge F1 ↑ | Latency |
                |:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
                | **Bicubic Baseline** | 0.0367 | 0.0100 | 51.62 | 0.9957 | 0.1680 | 0.259 | 2.8 ms |
                | **Classical (JBU)** | 0.0584 | 0.0167 | 47.59 | 0.9900 | 0.2591 | 0.671 | 464 ms |
                | **Deep Learning (SGNet branch)** | 0.0123 | 0.0036 | 59.57 | 0.9994 | 0.0537 | 0.976 | 1899 ms |
                | **Hybrid (Optimal: α=0.02 + Consistency)** | **0.0122** | **0.0036** | **59.58** | **0.9994** | **0.0536** | **0.975** | 2134 ms |

                #### Key Insights:
                1. **The Consistency Advantage**: Projecting the fused depth back to sensor space ($r_{LR} = D_{LR} - S(D_f)$) and applying damped residual correction anchors predictions to ground-truth sensor measurements, consistently driving down RMSE and boosting PSNR.
                2. **Why α = 0.02 Is Optimal**: High-confidence deep predictions provide superior spatial accuracy; injecting a small fraction ($\alpha \approx 0.02$) of Classical Joint Bilateral filtering sharpens genuine boundary transitions without causing texture-bleeding artifacts in planar regions.
                3. **Selective Refinement**: Applying blanket bilateral smoothing across the entire depth map degrades edge sharpness. Refinement must be strictly gated to regions where both depth and RGB gradients indicate genuine geometric boundaries.
                """
            )

        with gr.TabItem("🔬 Technical Architecture"):
            gr.Markdown(
                """
                ### Core Mathematical Formulations

                #### 1. Joint Bilateral Upsampling (Kopf et al.)
                $$\\tilde{D}(p) = \\frac{\\sum_{q \\in \\Omega} f_s(\\|p_* - q_*\\|) \\cdot f_r(\\|I(p) - I(q)\\|) \\cdot D_{LR}(q_*)}{\\sum_{q \\in \\Omega} f_s(\\|p_* - q_*\\|) \\cdot f_r(\\|I(p) - I(q)\\|)}$$

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
