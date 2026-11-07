# Team Owner: Kumar
# Workstream: Application + 3D Visualization
# Module: app/point_cloud.py
# Description: 3D point cloud and textured mesh generation (.PLY and .OBJ) for interactive web 3D inspection.

from typing import Optional, Tuple
import cv2
import numpy as np
import trimesh


def export_depth_to_ply(
    depth: np.ndarray,
    rgb: np.ndarray,
    output_path: str,
    focal_length: float = 500.0,
    max_points: int = 50000,
) -> str:
    """
    Project reconstructed 2D depth map and RGB image into a 3D colored point cloud
    and export as an industry-standard .PLY file.
    """
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = depth.shape[:2]

    c_x, c_y = w / 2.0, h / 2.0
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 15.0)
    if not np.any(valid):
        with open(output_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
        return output_path

    z = depth[valid]
    x_3d = ((grid_x[valid] - c_x) * z) / focal_length
    y_3d = ((grid_y[valid] - c_y) * z) / focal_length
    colors = rgb[valid]

    num_points = len(z)
    if num_points > max_points:
        indices = np.linspace(0, num_points - 1, max_points, dtype=int)
        x_3d = x_3d[indices]
        y_3d = y_3d[indices]
        z = z[indices]
        colors = colors[indices]
        num_points = max_points

    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            f.write(
                f"{x_3d[i]:.4f} {-y_3d[i]:.4f} {z[i]:.4f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )

    return output_path


def export_depth_to_mesh_obj(
    depth: np.ndarray,
    rgb: np.ndarray,
    output_path: str,
    focal_length: float = 350.0,
    target_dim: int = 70,
) -> str:
    """
    Generate an interactive 3D surface mesh with vertex colors for browser 3D rendering.
    Downsamples grid to target_dim for real-time 60 FPS WebGL rendering in Gradio.
    """
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    h_orig, w_orig = depth.shape[:2]

    # Resize to target dimension for lightweight interactive mesh
    scale_factor = target_dim / max(h_orig, w_orig)
    new_w = max(10, int(w_orig * scale_factor))
    new_h = max(10, int(h_orig * scale_factor))

    d_sub = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb_sub = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    c_x, c_y = new_w / 2.0, new_h / 2.0
    gx, gy = np.meshgrid(np.arange(new_w), np.arange(new_h))

    # Clamp depth
    d_clean = np.nan_to_num(d_sub, nan=2.0)
    d_clean = np.clip(d_clean, 0.2, 8.0)

    # 3D coordinates
    f = focal_length * scale_factor
    x_3d = ((gx - c_x) * d_clean) / f
    y_3d = ((gy - c_y) * d_clean) / f
    z_3d = d_clean

    vertices = np.stack([x_3d, -y_3d, -z_3d], axis=-1).reshape(-1, 3)

    # Colors with alpha
    colors = np.zeros((new_h * new_w, 4), dtype=np.uint8)
    colors[:, :3] = rgb_sub.reshape(-1, 3)
    colors[:, 3] = 255

    # Triangulate quad grid
    i, j = np.meshgrid(np.arange(new_h - 1), np.arange(new_w - 1), indexing="ij")
    v0 = (i * new_w + j).flatten()
    v1 = ((i + 1) * new_w + j).flatten()
    v2 = (i * new_w + j + 1).flatten()
    v3 = ((i + 1) * new_w + j + 1).flatten()

    faces_1 = np.stack([v0, v1, v2], axis=-1)
    faces_2 = np.stack([v2, v1, v3], axis=-1)
    faces = np.concatenate([faces_1, faces_2], axis=0)

    # Filter out faces that stretch across massive depth discontinuities
    z_flat = z_3d.flatten()
    face_z_diff = np.abs(z_flat[faces[:, 0]] - z_flat[faces[:, 1]]) + np.abs(z_flat[faces[:, 1]] - z_flat[faces[:, 2]])
    valid_faces = faces[face_z_diff < 0.6]

    mesh = trimesh.Trimesh(vertices=vertices, faces=valid_faces, vertex_colors=colors, process=False)
    mesh.export(output_path)
    return output_path
