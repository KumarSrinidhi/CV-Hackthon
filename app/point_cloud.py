# Team Owner: Kumar
# Workstream: Application + 3D Visualization
# Module: app/point_cloud.py
# Description: 3D point cloud generation and PLY file export from reconstructed RGB-D pairs.

from typing import Optional
import numpy as np


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

    Args:
        depth: Reconstructed high-resolution depth map in meters (H, W).
        rgb: High-resolution RGB guidance image (H, W, 3).
        output_path: Path where the .ply file will be saved.
        focal_length: Camera focal length in pixels.
        max_points: Maximum number of 3D vertices to write (for fast rendering).

    Returns:
        output_path
    """
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = depth.shape[:2]

    c_x, c_y = w / 2.0, h / 2.0
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Mask valid depth values
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 15.0)
    if not np.any(valid):
        # Fallback dummy point
        with open(output_path, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
        return output_path

    z = depth[valid]
    x_3d = ((grid_x[valid] - c_x) * z) / focal_length
    y_3d = ((grid_y[valid] - c_y) * z) / focal_length
    colors = rgb[valid]

    num_points = len(z)
    if num_points > max_points:
        # Subsample uniformly
        indices = np.linspace(0, num_points - 1, max_points, dtype=int)
        x_3d = x_3d[indices]
        y_3d = y_3d[indices]
        z = z[indices]
        colors = colors[indices]
        num_points = max_points

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            f.write(
                f"{x_3d[i]:.4f} {-y_3d[i]:.4f} {z[i]:.4f} "
                f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
            )

    return output_path
