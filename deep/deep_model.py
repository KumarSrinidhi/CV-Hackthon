# Team Owner: Rakshitha
# Workstream: Deep Learning
# Module: deep/deep_model.py
# Description: Edge-preserving guided depth super-resolution neural network architecture (SGNet-inspired).

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Residual block with batch normalization and ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))


class EdgeGuidedSRNet(nn.Module):
    """
    Edge-preserving guided depth super-resolution network.
    Uses multi-scale high-resolution RGB structural guidance to guide
    low-resolution depth upsampling and reconstruct crisp object boundaries.
    """

    def __init__(self, num_feats: int = 32):
        super().__init__()
        # RGB Guidance Feature Extractor
        self.rgb_head = nn.Sequential(
            nn.Conv2d(3, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.rgb_res = ResBlock(num_feats)

        # Depth Feature Extractor (processes bicubic upsampled depth)
        self.depth_head = nn.Sequential(
            nn.Conv2d(1, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_res = ResBlock(num_feats)

        # Joint Guidance & Depth Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_feats * 2, num_feats * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(num_feats * 2),
            ResBlock(num_feats * 2),
            nn.Conv2d(num_feats * 2, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Edge-aware attention module
        self.edge_gate = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=1),
            nn.Sigmoid(),
        )

        # Residual reconstruction head
        self.recon = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, 1, kernel_size=3, padding=1),
        )

        # Initialize reconstruction weights close to zero for stable residual output
        nn.init.normal_(self.recon[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.recon[-1].bias, 0.0)

    def forward(self, init_hr_depth: torch.Tensor, hr_rgb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            init_hr_depth: (B, 1, H, W) initial upsampled depth in physical units.
            hr_rgb: (B, 3, H, W) normalized RGB guidance image in [0, 1].
        Returns:
            (B, 1, H, W) refined HR depth map.
        """
        f_rgb = self.rgb_res(self.rgb_head(hr_rgb))
        f_depth = self.depth_res(self.depth_head(init_hr_depth))

        f_cat = torch.cat([f_depth, f_rgb], dim=1)
        f_fused = self.fusion(f_cat)
        gate = self.edge_gate(f_fused)
        f_attended = f_fused * gate

        residual = self.recon(f_attended)
        out_depth = init_hr_depth + residual
        return out_depth


class DeepDepthSRModel:
    """
    Wrapper model manager managing weights, device allocation (CPU / CUDA),
    and evaluation mode for deep depth super-resolution.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = EdgeGuidedSRNet(num_feats=32).to(self.device)
        self.model.eval()

    def predict(
        self,
        lr_depth: np.ndarray,
        hr_rgb: np.ndarray,
        scale: int = 4,
    ) -> np.ndarray:
        """
        Run inference on a single numpy pair.
        """
        h_lr, w_lr = lr_depth.shape[:2]
        h_hr, w_hr = hr_rgb.shape[:2]

        # Convert RGB to torch float tensor in [0, 1]
        rgb_tensor = torch.from_numpy(hr_rgb).float()
        if rgb_tensor.ndim == 3:
            # (H, W, 3) -> (1, 3, H, W)
            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0

        # Convert LR depth to initial HR depth via bicubic interpolation
        lr_tensor = torch.from_numpy(lr_depth).float().unsqueeze(0).unsqueeze(0)
        init_hr_tensor = F.interpolate(
            lr_tensor, size=(h_hr, w_hr), mode="bicubic", align_corners=False
        )

        rgb_tensor = rgb_tensor.to(self.device)
        init_hr_tensor = init_hr_tensor.to(self.device)

        with torch.no_grad():
            out_tensor = self.model(init_hr_tensor, rgb_tensor)

        pred = out_tensor.squeeze().cpu().numpy().astype(np.float32)
        # Ensure finite
        if not np.all(np.isfinite(pred)):
            fallback = init_hr_tensor.squeeze().cpu().numpy().astype(np.float32)
            pred = np.where(np.isfinite(pred), pred, fallback)

        return pred
