"""Spec 120 Minimap OBB Object Detector Model (T005).

Small from-scratch PyTorch network for Oriented Bounding Box (OBB) object detection on 256x256 minimap tiles.
Predicts continuous center offset (dx, dy), scale (w, h), rotation angle (sin_theta, cos_theta),
confidence, and coarse class logits on a 16x16 grid.

Constructable from `base` channels alone (Rule 7: small modular specialist).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from harvester.v50.model_stage_contract import sha256_json


class ConvBlock(nn.Module):
    """Standard Convolution + BatchNorm + LeakyReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MinimapOBBDetector(nn.Module):
    """Oriented Bounding Box (OBB) Detector Network for 256x256 Minimap Tiles.

    Grid: 16x16 cells over the 256x256 tile (cell size = 16x16 pixels).
    Per-grid predictions (9 channels):
    - [0]: Confidence (sigmoid)
    - [1, 2]: Sub-cell center offset dx, dy in [0, 1] (sigmoid)
    - [3, 4]: Box width, height w, h in [0, 1] (exp/softplus)
    - [5, 6]: Angle orientation sin(theta), cos(theta)
    - [7:9]: Class logits (2 classes for 0.5.3: wmo vs mdx)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base = base
        self.grid_size = 16  # 16x16 grid for 256x256 input

        b = base
        # Encoder: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16
        self.enc1 = ConvBlock(in_channels, b, stride=2)       # 128x128
        self.enc2 = ConvBlock(b, b * 2, stride=2)             # 64x64
        self.enc3 = ConvBlock(b * 2, b * 4, stride=2)         # 32x32
        self.enc4 = ConvBlock(b * 4, b * 8, stride=2)         # 16x16

        # Neck & Head
        self.neck = ConvBlock(b * 8, b * 8)
        out_pred_channels = 1 + 2 + 2 + 2 + num_classes  # 11 channels
        self.head = nn.Conv2d(b * 8, out_pred_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:  (N, 3, 256, 256)
        Output: (N, 16, 16, 11) raw predictions tensor
        """
        feat = self.enc1(x)
        feat = self.enc2(feat)
        feat = self.enc3(feat)
        feat = self.enc4(feat)
        feat = self.neck(feat)

        out = self.head(feat)  # (N, 11, 16, 16)
        out = out.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 11)
        return out

    def decode_predictions(
        self, raw_pred: torch.Tensor, conf_thresh: float = 0.25
    ) -> list[list[dict[str, Any]]]:
        """Decode raw network output tensor into list of OBB detection dicts per image."""
        batch_size = raw_pred.shape[0]
        results: list[list[dict[str, Any]]] = []

        device = raw_pred.device
        raw_pred_cpu = raw_pred.detach().cpu()

        for b_idx in range(batch_size):
            img_preds: list[dict[str, Any]] = []
            grid = raw_pred_cpu[b_idx]  # (16, 16, 11)

            conf_map = torch.sigmoid(grid[..., 0])
            dx_map = torch.sigmoid(grid[..., 1])
            dy_map = torch.sigmoid(grid[..., 2])
            w_map = F.softplus(grid[..., 3])
            h_map = F.softplus(grid[..., 4])
            sin_map = grid[..., 5]
            cos_map = grid[..., 6]
            class_logits = grid[..., 7:]

            mask = conf_map >= conf_thresh
            grid_y, grid_x = torch.where(mask)

            for gy, gx in zip(grid_y.numpy(), grid_x.numpy()):
                conf = float(conf_map[gy, gx])
                dx = float(dx_map[gy, gx])
                dy = float(dy_map[gy, gx])
                w_norm = float(w_map[gy, gx])
                h_norm = float(h_map[gy, gx])

                s_val = float(sin_map[gy, gx])
                c_val = float(cos_map[gy, gx])
                angle_rad = math.atan2(s_val, c_val)
                angle_deg = math.degrees(angle_rad) % 360.0

                logits = class_logits[gy, gx]
                class_id = int(torch.argmax(logits))

                # Compute normalized pixel center
                cx_norm = (gx + dx) / float(self.grid_size)
                cy_norm = (gy + dy) / float(self.grid_size)

                px = cx_norm * 256.0
                py = cy_norm * 256.0
                w_px = w_norm * 256.0
                h_px = h_norm * 256.0

                img_preds.append({
                    "conf": conf,
                    "class_id": class_id,
                    "cx_norm": cx_norm,
                    "cy_norm": cy_norm,
                    "px": px,
                    "py": py,
                    "w_px": w_px,
                    "h_px": h_px,
                    "w_norm": w_norm,
                    "h_norm": h_norm,
                    "angle_deg": angle_deg,
                })

            results.append(img_preds)

        return results

    def model_config(self) -> dict[str, Any]:
        """Return dict representing model config for provenance."""
        return {
            "arch": "minimap_obb_detector_v1",
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base": self.base,
            "grid_size": self.grid_size,
        }

    def config_sha256(self) -> str:
        """SHA256 hex string of model config."""
        return sha256_json(self.model_config())
