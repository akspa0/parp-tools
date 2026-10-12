"""Minimap to WDL Elevation Neural Model & Multi-Corpus Dataset (Spec 268).

Consumes 2D minimap image data and synthesizes authentic 17x17 WDL-quality
macro-mesh elevation lattices with calibrated physical relief scale [50, 600+] yards
for tiles where NO authentic WDL exists (such as development_17_14).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zarr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class ResidualBlock(nn.Module):
    """Conv-GN-GELU Residual Block for spatial terrain encoding."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + res)


class MinimapToWdlNet(nn.Module):
    """Neural Model mapping 256x256 minimap imagery to authentic 17x17 WDL elevation lattices.

    Architecture:
      - Automatic Photometric Feature Derivation: Computes luminance and Sobel surface
        gradient normals (Nx, Ny) on the fly for pure RGB inputs (6-channel total).
      - 4-Stage Hierarchical Encoder with GroupNorm and GELU (256x256 -> 16x16).
      - Decoupled Physical Heads:
        1. Shape Head: Relative topographic relief shape (17x17) strictly normalized in [0, 1].
        2. Span Head: Physical vertical mountain relief span ΔZ in world yards (> 10 yds).
        3. Base Head: Reference base elevation Z_min in world yards.
      - Composed WDL Elevation: Z_pred = Z_min + ΔZ * Shape.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 6   # 192

        # Fixed Sobel gradient filters for photometric normal derivation
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Stage 1: 256x256 -> 128x128
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            ResidualBlock(c1),
        )

        # Stage 2: 128x128 -> 64x64
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.GELU(),
            ResidualBlock(c2),
        )

        # Stage 3: 64x64 -> 32x32
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
            ResidualBlock(c3),
        )

        # Stage 4: 32x32 -> 16x16 (Bottleneck with dilated receptive field)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c4),
            nn.GELU(),
            ResidualBlock(c4),
            nn.Conv2d(c4, c4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, c4),
            nn.GELU(),
            ResidualBlock(c4),
        )

        # Shape Head: maps 16x16 bottleneck features to 17x17 local normalized terrain shape in [0, 1]
        self.shape_head = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c3),
            nn.GELU(),
            nn.Conv2d(c3, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Shared Global Pooling for Scale/Base Heads
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Vertical Relief Span Head: predicts physical mountain height span ΔZ in yards
        self.span_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Base Elevation Head: predicts reference elevation Z_min in yards
        self.base_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _prepare_features(self, x: torch.Tensor) -> torch.Tensor:
        """Derive 6-channel feature tensor: RGB (3) + shadow/luma (1) + Sobel normals (2)."""
        if x.shape[1] == 6:
            return x

        rgb = x[:, 0:3]
        luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        if x.shape[1] == 4:
            shadow = x[:, 3:4]
        else:
            shadow = luma

        # Compute photometric gradient normals
        pad_luma = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(pad_luma, self.sobel_x)
        gy = F.conv2d(pad_luma, self.sobel_y)
        slant = 4.0
        nx = -gx * slant
        ny = -gy * slant
        norm = torch.sqrt(nx * nx + ny * ny + 1.0)
        normals = torch.cat([nx / norm, ny / norm], dim=1)

        return torch.cat([rgb, shadow, normals], dim=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, C, 256, 256) tensor. Accepts C=3 (RGB), C=4 (RGB+shadow), or C=6.

        Returns:
            pred_h17: (B, 17, 17) reconstructed WDL elevation in world yards
            pred_shape: (B, 17, 17) normalized surface shape in [0, 1]
            pred_span: (B,) predicted vertical relief span ΔZ in yards
            pred_base: (B,) predicted base elevation Z_min in yards
        """
        feats = self._prepare_features(x)

        f1 = self.inc(feats)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)

        # 1. Normalized 17x17 terrain surface shape
        shape_16 = self.shape_head(f4)
        pred_shape = F.interpolate(
            shape_16,
            size=(17, 17),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # (B, 17, 17) in [0, 1]

        # 2. Physical vertical relief span ΔZ in world yards (calibrated for mountain ranges)
        pooled = self.pool(f4)
        raw_span = self.span_head(pooled).squeeze(1)
        pred_span = F.softplus(raw_span) * 250.0 + 15.0  # Spans range realistically from 15 to 800+ yds

        # 3. Base elevation level in world yards
        raw_base = self.base_head(pooled).squeeze(1)
        pred_base = raw_base * 200.0  # Centers around 0 to 400 yds

        # 4. Composed WDL coarse elevation lattice in yards
        pred_h17 = pred_base.view(-1, 1, 1) + pred_span.view(-1, 1, 1) * pred_shape
        return pred_h17, pred_shape, pred_span, pred_base

    def predict_tile(
        self,
        rgb: np.ndarray,
        shadow: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """High-level single-tile inference helper for minimap-only tiles.

        Args:
            rgb: (256, 256, 3) uint8 or float [0, 1]
            shadow: Optional (256, 256) float [0, 1]
            device: torch device

        Returns:
            (17, 17) numpy float32 elevation lattice in yards
        """
        if device is None:
            device = next(self.parameters()).device

        rgb_arr = np.asarray(rgb, dtype=np.float32)
        if rgb_arr.max() > 1.0:
            rgb_arr /= 255.0
        rgb_arr = np.clip(rgb_arr, 0.0, 1.0)

        if shadow is not None:
            sh_arr = np.asarray(shadow, dtype=np.float32)
            if sh_arr.max() > 1.0:
                sh_arr /= 255.0
            sh_arr = np.clip(sh_arr, 0.0, 1.0)
            feat = np.concatenate([rgb_arr, sh_arr[..., None]], axis=-1)  # (256, 256, 4)
        else:
            feat = rgb_arr

        tensor = torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).float().to(device)
        self.eval()
        with torch.no_grad():
            pred_h17, _, _, _ = self.forward(tensor)

        return pred_h17.squeeze(0).cpu().numpy().astype(np.float32)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def minimap_wdl_composite_loss(
    pred_h17: torch.Tensor,
    pred_shape: torch.Tensor,
    pred_span: torch.Tensor,
    pred_base: torch.Tensor,
    gt_h17: torch.Tensor,
    alpha_shape: float = 2.0,
    alpha_grad: float = 1.0,
    beta_span: float = 0.1,
    gamma_base: float = 0.05,
    delta_point: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Decoupled loss supervising relative terrain shape, vertical span, and absolute elevation."""
    # Compute ground truth shape in [0, 1], span, and base
    gt_min = gt_h17.amin(dim=(1, 2))  # (B,)
    gt_max = gt_h17.amax(dim=(1, 2))  # (B,)
    gt_span = torch.clamp(gt_max - gt_min, min=1.0)  # (B,)
    gt_shape = (gt_h17 - gt_min.view(-1, 1, 1)) / gt_span.view(-1, 1, 1)  # (B, 17, 17) in [0, 1]

    # 1. Surface shape L1 loss (independent of vertical scale!)
    shape_l1 = F.l1_loss(pred_shape, gt_shape)

    # 2. Surface shape gradient / ridge consistency
    pred_s_gy = pred_shape[:, 1:, :] - pred_shape[:, :-1, :]
    gt_s_gy = gt_shape[:, 1:, :] - gt_shape[:, :-1, :]
    pred_s_gx = pred_shape[:, :, 1:] - pred_shape[:, :, :-1]
    gt_s_gx = gt_shape[:, :, 1:] - gt_shape[:, :, :-1]
    shape_grad = F.l1_loss(pred_s_gy, gt_s_gy) + F.l1_loss(pred_s_gx, gt_s_gx)

    # 3. Relief span loss (supervises physical mountain height in yards)
    span_loss = F.smooth_l1_loss(pred_span, gt_span, beta=10.0)

    # 4. Base elevation loss in yards
    base_loss = F.smooth_l1_loss(pred_base, gt_min, beta=10.0)

    # 5. Composite elevation point loss in yards
    point_loss = F.smooth_l1_loss(pred_h17, gt_h17, beta=5.0)

    total_loss = (
        alpha_shape * shape_l1
        + alpha_grad * shape_grad
        + beta_span * span_loss
        + gamma_base * base_loss
        + delta_point * point_loss
    )

    metrics = {
        "shape_l1": float(shape_l1.item()),
        "shape_grad": float(shape_grad.item()),
        "span_loss": float(span_loss.item()),
        "base_loss": float(base_loss.item()),
        "point_loss": float(point_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics


class MultiCorpusWdlDataset(Dataset):
    """PyTorch Dataset loading both authentic development map terrain and client Zarr stores."""

    def __init__(
        self,
        dev_corpus_path: Optional[Union[Path, str]] = None,
        zarr_store_paths: Optional[List[Union[Path, str]]] = None,
        indices: Optional[List[Tuple[str, int]]] = None,
        augment: bool = False,
    ):
        self.augment = augment
        self.samples: List[Dict[str, any]] = []

        # 1. Load authentic development continent ground truth (vital for alpine mountain spans!)
        if dev_corpus_path is not None:
            p = Path(dev_corpus_path)
            if p.is_file():
                dev_data = np.load(p, allow_pickle=True)
                raw_dev = list(dev_data["train"]) + list(dev_data["val"])
                for s in raw_dev:
                    rgb = s["rgb"].astype(np.float32)
                    if rgb.max() > 1.0:
                        rgb /= 255.0
                    z257 = s["z"].astype(np.float32)
                    h17 = z257[0::16, 0::16]  # Blizzard MARE 17x17 lattice
                    shadow = np.clip(1.0 - rgb[..., 0], 0.0, 1.0)
                    self.samples.append({
                        "source": "development",
                        "stem": s.get("stem", "dev"),
                        "rgb": rgb,
                        "shadow": shadow,
                        "h17": h17,
                    })

        # 2. Load pre-decoded Zarr client stores (Azeroth & Kalimdor)
        if zarr_store_paths is not None:
            for zp in zarr_store_paths:
                zp = Path(zp)
                if zp.exists():
                    z = zarr.open_group(str(zp), mode="r")
                    rgbs = np.asarray(z["minimap_rgb_authored"])
                    shadows = np.asarray(z["shadow_mask"])
                    h17s = np.asarray(z["wdl_outer_17"])
                    for row in range(rgbs.shape[0]):
                        rgb = rgbs[row].astype(np.float32) / 255.0
                        sh = shadows[row].astype(np.float32)
                        h = h17s[row].astype(np.float32)
                        self.samples.append({
                            "source": zp.name,
                            "stem": f"{zp.name}_{row}",
                            "rgb": rgb,
                            "shadow": sh,
                            "h17": h,
                        })

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        s = self.samples[idx]
        rgb = s["rgb"].copy()
        shadow = s["shadow"].copy()
        h17 = s["h17"].copy()

        if self.augment:
            if np.random.rand() > 0.5:
                rgb = np.fliplr(rgb).copy()
                shadow = np.fliplr(shadow).copy()
                h17 = np.fliplr(h17).copy()

            if np.random.rand() > 0.5:
                rgb = np.flipud(rgb).copy()
                shadow = np.flipud(shadow).copy()
                h17 = np.flipud(h17).copy()

            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.92, 1.08)
                bias = np.random.uniform(-0.05, 0.05)
                rgb = np.clip(rgb * scale + bias, 0.0, 1.0)

        # 4-channel input: RGB (3) + shadow (1) (Sobel normals derived inside model)
        feat = np.concatenate([rgb, shadow[..., None]], axis=-1)  # (256, 256, 4)

        return {
            "features": torch.from_numpy(feat).permute(2, 0, 1).float(),
            "h17": torch.from_numpy(h17).float(),
            "stem": s["stem"],
        }
