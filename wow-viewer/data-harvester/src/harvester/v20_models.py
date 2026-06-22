"""V20 Multi-Modal Chained Models.

Defines:
1. V20SemanticSegmentor (Model 1): Multi-head segmentation (liquid, object, alpha)
2. V20FingerprintClassifier (Model 2): Terrain brush classifier + regressor
3. V20TerrainInpainter (Model 3): Context-guided height inpainter
4. V20PlacementRestorer (Model 4): Crop-based asset placement resolver
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from harvester.v19_models import ResConvBlock, BilinearUp


# ----------------------------------------------------------------------
# Model 1: Minimap Semantic Segmentor (V20-MSS)
# ----------------------------------------------------------------------
class V20SemanticSegmentor(nn.Module):
    """Multi-head semantic segmentor.
    - Input: Minimap RGB [3, 256, 256]
    - Head 1 (Liquid type): logits for [none, ocean, river, magma, slime] (5ch)
    - Head 2 (Object footprint): presence mask [1ch, Sigmoid]
    - Head 3 (Alpha blending): 4 texture layer weights [4ch, Sigmoid]
    """

    def __init__(self, in_channels: int = 3, norm_type: str = "group", groupnorm_groups: int = 8):
        super().__init__()
        # Encoder (256 -> 128 -> 64)
        self.enc0 = ResConvBlock(in_channels, 24, norm_type, groupnorm_groups)
        self.enc1 = ResConvBlock(24, 48, norm_type, groupnorm_groups)
        self.enc2 = ResConvBlock(48, 96, norm_type, groupnorm_groups)

        # Bottleneck (32x32)
        self.bottleneck = ResConvBlock(96, 96, norm_type, groupnorm_groups)

        # Decoder (32 -> 64 -> 128 -> 256)
        self.up2 = BilinearUp(96, 48)
        self.dec2 = ResConvBlock(48 + 96, 48, norm_type, groupnorm_groups)
        self.up1 = BilinearUp(48, 24)
        self.dec1 = ResConvBlock(24 + 48, 24, norm_type, groupnorm_groups)
        self.up0 = BilinearUp(24, 12)
        self.dec0 = ResConvBlock(12 + 24, 24, norm_type, groupnorm_groups)

        # Multi-head output decoders
        self.liquid_head = nn.Conv2d(24, 5, kernel_size=1)
        self.object_head = nn.Conv2d(24, 1, kernel_size=1)
        self.alpha_head = nn.Conv2d(24, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        enc0 = self.enc0(x)
        enc1 = self.enc1(F.max_pool2d(enc0, 2))
        enc2 = self.enc2(F.max_pool2d(enc1, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        # Decoder
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        dec0 = self.up0(dec1)
        dec0 = torch.cat([dec0, enc0], dim=1)
        dec0 = self.dec0(dec0)

        # Output heads
        liquid_logits = self.liquid_head(dec0)
        object_mask = torch.sigmoid(self.object_head(dec0))
        alpha_weights = torch.sigmoid(self.alpha_head(dec0))

        return liquid_logits, object_mask, alpha_weights


# ----------------------------------------------------------------------
# Model 2: Terrain Fingerprint Classifier (V20-TFC)
# ----------------------------------------------------------------------
class V20FingerprintClassifier(nn.Module):
    """Classifies terrain patches into a 3D terrain brush library.
    - Input: Minimap RGB (3ch) or + predicted alpha maps (4ch) = 7 channels
    - Outputs: Class probabilities (num_classes) + regression offsets (4 values: dx, dy, scale, rotation)
    """

    def __init__(self, in_channels: int = 7, num_classes: int = 200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.classifier = nn.Linear(256, num_classes)
        self.regressor = nn.Linear(256, 4)  # [dx, dy, scale, rot]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x).view(x.size(0), -1)
        cls_logits = self.classifier(features)
        reg_params = self.regressor(features)
        return cls_logits, reg_params


# ----------------------------------------------------------------------
# Model 3: Terrain Intent Inpainter (V20-TII)
# ----------------------------------------------------------------------
class V20TerrainInpainter(nn.Module):
    """Predicts ground intent heightmap beneath buildings and water.
    - Input: [10, 256, 256] consisting of:
        - Minimap RGB (3ch)
        - Predicted object footprint (1ch)
        - Predicted liquid type map (5ch)
        - Predicted terrain brush height prior (1ch)
    - Output: Reconstructed clean ground intent heightmap [1, 257, 257]
    """

    def __init__(self, in_channels: int = 10, norm_type: str = "group", groupnorm_groups: int = 8):
        super().__init__()
        # Encoder (256 -> 128 -> 64)
        self.enc0 = ResConvBlock(in_channels, 24, norm_type, groupnorm_groups)
        self.enc1 = ResConvBlock(24, 48, norm_type, groupnorm_groups)
        self.enc2 = ResConvBlock(48, 96, norm_type, groupnorm_groups)

        # Bottleneck (32x32)
        self.bottleneck = ResConvBlock(96, 96, norm_type, groupnorm_groups)

        # Decoder (32 -> 64 -> 128 -> 256)
        self.up2 = BilinearUp(96, 48)
        self.dec2 = ResConvBlock(48 + 96, 48, norm_type, groupnorm_groups)
        self.up1 = BilinearUp(48, 24)
        self.dec1 = ResConvBlock(24 + 48, 24, norm_type, groupnorm_groups)
        self.dec0 = ResConvBlock(24, 24, norm_type, groupnorm_groups)

        self.out_conv = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc0 = self.enc0(x)
        enc1 = self.enc1(F.max_pool2d(enc0, 2))
        enc2 = self.enc2(F.max_pool2d(enc1, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        dec0 = self.dec0(dec1)
        height_256 = self.out_conv(dec0)

        # Upsample to 257x257 height grid
        height_257 = F.interpolate(height_256, size=(257, 257), mode="bilinear", align_corners=False)
        return torch.clamp(height_257, -10.0, 10.0)  # clamp normalized height bounds safely


# ----------------------------------------------------------------------
# Model 4: Object Placement Restorer (V20-OPR)
# ----------------------------------------------------------------------
class V20PlacementRestorer(nn.Module):
    """Predicts asset layout placement parameters from object crop patches.
    - Input: RGB crop + object mask crop [4, 64, 64]
    - Output: WMO/M2 ID logits (classification) + 3D position, scale, rotation (5 regression values)
    """

    def __init__(self, in_channels: int = 4, num_models: int = 500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.classifier = nn.Linear(128, num_models)
        self.regressor = nn.Linear(128, 5)  # [dx, dy, dz_offset, scale, rotation_yaw]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x).view(x.size(0), -1)
        model_logits = self.classifier(features)
        reg_params = self.regressor(features)
        return model_logits, reg_params
