import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

class ObjectPlacementHead(nn.Module):
    """Predicts 3D coordinates, asset classifications, rotations, and existence logs of objects."""
    def __init__(self, in_channels=256, num_classes=32, max_objects=32):
        super().__init__()
        self.max_objects = max_objects
        self.num_classes = num_classes
        
        # Spatial pooling to get a flat representation of the 8x8 feature map
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Decoders for each parameter
        self.class_dec = nn.Linear(512, max_objects * num_classes)
        self.coords_dec = nn.Linear(512, max_objects * 3)
        self.rotations_dec = nn.Linear(512, max_objects * 3)
        self.exist_dec = nn.Linear(512, max_objects * 1)
        
    def forward(self, x):
        # x shape: (B, in_channels, H, W)
        x_pooled = self.pool(x).view(x.shape[0], -1) # (B, in_channels)
        feat = self.fc(x_pooled) # (B, 512)
        
        B = x.shape[0]
        
        class_logits = self.class_dec(feat).view(B, self.max_objects, self.num_classes)
        coords = self.coords_dec(feat).view(B, self.max_objects, 3)
        rotations = self.rotations_dec(feat).view(B, self.max_objects, 3)
        exist_logits = self.exist_dec(feat).view(B, self.max_objects, 1)
        
        return {
            "class_logits": class_logits,
            "coords": coords,
            "rotations": rotations,
            "exist_logits": exist_logits
        }

class DoubleConv(nn.Module):
    """Small double convolution block with skip connectivity."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TerrainInpaintHead(nn.Module):
    """Lightweight gated U-Net to inpaint objects with clean terrain textures and shadows."""
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x_rgb, object_mask):
        # x_rgb shape: (B, 3, 256, 256)
        # object_mask shape: (B, 1, 256, 256)
        x = torch.cat([x_rgb, object_mask], dim=1) # (B, 4, 256, 256)
        
        # U-Net forward pass
        x1 = self.inc(x) # (B, 32, 256, 256)
        x2 = self.down1(x1) # (B, 64, 128, 128)
        x3 = self.down2(x2) # (B, 128, 64, 64)
        
        x_up = self.up1(x3) # (B, 64, 128, 128)
        x_up = torch.cat([x_up, x2], dim=1) # (B, 128, 128, 128)
        x_up = self.conv_up1(x_up) # (B, 64, 128, 128)
        
        x_up = self.up2(x_up) # (B, 32, 256, 256)
        x_up = torch.cat([x_up, x1], dim=1) # (B, 64, 256, 256)
        x_up = self.conv_up2(x_up) # (B, 32, 256, 256)
        
        out = self.outc(x_up) # (B, 3, 256, 256)
        # Output clean terrain: restrict to RGB range using sigmoid or clamp
        return torch.sigmoid(out)

class V25SegformerDecompiler(nn.Module):
    """Visual decompiler frontend using a lightweight SegFormer backbone to output masks and placements."""
    def __init__(self, num_classes=32, max_objects=32, model_name_or_path="nvidia/mit-b0"):
        super().__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        try:
            # Attempt to load pretrained model configuration offline-compatibly
            config = SegformerConfig.from_pretrained(model_name_or_path, local_files_only=True)
            self.encoder = SegformerModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
        except Exception:
            # Fallback to local configuration matching mit-b0 architecture
            config = SegformerConfig(
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2],
                hidden_sizes=[32, 64, 160, 256],
                decoder_hidden_size=256,
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
            )
            self.encoder = SegformerModel(config)
            
        # Segformer multi-scale feature down-projections (mit-b0 hidden sizes are 32, 64, 160, 256)
        self.dec_proj1 = nn.Conv2d(32, 256, kernel_size=1)
        self.dec_proj2 = nn.Conv2d(64, 256, kernel_size=1)
        self.dec_proj3 = nn.Conv2d(160, 256, kernel_size=1)
        self.dec_proj4 = nn.Conv2d(256, 256, kernel_size=1)
        
        # Fuses channels and outputs object semantic mask logits
        self.dec_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        self.placement_head = ObjectPlacementHead(
            in_channels=256,
            num_classes=num_classes,
            max_objects=max_objects
        )

        # FR-102-102: the inpaint head is part of the unified decompiler — it
        # consumes the raw RGB plus the predicted object mask and emits the
        # clean terrain-shadow map that guides the progressive height solver.
        self.inpaint_head = TerrainInpaintHead(in_channels=4, out_channels=3)

    def forward(self, x):
        # Input raw minimap: (B, 3, 256, 256)
        outputs = self.encoder(x, output_hidden_states=True)
        # outputs.hidden_states maps to:
        # 0: input embed (B, 32, 64, 64)
        # 1: stage 1 output (B, 32, 64, 64)
        # 2: stage 2 output (B, 64, 32, 32)
        # 3: stage 3 output (B, 160, 16, 16)
        # 4: stage 4 output (B, 256, 8, 8)
        stage_outputs = [
            outputs.hidden_states[0],
            outputs.hidden_states[1],
            outputs.hidden_states[2],
            outputs.hidden_states[3]
        ]
        
        # Projects multi-scale features to 256 channels and interpolates to 64x64 size
        p1 = self.dec_proj1(stage_outputs[0]) # (B, 256, 64, 64)
        p2 = F.interpolate(self.dec_proj2(stage_outputs[1]), size=(64, 64), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.dec_proj3(stage_outputs[2]), size=(64, 64), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.dec_proj4(stage_outputs[3]), size=(64, 64), mode='bilinear', align_corners=False)
        
        fused = torch.cat([p1, p2, p3, p4], dim=1) # (B, 1024, 64, 64)
        mask_logits_64 = self.dec_fusion(fused) # (B, 1, 64, 64)
        
        # Upsample mask to full resolution
        mask_logits = F.interpolate(mask_logits_64, size=(256, 256), mode='bilinear', align_corners=False) # (B, 1, 256, 256)
        
        # Extracted high-level features for placements
        final_feats = stage_outputs[3] # (B, 256, 8, 8)
        placements = self.placement_head(final_feats)

        # Clean terrain-shadow map from the raw RGB gated by the predicted mask
        clean_rgb = self.inpaint_head(x, torch.sigmoid(mask_logits))

        return {
            "mask_logits": mask_logits,
            "clean_rgb": clean_rgb,
            "placements": placements,
            "final_feats": final_feats,
        }
