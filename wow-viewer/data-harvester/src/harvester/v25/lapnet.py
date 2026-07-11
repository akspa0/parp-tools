import torch
import torch.nn as nn
import torch.nn.functional as F
from harvester.v25.solver import BatchedSylvesterSolver

class ProgressiveFeatureExtractor(nn.Module):
    """Predicts Sylvester solver regularizers (gamma_c, gamma_r) from clean terrain visual features."""
    def __init__(self, in_channels=3, feat_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Output log-gammas to ensure positive regularizer weights (gamma = exp(log_gamma))
        self.fc = nn.Linear(feat_channels, 2)
        
    def forward(self, x):
        # x is clean terrain RGB downsampled to current scale
        feats = self.conv(x).view(x.shape[0], -1)
        log_gammas = self.fc(feats) # (B, 2)
        # Enforce scale boundaries (e.g. clamp log gammas to prevent numerical issues)
        log_gammas = torch.clamp(log_gammas, min=-5.0, max=5.0)
        gammas = torch.exp(log_gammas)
        return gammas[:, 0].view(-1, 1, 1), gammas[:, 1].view(-1, 1, 1)

class V25StageBPredictor(nn.Module):
    """Progressive Sylvester solver upscaling heights from 33x33 to 257x257 edge-aligned meshes."""
    def __init__(self, device="cpu"):
        super().__init__()
        # Pre-initialize Sylvester solvers for each scale step
        self.solver_65 = BatchedSylvesterSolver(65, 65, device=device)
        self.solver_129 = BatchedSylvesterSolver(129, 129, device=device)
        self.solver_257 = BatchedSylvesterSolver(257, 257, device=device)
        
        # Lightweight regularizer extractors for each stage
        self.extractor_65 = ProgressiveFeatureExtractor(in_channels=3, feat_channels=16)
        self.extractor_129 = ProgressiveFeatureExtractor(in_channels=3, feat_channels=16)
        self.extractor_257 = ProgressiveFeatureExtractor(in_channels=3, feat_channels=16)
        
    def to(self, device):
        """Move all solver structures to the target device."""
        self.solver_65.to(device)
        self.solver_129.to(device)
        self.solver_257.to(device)
        return super().to(device)
        
    def forward(self, h_33, clean_terrain_map):
        """Upscales h_33 (B, 33, 33) progressively to (B, 257, 257).
        
        Args:
            h_33: Coarse terrain heights, shape (B, 33, 33)
            clean_terrain_map: Clean terrain-shadow map, shape (B, 3, 256, 256)
            
        Returns:
            h_257: Sharp terrain heights, shape (B, 257, 257)
        """
        B = h_33.shape[0]
        device = h_33.device
        
        # --- Stage 1: 33x33 -> 65x65 ---
        h_65_init = F.interpolate(h_33.unsqueeze(1), size=(65, 65), mode='bilinear', align_corners=True).squeeze(1)
        # Extract visual guidance features at scale 65x65
        clean_65 = F.interpolate(clean_terrain_map, size=(65, 65), mode='bilinear', align_corners=False)
        gamma_c_65, gamma_r_65 = self.extractor_65(clean_65)
        # Solve Sylvester equation
        h_65 = self.solver_65.solve(h_65_init, gamma_c_65, gamma_r_65)
        
        # --- Stage 2: 65x65 -> 129x129 ---
        h_129_init = F.interpolate(h_65.unsqueeze(1), size=(129, 129), mode='bilinear', align_corners=True).squeeze(1)
        clean_129 = F.interpolate(clean_terrain_map, size=(129, 129), mode='bilinear', align_corners=False)
        gamma_c_129, gamma_r_129 = self.extractor_129(clean_129)
        h_129 = self.solver_129.solve(h_129_init, gamma_c_129, gamma_r_129)
        
        # --- Stage 3: 129x129 -> 257x257 ---
        h_257_init = F.interpolate(h_129.unsqueeze(1), size=(257, 257), mode='bilinear', align_corners=True).squeeze(1)
        clean_257 = F.interpolate(clean_terrain_map, size=(257, 257), mode='bilinear', align_corners=False)
        gamma_c_257, gamma_r_257 = self.extractor_257(clean_257)
        h_257 = self.solver_257.solve(h_257_init, gamma_c_257, gamma_r_257)
        
        return h_257
