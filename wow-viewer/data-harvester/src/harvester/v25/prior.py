import torch
import torch.nn as nn
import torch.nn.functional as F

class V25StageAPredictor(nn.Module):
    """Predicts coarse 33x33 height priors from SegFormer final feature maps."""
    def __init__(self, in_channels=256):
        super().__init__()
        # input feats: (B, 256, 8, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1) # (B, 1, 32, 32)
        )
        
    def forward(self, final_feats):
        """Map final feature maps to a 33x33 quincunx height grid.
        
        Args:
            final_feats: Encoder feature maps, shape (B, 256, 8, 8)
            
        Returns:
            h_33: Coarse heights, shape (B, 33, 33)
        """
        h_32 = self.conv(final_feats).squeeze(1) # (B, 32, 32)
        # Pad bounds to match (33, 33) coordinates (16 chunks of 2x2 blocks = 32, plus border edge)
        h_33 = F.pad(h_32, (0, 1, 0, 1), mode='replicate')
        return h_33

class WdlDownsampler(nn.Module):
    """Downsamples high-resolution terrain meshes (257x257) to low-resolution WDL priors (33x33)."""
    def __init__(self):
        super().__init__()
        
    def forward(self, h_257):
        """Downsamples high-res meshes using node stride mapping.
        
        Args:
            h_257: Detailed height map, shape (B, 257, 257) or (B, 1, 257, 257)
            
        Returns:
            wdl_prior: Downsampled height prior, shape (B, 33, 33) or (B, 1, 33, 33)
        """
        if h_257.dim() == 4:
            return h_257[:, :, ::8, ::8]
        elif h_257.dim() == 3:
            return h_257[:, ::8, ::8]
        else:
            raise ValueError(f"Expected 3D or 4D height tensor, got shape: {h_257.shape}")
