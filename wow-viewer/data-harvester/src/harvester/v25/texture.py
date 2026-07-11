import torch
import torch.nn as nn
import torch.nn.functional as F

class MtexPredictor(nn.Module):
    """Predicts a multi-hot index selection probability over the global texture vocabulary."""
    def __init__(self, in_channels=256, vocab_size=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, vocab_size)
        )
        
    def forward(self, x):
        # x shape: (B, 256, 8, 8)
        B = x.shape[0]
        feats = self.pool(x).view(B, -1)
        logits = self.fc(feats) # (B, vocab_size)
        return logits

class MclyDecoder(nn.Module):
    """Predicts active texture assignments (0-3) over the 16x16 ADT chunk layer grid."""
    def __init__(self, in_channels=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Upsampler: maps (B, 256, 8, 8) to (B, 128, 16, 16)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Outputs logits for 4 classes (one of the 4 MTEX selections) for each of the 4 layers
            nn.Conv2d(64, num_layers * 4, kernel_size=1) # (B, 16, 16, 16)
        )
        
    def forward(self, x):
        # x shape: (B, 256, 8, 8)
        B = x.shape[0]
        logits = self.conv(x) # (B, 16, 16, 16)
        # Reshape to (B, 4, 16, 16, 4) where:
        # Dim 1: 4 classes (which texture selection index)
        # Dim 4: 4 layers
        logits = logits.view(B, 4, 16, 16, self.num_layers)
        return logits
