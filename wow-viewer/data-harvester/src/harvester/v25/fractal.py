import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_noise_canvas(size=1024):
    """Generate a smooth, isotropic multi-frequency noise canvas of size size x size."""
    y = torch.linspace(0, 2 * math.pi * 20, size).view(-1, 1)
    x = torch.linspace(0, 2 * math.pi * 20, size).view(1, -1)
    
    canvas = torch.zeros(size, size)
    # Sum of sine waves at diverse orientations and frequencies
    angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for i, angle in enumerate(angles):
        freq = 1.5 ** (i // 2)
        rot_x = x * math.cos(angle) - y * math.sin(angle)
        canvas += (1.0 / freq) * torch.sin(rot_x * freq)
        
    # Normalize to [0, 1]
    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())
    return canvas.unsqueeze(0).unsqueeze(0) # (1, 1, size, size)

class DifferentiableFractalGenerator(nn.Module):
    """Generates continuous fractal noise on a 256x256 grid from predicted spatial coordinates."""
    def __init__(self, canvas_size=1024):
        super().__init__()
        # Pre-generate the static noise canvas and register it as a buffer
        canvas = generate_noise_canvas(canvas_size)
        self.register_buffer("noise_canvas", canvas)
        
    def forward(self, offsets, frequency, persistence, amplitude):
        """Samples from the noise canvas at 3 octaves.
        
        Args:
            offsets: Tensor of shape (B, 2) representing (x, y) seed translation offsets
            frequency: Tensor of shape (B, 1) representing noise frequency scale
            persistence: Tensor of shape (B, 1) representing octave weight decays
            amplitude: Tensor of shape (B, 1) representing final amplitude multipliers
            
        Returns:
            noise_map: Reconstructed noise maps of shape (B, 256, 256)
        """
        B = offsets.shape[0]
        device = offsets.device
        
        # Create standard normalized coordinates grid: shape (B, 256, 256, 2)
        # grid values are in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 256, device=device),
            torch.linspace(-1.0, 1.0, 256, device=device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # (B, 256, 256, 2)
        
        # We sample 3 octaves: f, 2f, 4f
        octaves = []
        weights = [1.0, persistence, persistence ** 2]
        
        for oct_idx in range(3):
            scale = (2.0 ** oct_idx) * frequency.view(B, 1, 1, 1)
            # Transform grid: scale and translate coordinates
            grid = base_grid * scale + offsets.view(B, 1, 1, 2)
            # Ensure periodic wrapping or reflection padding by taking mod or reflection
            # F.grid_sample expects input coordinates in [-1, 1], so we wrap using torch.sin or mod
            grid = torch.remainder(grid + 1.0, 2.0) - 1.0 # Wrap grid values to [-1, 1]
            
            # Sample from pre-generated noise canvas
            # self.noise_canvas shape: (1, 1, 1024, 1024)
            # F.grid_sample output shape: (B, 1, 256, 256)
            sampled = F.grid_sample(
                self.noise_canvas.repeat(B, 1, 1, 1),
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True
            ).squeeze(1) # (B, 256, 256)
            
            octaves.append(sampled)
            
        # Sum octaves
        noise_map = (
            octaves[0] * 1.0 +
            octaves[1] * weights[1].view(B, 1, 1) +
            octaves[2] * weights[2].view(B, 1, 1)
        )
        # Normalize/Scale by amplitude
        noise_map = noise_map * amplitude.view(B, 1, 1)
        return torch.clamp(noise_map, 0.0, 1.0)

class FractalParameterHead(nn.Module):
    """Predicts coarse boundaries and fractal seed parameters from visual features."""
    def __init__(self, in_channels=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Paint boundary mask decoders: outputs soft boundaries for each of the 4 blend layers
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # (B, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_layers, kernel_size=3, padding=1) # (B, num_layers, 32, 32)
        )
        
        # Spatial pooling to predict scalars
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Predict parameters for each of the 4 layers:
        # Each layer needs: 2 for offsets, 1 for frequency, 1 for persistence, 1 for amplitude = 5 params
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_layers * 5)
        )
        
    def forward(self, x):
        # x is Segformer feature maps: (B, 256, 8, 8)
        B = x.shape[0]
        
        # Predict paint boundaries
        boundaries_32 = self.mask_conv(x) # (B, num_layers, 32, 32)
        boundaries_256 = F.interpolate(boundaries_32, size=(256, 256), mode='bilinear', align_corners=False) # (B, num_layers, 256, 256)
        boundaries = torch.sigmoid(boundaries_256)
        
        # Predict scalar parameters
        pooled = self.pool(x).view(B, -1)
        params = self.fc(pooled).view(B, self.num_layers, 5)
        
        # Break down predicted parameters:
        # params[:, l, 0:2] = offsets (tanh to constrain translation boundaries)
        # params[:, l, 2] = frequency (sigmoid * max_freq)
        # params[:, l, 3] = persistence (sigmoid)
        # params[:, l, 4] = amplitude (sigmoid)
        offsets = torch.tanh(params[:, :, 0:2]) # (B, num_layers, 2)
        frequency = torch.sigmoid(params[:, :, 2]) * 5.0 + 0.1 # (B, num_layers)
        persistence = torch.sigmoid(params[:, :, 3]) # (B, num_layers)
        amplitude = torch.sigmoid(params[:, :, 4]) # (B, num_layers)
        
        return {
            "boundaries": boundaries,
            "offsets": offsets,
            "frequency": frequency,
            "persistence": persistence,
            "amplitude": amplitude
        }
