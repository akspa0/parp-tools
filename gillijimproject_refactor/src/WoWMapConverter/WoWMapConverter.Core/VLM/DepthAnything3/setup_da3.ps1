# DepthAnything3 Setup Script for VLM Dataset Tool (Windows PowerShell)
# Requires: conda or miniconda

$ENV_NAME = "da3"

Write-Host "=== DepthAnything3 Setup for VLM Dataset Tool ===" -ForegroundColor Cyan
Write-Host ""

# Check for conda
try {
    $condaPath = (Get-Command conda -ErrorAction Stop).Source
    Write-Host "Found conda: $condaPath" -ForegroundColor Green
} catch {
    Write-Host "Error: conda not found. Please install Miniconda or Anaconda first." -ForegroundColor Red
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

# Create conda environment
Write-Host "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
Write-Host "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch with CUDA
Write-Host "Installing PyTorch..."
pip install "torch>=2" torchvision xformers addict

# Clone and install DepthAnything3
Write-Host "Cloning DepthAnything3..."
if (-not (Test-Path "Depth-Anything-3")) {
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
}

Set-Location Depth-Anything-3

Write-Host "Installing DepthAnything3..."
pip install -e .

# Download model
Write-Host "Downloading DA3Mono-Large model..."
python -c "from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('depth-anything/DA3MONO-LARGE')"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "To use: conda activate $ENV_NAME"
Write-Host "Model: DA3MONO-LARGE (monocular depth estimation)"
