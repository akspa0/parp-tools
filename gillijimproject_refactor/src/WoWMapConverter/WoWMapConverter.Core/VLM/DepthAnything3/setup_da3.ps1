# DepthAnything3 Setup Script for VLM Dataset Tool (Windows PowerShell)
# Requires: python 3.10+ installed and in PATH

$VENV_DIR = Join-Path $PSScriptRoot ".venv"

Write-Host "=== DepthAnything3 Setup for VLM Dataset Tool (venv) ===" -ForegroundColor Cyan
Write-Host ""

# Check for python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: python not found. Please install Python 3.10+ and add it to PATH." -ForegroundColor Red
    exit 1
}

# Create venv if not exists
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment in $VENV_DIR..."
    python -m venv $VENV_DIR
} else {
    Write-Host "Virtual environment already exists in $VENV_DIR"
}

# Activate venv for this session
$env:VIRTUAL_ENV = $VENV_DIR
$env:Path = "$VENV_DIR\Scripts;$env:Path"

# Install PyTorch (CPU version is fine for small batches, but CUDA preferred if available)
# Using generic torch install which will pick up CUDA if available or CPU otherwise
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
Write-Host "Environment created at: $VENV_DIR"
