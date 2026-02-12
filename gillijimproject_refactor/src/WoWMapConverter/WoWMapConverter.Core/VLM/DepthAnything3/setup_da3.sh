#!/bin/bash
# DepthAnything3 Setup Script for VLM Dataset Tool
# Requires: conda or miniconda

ENV_NAME="da3"

echo "=== DepthAnything3 Setup for VLM Dataset Tool ==="
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch>=2 torchvision xformers addict

# Clone and install DepthAnything3
echo "Cloning DepthAnything3..."
if [ ! -d "Depth-Anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
fi

cd Depth-Anything-3

echo "Installing DepthAnything3..."
pip install -e .

# Download model (DA3Mono-Large for monocular depth)
echo "Downloading DA3Mono-Large model..."
python -c "from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('depth-anything/DA3MONO-LARGE')"

echo ""
echo "=== Setup Complete ==="
echo "To use: conda activate $ENV_NAME"
echo "Model: DA3MONO-LARGE (monocular depth estimation)"
