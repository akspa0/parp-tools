@echo off
REM DepthAnything3 Setup Script for VLM Dataset Tool
REM Run this from Anaconda Prompt or regular cmd with conda in PATH

echo === DepthAnything3 Setup for VLM Dataset Tool ===
echo.

REM Check if conda exists
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda not found in PATH
    echo Please run this from Anaconda Prompt or add conda to your PATH
    exit /b 1
)

REM Install missing dependency first (in base environment)
echo Installing missing dependency: addict
pip install addict

REM Install dependencies for DA3
echo Installing DepthAnything3 dependencies...
pip install "torch>=2" torchvision xformers addict

REM Clone DepthAnything3 if not present
if not exist "Depth-Anything-3" (
    echo Cloning DepthAnything3...
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
)

cd Depth-Anything-3

REM Install in editable mode
echo Installing DepthAnything3...
pip install -e .

REM Download model
echo Downloading DA3Mono-Large model...
python -c "from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('depth-anything/DA3MONO-LARGE')"

echo.
echo === Setup Complete ===
echo Model: DA3MONO-LARGE (monocular depth estimation)

cd ..
