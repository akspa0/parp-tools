#!/usr/bin/env pwsh
# setup_training_env.ps1 - Create a dedicated uv-managed training venv for train_v7.py.
#
# Usage examples:
#   ./scripts/setup_training_env.ps1 -Backend auto -Recreate
#   ./scripts/setup_training_env.ps1 -Backend cuda -PythonVersion 3.11
#   ./scripts/setup_training_env.ps1 -Backend cpu -VenvPath .venv-train-cpu

param(
    [ValidateSet("auto", "cuda", "cpu", "rocm", "mps")]
    [string]$Backend = "auto",
    [string]$PythonVersion = "3.11",
    [string]$VenvPath = ".venv-train",
    [switch]$Recreate,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$message) { Write-Host "[INFO] $message" -ForegroundColor Cyan }
function Write-Ok([string]$message) { Write-Host "[OK]   $message" -ForegroundColor Green }
function Write-Warn([string]$message) { Write-Warning $message }

function Invoke-Step([string]$Executable, [string[]]$Arguments) {
    $joined = if ($Arguments.Count -gt 0) { "$Executable $($Arguments -join ' ')" } else { $Executable }
    if ($DryRun) {
        Write-Host "[DRYRUN] $joined"
        return
    }

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $joined"
    }
}

function Resolve-TrainingBackend([string]$requested) {
    if ($requested -ne "auto") {
        return $requested
    }

    if (($IsWindows -or $IsLinux) -and (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
        return "cuda"
    }

    if ($IsLinux -and (Get-Command "rocminfo" -ErrorAction SilentlyContinue)) {
        return "rocm"
    }

    if ($IsMacOS) {
        return "mps"
    }

    return "cpu"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$requirementsPath = Join-Path $PSScriptRoot "requirements_train_v7.txt"
if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Missing requirements file: $requirementsPath"
}

if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $venvFullPath = $VenvPath
} else {
    $venvFullPath = Join-Path $repoRoot $VenvPath
}

$resolvedBackend = Resolve-TrainingBackend $Backend
$venvPython = if ($IsWindows) { Join-Path $venvFullPath "Scripts\python.exe" } else { Join-Path $venvFullPath "bin/python" }
$requiresWindowsTriton = $resolvedBackend -eq "cuda" -and $IsWindows
$requiresWindowsTritonPy = if ($requiresWindowsTriton) { "True" } else { "False" }

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    throw "uv is required but not found on PATH. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
}

Write-Info "Repo root: $repoRoot"
Write-Info "Requested backend: $Backend"
Write-Info "Resolved backend: $resolvedBackend"
Write-Info "Python version: $PythonVersion"
Write-Info "Training venv: $venvFullPath"

if ($Recreate -and (Test-Path -LiteralPath $venvFullPath)) {
    Write-Info "Recreating existing venv: $venvFullPath"
    if (-not $DryRun) {
        Remove-Item -LiteralPath $venvFullPath -Recurse -Force
    }
}

Invoke-Step "uv" @("python", "install", $PythonVersion)
Invoke-Step "uv" @("venv", $venvFullPath, "--python", $PythonVersion)

# Install non-torch dependencies first.
Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "-r", $requirementsPath)

switch ($resolvedBackend) {
    "cuda" {
        Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "--index-url", "https://download.pytorch.org/whl/cu128", "torch", "torchvision", "torchaudio")
        if ($requiresWindowsTriton) {
            Write-Info "Installing triton-windows for torch.compile support on Windows CUDA."
            Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "triton-windows")
        }
    }
    "rocm" {
        Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "--index-url", "https://download.pytorch.org/whl/rocm6.2.4", "torch", "torchvision", "torchaudio")
    }
    "cpu" {
        Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "--index-url", "https://download.pytorch.org/whl/cpu", "torch", "torchvision", "torchaudio")
    }
    "mps" {
        # macOS MPS support ships in regular PyPI torch wheels.
        Invoke-Step "uv" @("pip", "install", "--python", $venvPython, "torch", "torchvision", "torchaudio")
    }
    default {
        throw "Unsupported backend: $resolvedBackend"
    }
}

if ($DryRun) {
    Write-Info "Dry run complete. Skipped runtime validation."
} else {
    $validationScript = @"
import sys
import importlib.util
import torch

backend = "${resolvedBackend}"
requires_windows_triton = ${requiresWindowsTritonPy}
triton_available = importlib.util.find_spec("triton") is not None
print(f"PYTHON={sys.executable}")
print(f"TORCH={torch.__version__}")
print(f"TORCH_CUDA={torch.version.cuda}")
print(f"TORCH_HIP={getattr(torch.version, 'hip', None)}")
print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
print(f"MPS_AVAILABLE={getattr(torch.backends, 'mps', None).is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"TRITON_AVAILABLE={triton_available}")

if backend == "cuda" and not torch.cuda.is_available():
    raise SystemExit("Expected CUDA backend, but torch.cuda.is_available() is False.")
if requires_windows_triton and not triton_available:
    raise SystemExit("Expected Windows CUDA backend to include triton-windows, but the 'triton' module is unavailable.")
if backend == "rocm" and not bool(getattr(torch.version, "hip", None)):
    raise SystemExit("Expected ROCm backend, but torch.version.hip is not available.")
if backend == "mps":
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not mps_ok:
        raise SystemExit("Expected MPS backend, but torch.backends.mps.is_available() is False.")
"@
    Invoke-Step $venvPython @("-c", $validationScript)
}

$trainScriptPath = Join-Path $repoRoot "src\WoWMapConverter\scripts\train_v7.py"
Write-Ok "Training environment is ready."
Write-Host ""
Write-Host "Run training with:"
Write-Host "  $venvPython $trainScriptPath --profile development-map"
Write-Host ""
Write-Host "If you intentionally need CPU fallback for a debug run, add: --allow-cpu"
