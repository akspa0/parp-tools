#!/usr/bin/env pwsh
# run_v76_pipeline.ps1 — Full V7.6 dataset → training pipeline.
#
# Steps
# -----
#   1. Mask generation   — generate_m2_masks.py (shared with v7.5.1, optional via -SkipMasks)
#   2. Cache build       — cache_v7_6_data.py pre-processes tiles into tensor cache
#   3. Training          — train_v7_6.py trains the dual-head ResNet34 U-Net
#
# Requirements
# ------------
#   - .venv or .venv-train created by setup script (or pass -PythonExe)
#   - datasets/ populated by WoWMapConverter ml-corpus exports
#
# Usage examples
# --------------
#   .\scripts\run_v76_pipeline.ps1
#   .\scripts\run_v76_pipeline.ps1 -SkipCache -ResumeFrom output\ml-training\v7_6\checkpoints\best.pth
#   .\scripts\run_v76_pipeline.ps1 -SkipMasks -CacheDir cached_v7_6 -DryRun

param(
    # --- Step control ---
    [switch]$SkipMasks,            # Skip M2 mask generation
    [switch]$SkipCache,            # Skip cache build (assume cached_v7_6/ is current)
    [switch]$ForceRemask,          # Re-generate masks even if they already exist

    # --- Cache ---
    [string]$CacheDir = "",        # Where to write/read cached tensors. Default: cached_v7_6/
    [string[]]$SearchRoots = @("datasets"),

    # --- Training ---
    [string]$ResumeFrom = "",      # Checkpoint .pth to resume from
    [int]$NumEpochs = 0,           # 0 = use train_v7_6.py default
    [int]$BatchSize = 0,           # 0 = use train_v7_6.py default
    [string]$OutputDir = "",       # Training output dir. Default: output/ml-training/v7_6/

    # --- Dataset ---
    [string]$BuildFilter = "",
    [string]$ArchiveRootsFile = "",
    [string]$ArchiveRootFallback = "",

    # --- Python env ---
    [string]$PythonExe = "",
    [string]$LegacyScriptsDir = "",

    # --- Safety ---
    [switch]$AllowCpu,
    [switch]$DryRun,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step([string]$msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)    { Write-Host "[OK]  $msg" -ForegroundColor Green }
function Write-Warn([string]$msg)  { Write-Warning $msg }
function Write-Err([string]$msg)   { Write-Host "[ERR] $msg" -ForegroundColor Red }

function Invoke-Step {
    param([string]$Exe, [string[]]$CommandArgs, [string]$Cwd = "")
    $display = "$Exe $($CommandArgs -join ' ')"
    Write-Host "  $ $display" -ForegroundColor DarkGray
    if ($DryRun) { return }
    if ($Cwd) { Push-Location $Cwd }
    try {
        & $Exe @CommandArgs
        if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $display" }
    } finally {
        if ($Cwd) { Pop-Location }
    }
}

function Find-Python {
    foreach ($candidate in @(".venv\Scripts\python.exe", ".venv\Scripts\python", ".venv-train\Scripts\python.exe", ".venv-train\Scripts\python", "python", "python3")) {
        $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($resolved) { return $resolved.Source }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
$ScriptSelf = $PSScriptRoot
$WowViewerRoot = (Get-Item $ScriptSelf).Parent.FullName
$ParpToolsRoot = (Get-Item $WowViewerRoot).Parent.FullName

if (!$LegacyScriptsDir) {
    $LegacyScriptsDir = Join-Path $ParpToolsRoot "gillijimproject_refactor\src\WoWMapConverter\scripts"
}

$DatasetScriptsDir = Join-Path $WowViewerRoot "scripts"

if (!$CacheDir) {
    $CacheDir = Join-Path $ParpToolsRoot "cached_v7_6"
}

Push-Location $ParpToolsRoot
try {

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
Write-Step "Checking GPU availability …"
$hasCuda = $false
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    $smiOut = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $smiOut) {
        $hasCuda = $true
        Write-Ok "CUDA GPU: $($smiOut -join ', ')"
    }
}
if (!$hasCuda -and !$AllowCpu) {
    Write-Err "No CUDA GPU detected. Add -AllowCpu to proceed on CPU."
    exit 1
} elseif (!$hasCuda) {
    Write-Warn "No CUDA GPU — training on CPU (-AllowCpu set)."
}

# ---------------------------------------------------------------------------
# Resolve Python
# ---------------------------------------------------------------------------
Write-Step "Resolving Python …"
if (!$PythonExe) {
    Push-Location $ParpToolsRoot
    $PythonExe = Find-Python
    Pop-Location
}
if (!$PythonExe) {
    Write-Err "Python not found. Run scripts/setup_training_env.ps1 first."
    exit 1
}
Write-Ok "Python: $PythonExe"

# ---------------------------------------------------------------------------
# Step 1 — M2 mask generation (shared with v7.5.1)
# ---------------------------------------------------------------------------
if (!$SkipMasks) {
    Write-Step "Step 1 — Generating M2 object masks …"

    $maskArgs = @((Join-Path $DatasetScriptsDir "generate_m2_masks.py"))
    foreach ($sr in $SearchRoots) { $maskArgs += @("--search-root", $sr) }
    if ($BuildFilter)       { $maskArgs += @("--build-filter", $BuildFilter) }
    if ($ArchiveRootsFile)  { $maskArgs += @("--archive-roots-file", $ArchiveRootsFile) }
    if ($ArchiveRootFallback) { $maskArgs += @("--archive-root-fallback", $ArchiveRootFallback) }
    if (!$ForceRemask)      { $maskArgs += @("--skip-existing", "true") } else { $maskArgs += @("--skip-existing", "false") }
    if ($DryRun)            { $maskArgs += @("--dry-run") }
    if ($Verbose)           { $maskArgs += @("--verbose") }

    Invoke-Step -Exe $PythonExe -CommandArgs $maskArgs
    Write-Ok "Mask generation complete."
} else {
    Write-Warn "Skipping mask generation (-SkipMasks)."
}

# ---------------------------------------------------------------------------
# Step 2 — Build tensor cache
# ---------------------------------------------------------------------------
if (!$SkipCache) {
    Write-Step "Step 2 — Building V7.6 tensor cache → $CacheDir …"

    $cacheArgs = @(
        (Join-Path $LegacyScriptsDir "cache_v7_6_data.py"),
        "--output-dir", $CacheDir
    )
    foreach ($sr in $SearchRoots) { $cacheArgs += @("--search-root", $sr) }

    Invoke-Step -Exe $PythonExe -CommandArgs $cacheArgs
    Write-Ok "Cache build complete."
} else {
    Write-Warn "Skipping cache build (-SkipCache). Using: $CacheDir"
    if (!$DryRun -and !(Test-Path $CacheDir)) {
        Write-Err "Cache dir not found: $CacheDir"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Step 3 — Training
# ---------------------------------------------------------------------------
Write-Step "Step 3 — V7.6 training …"

$trainArgs = @(
    (Join-Path $LegacyScriptsDir "train_v7_6.py"),
    "--cache-dir", $CacheDir
)

if ($ResumeFrom) {
    if (!(Test-Path $ResumeFrom)) {
        Write-Warn "Checkpoint not found: $ResumeFrom — starting fresh."
    } else {
        $trainArgs += @("--resume", $ResumeFrom)
    }
}

if ($NumEpochs -gt 0)  { $trainArgs += @("--epochs", $NumEpochs) }
if ($BatchSize -gt 0)  { $trainArgs += @("--batch-size", $BatchSize) }

if ($OutputDir) {
    $trainArgs += @("--output-dir", $OutputDir)
} else {
    $ts = (Get-Date -Format "yyyyMMdd_HHmmss")
    $autoOutputDir = Join-Path $ParpToolsRoot "output\ml-training\v7_6\run_$ts"
    $trainArgs += @("--output-dir", $autoOutputDir)
    Write-Host "  Output dir: $autoOutputDir" -ForegroundColor DarkGray
}

Invoke-Step -Exe $PythonExe -CommandArgs $trainArgs -Cwd $LegacyScriptsDir
Write-Ok "V7.6 training complete."

} finally {
    Pop-Location
}
