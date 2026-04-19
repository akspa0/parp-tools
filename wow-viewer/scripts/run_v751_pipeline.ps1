#!/usr/bin/env pwsh
# run_v751_pipeline.ps1 — Full V7.5.1 dataset → training pipeline.
#
# Steps
# -----
#   1. Mask generation   — generate_m2_masks.py builds per-tile M2 footprint masks
#                         (skips tiles that already have a mask unless -ForceRemask)
#   2. Training          — train_v7.py runs the V7.5.1 multichannel U-Net
#
# Requirements
# ------------
#   - .venv or .venv-train created by setup script (or pass -PythonExe)
#   - wow-viewer built: dotnet build wow-viewer/WowViewer.slnx -c Debug
#   - datasets/ populated by WoWMapConverter ml-corpus exports
#
# Usage examples
# --------------
#   .\scripts\run_v751_pipeline.ps1
#   .\scripts\run_v751_pipeline.ps1 -SkipMasks -ResumeFrom output\ml-training\v7_5_1\best_ckpt.pth
#   .\scripts\run_v751_pipeline.ps1 -BuildFilter 3_3_5_12340 -DryRun
#   .\scripts\run_v751_pipeline.ps1 -ForceRemask -AllowCpu

param(
    # --- Step control ---
    [switch]$SkipMasks,
    [switch]$ForceRemask,          # Re-generate masks even if they already exist

    # --- Training ---
    [string]$ResumeFrom = "",      # Path to checkpoint .pth to resume from
    [int]$NumEpochs = 0,           # 0 = use train_v7.py default
    [int]$BatchSize = 0,           # 0 = use train_v7.py default
    [string]$OutputDir = "",       # Training output dir (default: auto-timestamped)

    # --- Dataset ---
    [string[]]$SearchRoots = @("datasets"),   # Where to look for dataset roots
    [string]$BuildFilter = "",     # Restrict mask gen to one build, e.g. 3_3_5_12340
    [string]$ArchiveRootsFile = "", # JSON file mapping build_label → archive_root
    [string]$ArchiveRootFallback = "", # Fallback archive root for builds without explicit mapping

    # --- Python env ---
    [string]$PythonExe = "",       # Python executable. Auto-detected if empty.
    [string]$LegacyScriptsDir = "", # Path to legacy WoWMapConverter scripts dir. Auto-detected.

    # --- Safety ---
    [switch]$AllowCpu,             # Allow training without CUDA (slow)
    [switch]$DryRun,               # Print commands but do not run them
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
    Write-Err "No CUDA GPU detected. Training on CPU will be extremely slow."
    Write-Err "Add -AllowCpu to proceed anyway, or ensure CUDA drivers and nvidia-smi are on PATH."
    exit 1
} elseif (!$hasCuda) {
    Write-Warn "No CUDA GPU detected. Training on CPU (-AllowCpu set)."
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
    Write-Err "Python not found. Run scripts/setup_training_env.ps1 first, or pass -PythonExe."
    exit 1
}
Write-Ok "Python: $PythonExe"

# ---------------------------------------------------------------------------
# Step 1 — M2 mask generation
# ---------------------------------------------------------------------------
if (!$SkipMasks) {
    Write-Step "Step 1 — Generating M2 object masks …"

    $maskArgs = @(
        (Join-Path $DatasetScriptsDir "generate_m2_masks.py")
    )
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
# Step 2 — Training
# ---------------------------------------------------------------------------
Write-Step "Step 2 — V7.5.1 training …"

$trainScript = Join-Path $LegacyScriptsDir "train_v7.py"
$trainArgs = @($trainScript)

foreach ($sr in $SearchRoots) { $trainArgs += @("--search-root", $sr) }

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
    $autoOutputDir = Join-Path $ParpToolsRoot "output\ml-training\v7_5_1\run_$ts"
    $trainArgs += @("--output-dir", $autoOutputDir)
    Write-Host "  Output dir: $autoOutputDir" -ForegroundColor DarkGray
}

Invoke-Step -Exe $PythonExe -CommandArgs $trainArgs -Cwd $LegacyScriptsDir
Write-Ok "V7.5.1 training complete."

} finally {
    Pop-Location
}
