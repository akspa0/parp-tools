#!/usr/bin/env pwsh
# setup-libs.ps1 — Bootstrap external library dependencies for building from source.
# Run this once after cloning the repo. Re-run to update libraries.
#
# Usage:
#   ./setup-libs.ps1          # Clone all libraries
#   ./setup-libs.ps1 -Force   # Remove and re-clone all libraries

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$libDir = Join-Path $PSScriptRoot "lib"

function Clone-Lib {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Target,
        [switch]$InitSubmodules,
        [string[]]$SubmoduleFilter
    )

    $fullPath = Join-Path $libDir $Target

    if ($Force -and (Test-Path $fullPath)) {
        Write-Host "  Removing existing $Target..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $fullPath
    }

    if ((Test-Path $fullPath) -and (Get-ChildItem $fullPath -Force | Measure-Object).Count -gt 0) {
        Write-Host "  $Name already exists at $Target — skipping (use -Force to re-clone)" -ForegroundColor DarkGray
        return
    }
    # Remove empty placeholder directories (e.g. from uninitialized submodules)
    if ((Test-Path $fullPath) -and (Get-ChildItem $fullPath -Force | Measure-Object).Count -eq 0) {
        Remove-Item -Recurse -Force $fullPath
    }

    Write-Host "  Cloning $Name..." -ForegroundColor Cyan
    git clone --depth 1 $Url $fullPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone $Name" }

    if ($InitSubmodules) {
        Push-Location $fullPath
        try {
            if ($SubmoduleFilter) {
                foreach ($sub in $SubmoduleFilter) {
                    Write-Host "    Init submodule: $sub" -ForegroundColor DarkCyan
                    git submodule update --init --depth 1 $sub
                }
            } else {
                git submodule update --init --depth 1
            }
            if ($LASTEXITCODE -ne 0) { throw "Failed to init submodules for $Name" }
        } finally {
            Pop-Location
        }
    }
}

Write-Host ""
Write-Host "=== parp-tools: Library Bootstrap ===" -ForegroundColor Green
Write-Host "Target: $libDir"
Write-Host ""

if (!(Test-Path $libDir)) {
    New-Item -ItemType Directory -Path $libDir | Out-Null
}

# --- SereniaBLPLib (BLP texture loading) ---
Clone-Lib -Name "SereniaBLPLib" `
    -Url "https://github.com/WoW-Tools/SereniaBLPLib.git" `
    -Target "SereniaBLPLib"

# --- WoWDBDefs (DBC definition files for DBCD) ---
Clone-Lib -Name "WoWDBDefs" `
    -Url "https://github.com/wowdev/WoWDBDefs.git" `
    -Target "WoWDBDefs"

# --- wow.tools.local (DBCD library + CascLib) ---
Clone-Lib -Name "wow.tools.local" `
    -Url "https://github.com/Marlamin/wow.tools.local.git" `
    -Target "wow.tools.local" `
    -InitSubmodules `
    -SubmoduleFilter @("DBCD", "CascLib")

# --- Warcraft.NET (WoW file format library) ---
Clone-Lib -Name "Warcraft.NET" `
    -Url "https://github.com/ModernWoWTools/Warcraft.NET.git" `
    -Target "Warcraft.NET"

# --- WoWTools.Minimaps (StormLibWrapper) ---
# Skip submodules — SereniaBLPLib cloned separately above, TACT.Net cloned below
Clone-Lib -Name "WoWTools.Minimaps" `
    -Url "https://github.com/Marlamin/WoWTools.Minimaps.git" `
    -Target "WoWTools.Minimaps"

# --- TACT.Net (required by StormLibWrapper) ---
# StormLibWrapper references ../TACT.Net/TACT.Net/TACT.Net.csproj
Clone-Lib -Name "TACT.Net" `
    -Url "https://github.com/wowdev/TACT.Net.git" `
    -Target "WoWTools.Minimaps/TACT.Net"

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "You can now build MdxViewer:"
Write-Host "  cd src/MdxViewer"
Write-Host "  dotnet build"
Write-Host ""
