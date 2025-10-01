#!/usr/bin/env pwsh
# Rebuild WoWRollback with coordinate extraction and regenerate viewer data

[CmdletBinding(PositionalBinding=$false)]
param(
    [string[]]$Maps = @("DeadminesInstance"),
    [string[]]$Versions = @("0.5.3.3368","0.5.5.3494"),
    [string]$AlphaRoot,
    [switch]$Serve,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

# Ensure all relative paths resolve from this script's directory (WoWRollback/)
Set-Location -Path $PSScriptRoot

Write-Host "=== WoWRollback Rebuild & Regenerate ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean build
Write-Host "[1/3] Building solution..." -ForegroundColor Yellow
dotnet build WoWRollback.sln --configuration Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# Helper: Discover map names from AlphaRoot when -Maps auto
function Get-MapsFromAlphaRoot([string]$root) {
    if (-not $root -or -not (Test-Path $root)) { return @() }
    $results = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
    $tryPaths = @(
        Join-Path $root 'tree\World\Maps',
        Join-Path $root 'tree\world\maps',
        Join-Path $root 'World\Maps',
        Join-Path $root 'world\maps'
    )
    foreach ($p in $tryPaths) {
        if (Test-Path $p) {
            Get-ChildItem -Path $p -Recurse -File -Filter '*.wdt' -ErrorAction SilentlyContinue |
                ForEach-Object { [void]$results.Add($_.Directory.Name) }
        }
    }
    # Also consider pre-numbered minimaps folder structure
    $miniPaths = @(
        Join-Path $root 'tree\World\Minimaps',
        Join-Path $root 'tree\world\minimaps',
        Join-Path $root 'World\Minimaps',
        Join-Path $root 'world\minimaps'
    )
    foreach ($mp in $miniPaths) {
        if (Test-Path $mp) {
            Get-ChildItem -Path $mp -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { [void]$results.Add($_.Name) }
        }
    }
    # Ensure commonly useful maps are included if present
    foreach ($m in @('PVPZone01','Shadowfang')) { [void]$results.Add($m) }
    return $results.ToArray() | Sort-Object
}

# Step 2: Regenerate comparison data
Write-Host "[2/3] Regenerating comparison data (this may take several minutes)..." -ForegroundColor Yellow
$versionsArg = ($Versions -join ',')

# Auto-discover maps if requested
if ($Maps.Count -eq 1 -and $Maps[0].ToLower() -eq 'auto') {
    if (-not $AlphaRoot) {
        Write-Host "-Maps auto requires -AlphaRoot" -ForegroundColor Red
        exit 1
    }
    $discovered = Get-MapsFromAlphaRoot -root $AlphaRoot
    if (-not $discovered -or $discovered.Count -eq 0) {
        Write-Host "No maps discovered under $AlphaRoot" -ForegroundColor Yellow
        $Maps = @('Azeroth','Kalimdor','Kalidar','PVPZone01','Shadowfang')
    } else {
        $Maps = $discovered
    }
}

$mapsArg = ($Maps -join ',')
Write-Host ("Versions: {0}" -f $versionsArg) -ForegroundColor Gray
Write-Host ("Maps: {0}" -f $mapsArg) -ForegroundColor Gray

 # Build dotnet run command with optional alpha-root and any extra pass-through args
 $cmd = @(
    'run','--project','WoWRollback.Cli','--configuration','Release','--',
    'compare-versions',
    '--versions', $versionsArg,
    '--maps', $mapsArg,
    '--viewer-report'
 )
 if ($AlphaRoot) { $cmd += @('--alpha-root', $AlphaRoot) }
 if ($ExtraArgs) { $cmd += $ExtraArgs }
 # Capture dotnet output so we can parse the comparison directory path
 $dotnetOut = & dotnet @cmd 2>&1
 $dotnetOut | ForEach-Object { Write-Host $_ }

if ($LASTEXITCODE -ne 0) {
    Write-Host "Comparison generation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Data regenerated" -ForegroundColor Green
Write-Host ""

# Step 3: Locate the exact comparison directory from CLI output; fallback to latest with index.json
Write-Host "[3/3] Locating viewer output..." -ForegroundColor Yellow

$viewerPath = $null
$match = $dotnetOut | Select-String -Pattern 'Outputs written to:\s*(.+)$' | Select-Object -Last 1
if ($match) {
    $candidate = $match.Matches[0].Groups[1].Value.Trim()
    if (Test-Path $candidate) { $viewerPath = $candidate }
}

if (-not $viewerPath) {
    $latestWithIndex = Get-ChildItem -Path "rollback_outputs\comparisons" -Directory | `
        Sort-Object LastWriteTime -Descending | `
        Where-Object { Test-Path (Join-Path $_.FullName 'viewer\index.json') } | `
        Select-Object -First 1
    if ($latestWithIndex) { $viewerPath = $latestWithIndex.FullName }
}

if (-not $viewerPath) {
    Write-Host "Could not find comparison output!" -ForegroundColor Red
    exit 1
}

$viewerDir = Join-Path $viewerPath "viewer"
Write-Host "Viewer at: $viewerDir" -ForegroundColor Gray

# Ensure viewer directory exists and contains the static assets
if (-not (Test-Path $viewerDir)) {
    New-Item -ItemType Directory -Path $viewerDir | Out-Null
}

# Copy ViewerAssets (index.html, tile.html, js, styles) into viewer output
$assetsSrc = Join-Path $PSScriptRoot 'ViewerAssets'
if (Test-Path $assetsSrc) {
    Copy-Item -Path (Join-Path $assetsSrc '*') -Destination $viewerDir -Recurse -Force
} else {
    Write-Host "Warning: ViewerAssets directory not found at $assetsSrc" -ForegroundColor Yellow
}

# Check if viewer has data
$indexJson = Join-Path $viewerDir "index.json"
if (-not (Test-Path $indexJson)) {
    Write-Host "index.json not found!" -ForegroundColor Red
    exit 1
}


$overlayDir = Join-Path $viewerDir "overlays"
$overlayCount = (Get-ChildItem -Path $overlayDir -Recurse -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "✓ Viewer generated with $overlayCount overlay files" -ForegroundColor Green
Write-Host ""

# Step 4: Launch viewer
Write-Host "=== Ready to View ===" -ForegroundColor Cyan
Write-Host "" 
Write-Host "To start the viewer:" -ForegroundColor White
Write-Host "  cd \"$viewerDir\"" -ForegroundColor Yellow
Write-Host "  python -m http.server 8080" -ForegroundColor Yellow
Write-Host "" 
Write-Host "Then open: http://localhost:8080/index.html" -ForegroundColor Green
Write-Host "" 

if ($Serve) {
    Write-Host "Starting server on http://localhost:8080..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host "" 
    Set-Location $viewerDir
    python -m http.server 8080
} else {
    # Offer to start server interactively
    $response = Read-Host "Start Python HTTP server now? (Y/n)"
    if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
        Write-Host "Starting server on http://localhost:8080..." -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
        Write-Host "" 
        Set-Location $viewerDir
        python -m http.server 8080
    }
}
