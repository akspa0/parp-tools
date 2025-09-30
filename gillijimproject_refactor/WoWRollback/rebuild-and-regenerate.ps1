#!/usr/bin/env pwsh
# Rebuild WoWRollback with coordinate extraction and regenerate viewer data

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

# Step 2: Regenerate comparison data
Write-Host "[2/3] Regenerating comparison data (this may take several minutes)..." -ForegroundColor Yellow
Write-Host "Versions: 0.5.3.3368, 0.5.5.3494" -ForegroundColor Gray
Write-Host "Maps: Kalimdor, Azeroth" -ForegroundColor Gray

dotnet run --project WoWRollback.Cli --configuration Release -- `
    compare-versions `
    --versions 0.5.3.3368,0.5.5.3494 `
    --maps Kalimdor,Azeroth `
    --viewer-report

if ($LASTEXITCODE -ne 0) {
    Write-Host "Comparison generation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Data regenerated" -ForegroundColor Green
Write-Host ""

# Step 3: Find latest comparison output
Write-Host "[3/3] Locating viewer output..." -ForegroundColor Yellow
$viewerPath = Get-ChildItem -Path "rollback_outputs\comparisons" -Directory -Filter "*0_5_3_3368_vs_0_5_5_3494*" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 -ExpandProperty FullName

if (-not $viewerPath) {
    Write-Host "Could not find comparison output!" -ForegroundColor Red
    exit 1
}

$viewerDir = Join-Path $viewerPath "viewer"
Write-Host "Viewer at: $viewerDir" -ForegroundColor Gray

# Check if viewer has data
$indexJson = Join-Path $viewerDir "index.json"
if (-not (Test-Path $indexJson)) {
    Write-Host "index.json not found!" -ForegroundColor Red
    exit 1
}

$overlayDir = Join-Path $viewerDir "overlays"
$overlayCount = (Get-ChildItem -Path $overlayDir -Recurse -Filter "*.json" -ErrorAction SilentlyContinue).Count

Write-Host "✓ Viewer generated with $overlayCount overlay files" -ForegroundColor Green
Write-Host ""

# Step 4: Launch viewer
Write-Host "=== Ready to View ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the viewer:" -ForegroundColor White
Write-Host "  cd `"$viewerDir`"" -ForegroundColor Yellow
Write-Host "  python -m http.server 8080" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then open: http://localhost:8080/index.html" -ForegroundColor Green
Write-Host ""

# Offer to start server
$response = Read-Host "Start Python HTTP server now? (Y/n)"
if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
    Write-Host "Starting server on http://localhost:8080..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host ""
    
    Set-Location $viewerDir
    python -m http.server 8080
}
