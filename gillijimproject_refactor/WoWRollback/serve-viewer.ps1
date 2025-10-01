# Simple HTTP server for WoWRollback viewer
# Serves the latest comparison viewer on http://localhost:8080

$Port = 8080

# Find latest viewer directory
$comparisonsDir = "rollback_outputs\comparisons"

if (-not (Test-Path $comparisonsDir)) {
    Write-Host "❌ No comparisons found. Run regeneration first." -ForegroundColor Red
    exit 1
}

$comparisonDirs = Get-ChildItem $comparisonsDir -Directory |
    Where-Object { $_.Name -like "*_vs_*" } |
    Sort-Object LastWriteTime -Descending
if ($comparisonDirs.Count -eq 0) {
    Write-Host "❌ No comparison directories found (need *_vs_* format)." -ForegroundColor Red
    exit 1
}

$viewerDir = Join-Path $comparisonDirs[0].FullName "viewer"

if (-not (Test-Path $viewerDir)) {
    Write-Host "❌ Viewer directory not found: $viewerDir" -ForegroundColor Red
    exit 1
}

Write-Host "🌐 Starting web server..." -ForegroundColor Cyan
Write-Host "📁 Serving: $viewerDir" -ForegroundColor Gray
Write-Host "🔗 URL: http://localhost:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Server running. Press Ctrl+C to stop." -ForegroundColor Green
Write-Host ""

# Change to viewer directory
Set-Location $viewerDir

# Open browser
Start-Process "http://localhost:$Port/index.html"

# Start Python HTTP server
python -m http.server $Port
