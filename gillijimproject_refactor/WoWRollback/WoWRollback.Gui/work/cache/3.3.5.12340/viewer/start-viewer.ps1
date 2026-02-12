#!/usr/bin/env pwsh
# Simple HTTP server launcher for WoW Rollback Viewer

param(
    [int]$Port = 8080
)

Write-Host "Starting WoW Rollback Viewer..." -ForegroundColor Green
Write-Host "Port: $Port" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $pythonCmd = $cmd
        break
    }
}

if ($pythonCmd) {
    Write-Host "Using Python HTTP server..." -ForegroundColor Yellow
    Write-Host "Open your browser to: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host ""
    
    & $pythonCmd -m http.server $Port
}
else {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python or use one of these alternatives:" -ForegroundColor Yellow
    Write-Host "  1. Node.js: npx http-server -p $Port" -ForegroundColor Gray
    Write-Host "  2. VS Code: Install 'Live Server' extension" -ForegroundColor Gray
    exit 1
}
