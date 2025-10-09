#!/usr/bin/env pwsh
# Simple HTTP server for testing the plugin system
# Serves the WoWRollback.Viewer/assets directory

param(
    [int]$Port = 8081,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

Write-Host "🧪 Plugin System Test Server" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Find the assets directory
$assetsDir = Join-Path $PSScriptRoot "WoWRollback.Viewer\assets"

if (-not (Test-Path $assetsDir)) {
    Write-Host "❌ Assets directory not found: $assetsDir" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Serving directory: $assetsDir" -ForegroundColor Green
Write-Host "🌐 Port: $Port" -ForegroundColor Green
Write-Host ""

# Check if Python is available
$pythonCmd = $null
foreach ($cmd in @('python', 'python3', 'py')) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $pythonCmd = $cmd
        break
    }
}

if (-not $pythonCmd) {
    Write-Host "❌ Python not found. Please install Python to run the test server." -ForegroundColor Red
    Write-Host "   Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Get Python version
$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "🐍 Using: $pythonVersion" -ForegroundColor Green
Write-Host ""

# URLs to test
$testUrl = "http://localhost:$Port/test-plugin-system.html"
$mainUrl = "http://localhost:$Port/index.html"

Write-Host "📋 Available Test Pages:" -ForegroundColor Cyan
Write-Host "   Plugin Test: $testUrl" -ForegroundColor White
Write-Host "   Main Viewer: $mainUrl" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Starting server..." -ForegroundColor Yellow
Write-Host "   Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# Open browser if requested
if ($OpenBrowser) {
    Start-Sleep -Seconds 2
    Write-Host "🌐 Opening browser..." -ForegroundColor Green
    Start-Process $testUrl
}

# Start Python HTTP server
Push-Location $assetsDir
try {
    # Try Python 3 syntax first
    & $pythonCmd -m http.server $Port --bind localhost
} catch {
    # Fallback to Python 2 syntax
    & $pythonCmd -m SimpleHTTPServer $Port
} finally {
    Pop-Location
}
