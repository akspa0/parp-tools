# compile_openscad_assets.ps1
# Compiles all .scad files under src/viewer/WoWViewer/Assets/OpenScad/ to .off

$ErrorActionPreference = "Stop"

$OpenScadExe = "C:\Program Files\OpenSCAD\openscad.com"
if (-not (Test-Path $OpenScadExe)) {
    $OpenScadExe = "openscad"
}

$AssetsDir = Join-Path $PSScriptRoot "..\src\viewer\WoWViewer\Assets\OpenScad"
$ScadFiles = Get-ChildItem -Path $AssetsDir -Filter "*.scad"

Write-Host "Compiling $($ScadFiles.Count) OpenSCAD asset(s) to .off in $AssetsDir..." -ForegroundColor Cyan

foreach ($file in $ScadFiles) {
    $offPath = [System.IO.Path]::ChangeExtension($file.FullName, ".off")
    Write-Host "  -> Compiling $($file.Name) to $([System.IO.Path]::GetFileName($offPath))..."
    & $OpenScadExe $file.FullName -o $offPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $($file.FullName)"
    }
}

Write-Host "All OpenSCAD assets compiled successfully." -ForegroundColor Green
