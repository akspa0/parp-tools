#!/usr/bin/env pwsh
# Regenerate comparison data WITH coordinate extraction from converted LK ADTs

param(
    [string]$ConvertedAdtDir = "i:\parp-tools\pm4next-branch\parp-tools\gillijimproject_refactor\output_dirfart2\World\Maps",
    [string[]]$Versions = @("0.5.3.3368", "0.5.5.3494"),
    [string[]]$Maps = @("Kalimdor", "Azeroth"),
    [string]$TestDataRoot = "..\test_data",
    [string]$OutputRoot = "rollback_outputs"
)

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  WoWRollback: Regenerate with Coordinate Extraction       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build
Write-Host "[1/4] Building solution..." -ForegroundColor Yellow
dotnet build WoWRollback.sln --configuration Release --verbosity quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

$exePath = ".\WoWRollback.Cli\bin\Release\net9.0\WoWRollback.Cli.exe"

# Step 2: Analyze each version/map with coordinate extraction
Write-Host "[2/4] Analyzing Alpha WDTs with coordinate extraction..." -ForegroundColor Yellow

foreach ($version in $Versions) {
    foreach ($map in $Maps) {
        $wdtPath = Join-Path $TestDataRoot "$($version.Substring(0,5))\tree\World\Maps\${map}\${map}.wdt"
        $adtDir = Join-Path $ConvertedAdtDir $map
        
        if (!(Test-Path $wdtPath)) {
            Write-Host "  ⚠ Skipping ${version}/${map} - WDT not found: ${wdtPath}" -ForegroundColor Yellow
            continue
        }
        
        if (!(Test-Path $adtDir)) {
            Write-Host "  ⚠ Warning: Converted ADT dir not found: ${adtDir}" -ForegroundColor Yellow
            Write-Host "    (Will proceed but coordinates will be 0,0,0)" -ForegroundColor Gray
            $adtDir = $null
        }
        
        Write-Host "  → Analyzing ${version} / ${map}..." -ForegroundColor Cyan
        
        $args = @(
            "analyze-alpha-wdt",
            "--wdt-file", $wdtPath,
            "--out", $OutputRoot
        )
        
        if ($adtDir) {
            $args += "--converted-adt-dir"
            $args += $adtDir
            Write-Host "    Using converted ADTs: $adtDir" -ForegroundColor Gray
        }
        
        & $exePath @args
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ✗ Analysis failed for ${version}/${map}" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "  ✓ Analyzed ${version}/${map}" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "✓ All analyses complete" -ForegroundColor Green
Write-Host ""

# Step 3: Verify coordinates in CSV
Write-Host "[3/4] Verifying coordinate extraction..." -ForegroundColor Yellow

$verificationResults = @()

foreach ($version in $Versions) {
    foreach ($map in $Maps) {
        # CSVs with coordinates are in: {version}/{map}/assets_alpha_{map}.csv
        $csvPattern = Join-Path $OutputRoot "${version}\${map}\assets_alpha_${map}.csv"
        $csvFile = Get-Item -Path $csvPattern -ErrorAction SilentlyContinue
        
        if (!$csvFile) {
            Write-Host "  ⚠ Assets CSV not found: ${csvPattern}" -ForegroundColor Yellow
            continue
        }
        
        # Read first few lines to check for coordinates
        $lines = Get-Content $csvFile.FullName -TotalCount 5
        $header = $lines[0]
        
        # Check for world_x, world_y, world_z columns (note: underscore, not camelCase)
        if ($header -match "world_x.*world_y.*world_z") {
            Write-Host "  ✓ ${version}/${map}: Has coordinate columns" -ForegroundColor Green
            
            # Check if any non-zero coordinates exist
            $sampleLine = $lines[1]
            if ($sampleLine -match ",(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)") {
                $x = [double]$matches[1]
                $y = [double]$matches[2]
                $z = [double]$matches[3]
                
                if ($x -eq 0 -and $y -eq 0 -and $z -eq 0) {
                    Write-Host "    ⚠ Warning: Sample shows (0,0,0) - may need converted ADTs" -ForegroundColor Yellow
                    $verificationResults += [PSCustomObject]@{
                        Version = $version
                        Map = $map
                        Status = "Zero"
                    }
                } else {
                    Write-Host "    ✓ Sample coords: ($($x.ToString('F2')), $($y.ToString('F2')), $($z.ToString('F2')))" -ForegroundColor Green
                    $verificationResults += [PSCustomObject]@{
                        Version = $version
                        Map = $map
                        Status = "OK"
                    }
                }
            }
        } else {
            Write-Host "  ✗ ${version}/${map}: Missing coordinate columns!" -ForegroundColor Red
            $verificationResults += [PSCustomObject]@{
                Version = $version
                Map = $map
                Status = "Missing"
            }
        }
    }
}

Write-Host ""

# Step 4: Generate comparison with viewer
Write-Host "[4/4] Generating version comparison..." -ForegroundColor Yellow

$versionList = $Versions -join ","
$mapList = $Maps -join ","

& $exePath compare-versions `
    --versions $versionList `
    --maps $mapList `
    --viewer-report `
    --out $OutputRoot

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Comparison generation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Comparison complete" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Summary                                                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Coordinate Verification:" -ForegroundColor White
$okCount = ($verificationResults | Where-Object Status -eq "OK").Count
$zeroCount = ($verificationResults | Where-Object Status -eq "Zero").Count
$missingCount = ($verificationResults | Where-Object Status -eq "Missing").Count

Write-Host "  ✓ With coordinates: $okCount" -ForegroundColor Green
if ($zeroCount -gt 0) {
    Write-Host "  ⚠ All zeros: $zeroCount (need converted ADTs)" -ForegroundColor Yellow
}
if ($missingCount -gt 0) {
    Write-Host "  ✗ Missing columns: $missingCount" -ForegroundColor Red
}

Write-Host ""

# Find viewer directory
$comparisonKey = ($Versions | ForEach-Object { $_.Replace(".", "_") }) -join "_vs_"
$viewerDir = Join-Path $OutputRoot "comparisons\$comparisonKey\viewer"

if (Test-Path $viewerDir) {
    $overlayCount = (Get-ChildItem -Path (Join-Path $viewerDir "overlays") -Recurse -Filter "*.json" -ErrorAction SilentlyContinue).Count
    Write-Host "Viewer:" -ForegroundColor White
    Write-Host "  Location: $viewerDir" -ForegroundColor Gray
    Write-Host "  Overlay files: $overlayCount" -ForegroundColor Gray
    Write-Host ""
    
    # Offer to start server
    Write-Host "Start viewer? (Y/n): " -NoNewline -ForegroundColor Cyan
    $response = Read-Host
    
    if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
        Write-Host ""
        Write-Host "Starting server at http://localhost:8080..." -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        
        Set-Location $viewerDir
        python -m http.server 8080
    }
} else {
    Write-Host "⚠ Viewer directory not found: $viewerDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
