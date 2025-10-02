#!/usr/bin/env pwsh
# Regenerate comparison data WITH coordinate extraction from converted LK ADTs

param(
    [string]$ConvertedAdtDir,
    [string]$ConvertedCacheRoot,
    [string[]]$Versions = @("0.5.3.3368", "0.5.5.3494"),
    [string[]]$Maps = @("Kalimdor", "Azeroth"),
    [string]$TestDataRoot = "..\test_data",
    [string]$OutputRoot = "rollback_outputs",
    [string]$ImageFormat = "webp",
    [int]$ImageQuality = 85,
    [string]$ListfilePath,
    [string]$LkListfilePath,
    [switch]$ForceConvert,
    [switch]$CleanConverted,
    [switch]$SkipViewerPrompt
)

$ErrorActionPreference = "Stop"

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  WoWRollback: Regenerate with Coordinate Extraction and Caching  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

function Ensure-Directory {
    param([string]$Path)
{{ ... }}
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    if (-not (Test-Path $Path)) {
        [void](New-Item -ItemType Directory -Path $Path)
    }
}

function Resolve-DefaultListfile {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if (Test-Path $candidate) {
            $resolved = Resolve-Path -Path $candidate -ErrorAction SilentlyContinue
            if ($resolved) { return $resolved.Path }
        }
    }
    return $null
}

function Resolve-ConvertedMapDir {
    param(
        [string]$BaseDir,
        [string]$Map
    )

    if ([string]::IsNullOrWhiteSpace($BaseDir) -or [string]::IsNullOrWhiteSpace($Map)) {
        return $null
    }

    $directCandidate = Join-Path $BaseDir $Map
    $mapsRoot = Join-Path $BaseDir "World\Maps"
    $nestedCandidate = Join-Path $mapsRoot $Map

    $candidates = @($directCandidate, $nestedCandidate)

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }

    return $candidates[1]
}

function Invoke-AlphaConversion {
    param(
        [string]$ProjectPath,
        [string]$WdtPath,
        [string]$Version,
        [string]$Map,
        [string]$ExportRoot,
        [string]$Listfile,
        [string]$LkListfile,
        [switch]$Force
    )

    if (-not (Test-Path $WdtPath)) {
        Write-Host "  ⚠ Skipping ${Version}/${Map} - WDT not found: ${WdtPath}" -ForegroundColor Yellow
        return $null
    }

    Ensure-Directory -Path $ExportRoot
    $logsRoot = Join-Path $ExportRoot "__awdt_logs"
    Ensure-Directory -Path $logsRoot

    $existingDir = Resolve-ConvertedMapDir -BaseDir $ExportRoot -Map $Map
    if (-not $Force -and $existingDir -and (Get-ChildItem -Path $existingDir -Filter '*.adt' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1)) {
        Write-Host "  ↺ Reusing converted ADTs for ${Version}/${Map} at $existingDir" -ForegroundColor Gray
        return $existingDir
    }

    Write-Host "  → Converting ${Version}/${Map} to LK ADT format..." -ForegroundColor Cyan

    $dotnetArgs = @(
        'run',
        '--project', $ProjectPath,
        '--configuration', 'Release',
        '--',
        '--input', $WdtPath,
        '--listfile', $Listfile,
        '--out', (Join-Path $logsRoot $Version),
        '--export-adt',
        '--export-dir', $ExportRoot,
        '--no-web'
    )

    if (-not [string]::IsNullOrWhiteSpace($LkListfile)) {
        $dotnetArgs += @('--lk-listfile', $LkListfile)
    }

    & dotnet @dotnetArgs
    if ($LASTEXITCODE -ne 0) {
        throw "AlphaWDT conversion failed for ${Version}/${Map}"
    }

    $mapDir = Resolve-ConvertedMapDir -BaseDir $ExportRoot -Map $Map
    if (-not (Test-Path $mapDir)) {
        throw "Converted ADT directory not found after conversion: ${mapDir}"
    }

    $tileProbe = Get-ChildItem -Path $mapDir -Filter '*.adt' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $tileProbe) {
        throw "Conversion produced no ADT tiles for ${Version}/${Map} at ${mapDir}"
    }

    Write-Host "  ✓ Converted ADTs cached at $mapDir" -ForegroundColor Green
    return $mapDir
}

# Step 1: Build
Write-Host "[1/5] Building solution..." -ForegroundColor Yellow
dotnet build WoWRollback.sln --configuration Release --verbosity quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

$exePath = ".\WoWRollback.Cli\bin\Release\net9.0\WoWRollback.Cli.exe"

$explicitConverted = -not [string]::IsNullOrWhiteSpace($ConvertedAdtDir)
if ($explicitConverted) {
    $ConvertedCacheRoot = $ConvertedAdtDir
} elseif ([string]::IsNullOrWhiteSpace($ConvertedCacheRoot)) {
    $ConvertedCacheRoot = Join-Path $OutputRoot 'converted_adts'
}

if ($CleanConverted -and -not $explicitConverted -and (Test-Path $ConvertedCacheRoot)) {
    Write-Host "[convert] Cleaning cached converted ADTs at $ConvertedCacheRoot" -ForegroundColor Yellow
    Remove-Item $ConvertedCacheRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not $explicitConverted) {
    Ensure-Directory -Path $ConvertedCacheRoot
}

$mapConvertedLookup = @{}

if ($explicitConverted) {
    Write-Host "[2/5] Using provided converted ADT directory: $ConvertedCacheRoot" -ForegroundColor Yellow
    foreach ($version in $Versions) {
        foreach ($map in $Maps) {
            $resolvedDir = Resolve-ConvertedMapDir -BaseDir $ConvertedCacheRoot -Map $map
            if ($resolvedDir -and (Test-Path $resolvedDir)) {
                $mapConvertedLookup["$version|$map"] = $resolvedDir
            } else {
                Write-Host "  ⚠ Converted ADTs not found for ${version}/${map} under $ConvertedCacheRoot" -ForegroundColor Yellow
                $mapConvertedLookup["$version|$map"] = $null
            }
        }
    }
    Write-Host "✓ Conversion cache stage skipped (user-supplied ConvertedAdtDir)" -ForegroundColor Green
    Write-Host ""
} else {
    if (-not $PSBoundParameters.ContainsKey('ListfilePath')) {
        $ListfilePath = Resolve-DefaultListfile @(
            (Join-Path $PSScriptRoot "..\test_data\community-listfile-withcapitals.csv"),
            (Join-Path $PSScriptRoot "..\reference_data\community-listfile-withcapitals.csv"),
            (Join-Path $PSScriptRoot "..\lib\noggit-red\dist\listfile\listfile.csv")
        )
    }

    if ([string]::IsNullOrWhiteSpace($ListfilePath) -or -not (Test-Path $ListfilePath)) {
        Write-Host "✗ A community listfile is required for conversion but none was found." -ForegroundColor Red
        exit 1
    }

    if ($PSBoundParameters.ContainsKey('LkListfilePath') -and -not [string]::IsNullOrWhiteSpace($LkListfilePath) -and -not (Test-Path $LkListfilePath)) {
        Write-Host "✗ LK listfile not found: $LkListfilePath" -ForegroundColor Red
        exit 1
    }

    $alphaCliProject = Join-Path $PSScriptRoot "..\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli\AlphaWdtAnalyzer.Cli.csproj"
    if (-not (Test-Path $alphaCliProject)) {
        Write-Host "✗ AlphaWDTAnalysisTool CLI project not found at $alphaCliProject" -ForegroundColor Red
        exit 1
    }

    Write-Host "[2/5] Preparing LK-converted ADTs (with caching)..." -ForegroundColor Yellow

    foreach ($version in $Versions) {
        $versionRoot = Join-Path $ConvertedCacheRoot $version
        foreach ($map in $Maps) {
            $wdtPath = Join-Path $TestDataRoot ("{0}\tree\World\Maps\{1}\{1}.wdt" -f $version.Substring(0,5), $map)
            try {
                $mapDir = Invoke-AlphaConversion -ProjectPath $alphaCliProject -WdtPath $wdtPath -Version $version -Map $map -ExportRoot $versionRoot -Listfile $ListfilePath -LkListfile $LkListfilePath -Force:$ForceConvert
                $mapConvertedLookup["$version|$map"] = $mapDir
            } catch {
                Write-Host "  ✗ Conversion failed for ${version}/${map}: $_" -ForegroundColor Red
                exit 1
            }
        }
    }

    Write-Host "✓ Conversion stage complete" -ForegroundColor Green
    Write-Host ""
}

# Step 3: Analyze each version/map with coordinate extraction
Write-Host "[3/5] Analyzing Alpha WDTs with coordinate extraction..." -ForegroundColor Yellow

foreach ($version in $Versions) {
    foreach ($map in $Maps) {
        $wdtPath = Join-Path $TestDataRoot "$( $version.Substring(0,5) )\tree\World\Maps\${map}\${map}.wdt"
        $adtDir = $null
        $lookupKey = "$version|$map"
        if ($mapConvertedLookup.ContainsKey($lookupKey)) {
            $adtDir = $mapConvertedLookup[$lookupKey]
        }

        if (-not (Test-Path $wdtPath)) {
            Write-Host "  ⚠ Skipping ${version}/${map} - WDT not found: ${wdtPath}" -ForegroundColor Yellow
            continue
        }

        if ($adtDir -and -not (Test-Path $adtDir)) {
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
Write-Host "[4/5] Verifying coordinate extraction..." -ForegroundColor Yellow

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
Write-Host "[5/5] Generating version comparison..." -ForegroundColor Yellow

$versionList = $Versions -join ","
$mapList = $Maps -join ","

& $exePath compare-versions `
    --versions $versionList `
    --maps $mapList `
    --viewer-report `
    --image-format $ImageFormat `
    --image-quality $ImageQuality `
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
    
    if (-not $SkipViewerPrompt) {
        # Offer to start server
        Write-Host "Start viewer? (Y/n): " -NoNewline -ForegroundColor Cyan
        $response = Read-Host
        
        if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
            Write-Host ""
            Write-Host "Starting server at http://localhost:8080..." -ForegroundColor Cyan
            Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
            Write-Host ""
            
            Push-Location $viewerDir
            try {
                python -m http.server 8080
            } finally {
                Pop-Location
            }
        }
    }
} else {
    Write-Host "Viewer directory not found: $viewerDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
