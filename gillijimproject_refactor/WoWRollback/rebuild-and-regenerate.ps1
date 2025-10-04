#!/usr/bin/env pwsh
# Rebuild WoWRollback with coordinate extraction and regenerate viewer data

[CmdletBinding(PositionalBinding=$false)]
param(
    [string[]]$Maps = @("DeadminesInstance"),
    [string[]]$Versions = @("0.5.3.3368","0.5.5.3494"),
    [string]$AlphaRoot,
    [string]$ConvertedAdtRoot,
    [string]$CacheRoot = "cached_maps",
    [switch]$RefreshCache,
    [switch]$Serve,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

# Ensure all relative paths resolve from this script's directory (WoWRollback/)
Set-Location -Path $PSScriptRoot

Write-Host "=== WoWRollback Rebuild & Regenerate ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean build
Write-Host "[1/4] Building solution..." -ForegroundColor Yellow
dotnet build WoWRollback.sln --configuration Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# Helper: Discover map names from AlphaRoot when -Maps auto
# Searches version-specific paths: test_data/<version>/tree/World/Maps/<MapName>/<MapName>.wdt
function Get-MapsFromAlphaRoot([string]$root, [string[]]$versions) {
    if (-not $root -or -not (Test-Path $root)) { return @() }
    $results = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
    
    # Search version-specific paths
    foreach ($ver in $versions) {
        $versionPaths = @(
            Join-Path $root "$ver\tree\World\Maps",
            Join-Path $root "$ver\World\Maps",
            Join-Path $root "0.5.3\tree\World\Maps",  # Fallback for common structure
            Join-Path $root "tree\World\Maps"          # Legacy structure
        )
        
        foreach ($p in $versionPaths) {
            if (Test-Path $p) {
                Get-ChildItem -Path $p -Directory -ErrorAction SilentlyContinue | ForEach-Object {
                    $mapName = $_.Name
                    $wdtPath = Join-Path $_.FullName "$mapName.wdt"
                    if (Test-Path $wdtPath) {
                        [void]$results.Add($mapName)
                    }
                }
            }
        }
        
        # Also check Minimaps folder
        $miniPaths = @(
            Join-Path $root "$ver\tree\World\Minimaps",
            Join-Path $root "$ver\World\Minimaps"
        )
        foreach ($mp in $miniPaths) {
            if (Test-Path $mp) {
                Get-ChildItem -Path $mp -Directory -ErrorAction SilentlyContinue |
                    ForEach-Object { [void]$results.Add($_.Name) }
            }
        }
    }
    
    $discovered = $results.ToArray() | Sort-Object
    if ($discovered.Count -gt 0) {
        Write-Host "[auto-discovery] Found $($discovered.Count) maps: $($discovered -join ', ')" -ForegroundColor Cyan
    }
    return $discovered
}

# Step 2: Regenerate comparison data
Write-Host "[2/4] Preparing cached Alpha -> LK ADTs..." -ForegroundColor Yellow

$cacheRootPath = Join-Path $PSScriptRoot $CacheRoot

function Resolve-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $full = Resolve-Path -Path $candidate -ErrorAction SilentlyContinue
        if ($full) { return $full.Path }
    }
    return $null
}

$communityListfile = Resolve-FirstExistingPath @(
    (Join-Path $PSScriptRoot "test_data/community-listfile-withcapitals.csv"),
    (Join-Path $PSScriptRoot "../test_data/community-listfile-withcapitals.csv")
)
$lkListfile = Resolve-FirstExistingPath @(
    (Join-Path $PSScriptRoot "test_data/World of Warcraft 3x.txt"),
    (Join-Path $PSScriptRoot "../test_data/World of Warcraft 3x.txt")
)
$alphaToolProject = Join-Path $PSScriptRoot "..\AlphaWDTAnalysisTool\AlphaWdtAnalyzer.Cli\AlphaWdtAnalyzer.Cli.csproj"

if (-not (Test-Path $cacheRootPath)) {
    New-Item -Path $cacheRootPath -ItemType Directory | Out-Null
}

function Get-WdtScore {
    param(
        [string]$Version,
        [string]$Path
    )
    if ($Path -like "*${Version}*") { return 0 }
    $prefix = $Version.Substring(0, [Math]::Min(5, $Version.Length))
    if ($Path -like "*${prefix}*") { return 1 }
    $parts = $Version.Split('.')
    if ($parts.Length -ge 2) {
        $majorMinor = "$($parts[0]).$($parts[1])"
        if ($Path -like "*${majorMinor}*") { return 2 }
    }
    return 3
}

function Find-WdtPath {
    param(
        [string]$Root,
        [string]$Version,
        [string]$Map
    )
    if (-not (Test-Path $Root)) { return $null }
    $candidates = Get-ChildItem -Path $Root -Recurse -File -Filter "$Map.wdt" -ErrorAction SilentlyContinue
    if (-not $candidates) { return $null }
    $sorted = $candidates |
        Sort-Object @{Expression = { Get-WdtScore -Version $Version -Path $_.FullName }} , @{Expression = { $_.FullName.Length }}
    return $sorted[0].FullName
}

function Ensure-CachedMap {
    param(
        [string]$Version,
        [string]$Map
    )

    $versionRoot = Join-Path $cacheRootPath $Version
    $worldMapsRoot = Join-Path (Join-Path $versionRoot 'World') 'Maps'
    $mapRoot = Join-Path $worldMapsRoot $Map
    $analysisRoot = Join-Path (Join-Path $cacheRootPath 'analysis') $Version
    $analysisDir = Join-Path $analysisRoot $Map

    $needsRefresh = $RefreshCache.IsPresent -or -not (Test-Path $mapRoot)
    if (-not $needsRefresh) {
        Write-Host "  [cache] Reusing $Version/$Map" -ForegroundColor DarkGray
        return $mapRoot
    }

    if (Test-Path $mapRoot) {
        Remove-Item $mapRoot -Recurse -Force
    }
    if (Test-Path $analysisDir) {
        Remove-Item $analysisDir -Recurse -Force
    }

    New-Item -Path $versionRoot -ItemType Directory -Force | Out-Null
    New-Item -Path $worldMapsRoot -ItemType Directory -Force | Out-Null
    New-Item -Path $mapRoot -ItemType Directory -Force | Out-Null
    New-Item -Path $analysisRoot -ItemType Directory -Force | Out-Null
    New-Item -Path $analysisDir -ItemType Directory -Force | Out-Null

    if (-not $AlphaRoot) {
        throw "AlphaRoot must be provided to build cached maps"
    }

    $wdtPath = Find-WdtPath -Root $AlphaRoot -Version $Version -Map $Map
    if (-not $wdtPath) {
        throw "Unable to locate $Map.wdt beneath $AlphaRoot for version $Version"
    }

    if (-not (Test-Path $communityListfile)) {
        throw "Missing community listfile at $communityListfile"
    }
    if (-not (Test-Path $lkListfile)) {
        throw "Missing LK listfile at $lkListfile"
    }
    if (-not (Test-Path $alphaToolProject)) {
        throw "AlphaWDTAnalysisTool project not found at $alphaToolProject"
    }

    Write-Host "  [cache] Building LK ADTs for $Version/$Map" -ForegroundColor Yellow
    $tempExportDir = Join-Path $cacheRootPath ("_awdt_tmp_" + ($Version -replace '[^0-9A-Za-z]', '_') + "_" + ($Map -replace '[^0-9A-Za-z]', '_'))
    if (Test-Path $tempExportDir) {
        Remove-Item $tempExportDir -Recurse -Force
    }
    New-Item -Path $tempExportDir -ItemType Directory -Force | Out-Null

    $toolArgs = @(
        'run','--project',$alphaToolProject,'--configuration','Release','--',
        '--input', $wdtPath,
        '--listfile', $communityListfile,
        '--lk-listfile', $lkListfile,
        '--out', $analysisDir,
        '--export-adt',
        '--export-dir', $tempExportDir,
        '--extract-mcnk-terrain',
        '--no-web',
        '--profile', 'raw',
        '--no-fallbacks'
    )

    Write-Host "  [debug] Running AlphaWdtAnalyzer with --extract-mcnk-terrain" -ForegroundColor Magenta
    Write-Host "  [debug] Analysis output dir: $analysisDir" -ForegroundColor Magenta
    & dotnet @toolArgs
    if ($LASTEXITCODE -ne 0) {
        throw "AlphaWdtAnalyzer conversion failed for $Version/$Map"
    }
    
    # Verify terrain CSV was created
    $expectedTerrainCsv = Join-Path (Join-Path (Join-Path $analysisDir 'csv') $Map) ($Map + '_mcnk_terrain.csv')
    if (Test-Path $expectedTerrainCsv) {
        Write-Host "  [debug] ✓ Terrain CSV created: $expectedTerrainCsv" -ForegroundColor Green
    } else {
        Write-Host "  [warn] ✗ Terrain CSV NOT created at: $expectedTerrainCsv" -ForegroundColor Red
        Write-Host "  [warn] Check AlphaWdtAnalyzer output above for errors" -ForegroundColor Yellow
    }

    $sourceMapDir = Join-Path (Join-Path (Join-Path $tempExportDir 'World') 'Maps') $Map
    if (-not (Test-Path $sourceMapDir)) {
        throw "Expected converted ADTs at $sourceMapDir but none were produced"
    }

    if (Test-Path $mapRoot) {
        Remove-Item $mapRoot -Recurse -Force
    }
    New-Item -Path $mapRoot -ItemType Directory -Force | Out-Null
    Copy-Item -Path (Join-Path $sourceMapDir '*') -Destination $mapRoot -Recurse -Force

    Remove-Item $tempExportDir -Recurse -Force
    
    # Copy terrain CSVs to rollback_outputs for viewer generation
    $rollbackVersionRoot = Join-Path $PSScriptRoot ('rollback_outputs\' + $Version)
    $rollbackMapCsvDir = Join-Path $rollbackVersionRoot ('csv\' + $Map)
    if (-not (Test-Path $rollbackMapCsvDir)) {
        New-Item -Path $rollbackMapCsvDir -ItemType Directory -Force | Out-Null
    }
    
    # Copy MCNK terrain CSV (AlphaWdtAnalyzer outputs to csv/MapName/MapName_mcnk_terrain.csv)
    $terrainCsvDir = Join-Path (Join-Path $analysisDir 'csv') $Map
    $terrainCsv = Join-Path $terrainCsvDir ($Map + '_mcnk_terrain.csv')
    if (Test-Path $terrainCsv) {
        Copy-Item -Path $terrainCsv -Destination $rollbackMapCsvDir -Force
        Write-Host "  [cache] Copied terrain CSV to rollback_outputs" -ForegroundColor DarkGray
    } else {
        Write-Host "  [warn] Terrain CSV not found at: $terrainCsv" -ForegroundColor Yellow
    }
    
    # Copy AreaTable CSVs to version root (one per version, not per map)
    # Look in DBCTool.V2 outputs for AreaTable dumps
    $dbcToolOutputs = Join-Path $PSScriptRoot '..\DBCTool.V2\dbctool_outputs'
    if (Test-Path $dbcToolOutputs) {
        $latestSession = Get-ChildItem -Path $dbcToolOutputs -Directory | 
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestSession) {
            $areaTableAlpha = Join-Path $latestSession.FullName 'compare\v2\AreaTable_dump_0.5.5.csv'
            $areaTableLk = Join-Path $latestSession.FullName 'compare\v2\AreaTable_dump_3.3.5.csv'
            
            if (Test-Path $areaTableAlpha) {
                $destAlpha = Join-Path $rollbackVersionRoot 'AreaTable_Alpha.csv'
                if (-not (Test-Path $destAlpha)) {
                    Copy-Item -Path $areaTableAlpha -Destination $destAlpha -Force
                    Write-Host "  [cache] Copied AreaTable_Alpha.csv" -ForegroundColor DarkGray
                }
            }
            if (Test-Path $areaTableLk) {
                $destLk = Join-Path $rollbackVersionRoot 'AreaTable_335.csv'
                if (-not (Test-Path $destLk)) {
                    Copy-Item -Path $areaTableLk -Destination $destLk -Force
                    Write-Host "  [cache] Copied AreaTable_335.csv" -ForegroundColor DarkGray
                }
            }
        }
    }

    return $mapRoot
}

$ensuredMaps = @{}
foreach ($version in $Versions) {
    foreach ($map in $Maps) {
        $key = "$version::$map"
        $ensuredMaps[$key] = Ensure-CachedMap -Version $version -Map $map
    }
}

Write-Host "✓ Cached maps ready" -ForegroundColor Green
Write-Host ""

Write-Host "[3/4] Regenerating comparison data (this may take several minutes)..." -ForegroundColor Yellow
$versionsArg = ($Versions -join ',')

# Auto-discover maps if requested
if ($Maps.Count -eq 1 -and $Maps[0].ToLower() -eq 'auto') {
    if (-not $AlphaRoot) {
        Write-Host "-Maps auto requires -AlphaRoot" -ForegroundColor Red
        exit 1
    }
    $discovered = Get-MapsFromAlphaRoot -root $AlphaRoot -versions $Versions
    if (-not $discovered -or $discovered.Count -eq 0) {
        Write-Host "No maps discovered under $AlphaRoot" -ForegroundColor Yellow
        Write-Host "Defaulting to common maps" -ForegroundColor Gray
        $Maps = @('Azeroth','Kalimdor','Kalidar','PVPZone01','Shadowfang','DeadminesInstance')
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
   '--viewer-report',
   '--converted-adt-cache', $cacheRootPath
)
if ($AlphaRoot) { $cmd += @('--alpha-root', $AlphaRoot) }
if ($ConvertedAdtRoot) { $cmd += @('--converted-adt-root', $ConvertedAdtRoot) }
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
Write-Host "[4/4] Locating viewer output..." -ForegroundColor Yellow

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
