#!/usr/bin/env pwsh
# Rebuild WoWRollback with coordinate extraction and regenerate viewer data

[CmdletBinding(PositionalBinding=$false)]
param(
    [string[]]$Maps = @("DeadminesInstance"),
    [string[]]$Versions = @("0.5.3.3368","0.5.5.3494"),
    [string]$AlphaRoot,
    [string]$ConvertedAdtRoot,
    [string]$CacheRoot = "cached_maps",
    [switch]$RefreshCache,        # Force full rebuild: delete ADTs + CSVs, regenerate everything
    [switch]$RefreshAnalysis,     # Delete only CSVs, regenerate from existing LK ADTs
    [switch]$RefreshOverlays,     # Delete viewer overlays, regenerate from existing CSVs
    [switch]$Serve,
    [switch]$UseNewViewerAssets,  # Phase 1: Use WoWRollback.Viewer project assets
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

# Ensure all relative paths resolve from this script's directory (WoWRollback/)
Set-Location -Path $PSScriptRoot

# === LOGGING INFRASTRUCTURE ===
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRootDir = Join-Path $PSScriptRoot "logs\session_$timestamp"
New-Item -Path $logRootDir -ItemType Directory -Force | Out-Null

$script:currentStage = "init"
$script:currentLogFile = Join-Path $logRootDir "00_init.log"

function Write-Log {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewline
    )
    
    $timePrefix = "[$(Get-Date -Format 'HH:mm:ss')]"
    $fullMessage = "$timePrefix $Message"
    
    # Write to console
    if ($NoNewline) {
        Write-Host $Message -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Message -ForegroundColor $Color
    }
    
    # Write to current log file (strip ANSI color codes)
    if ($script:currentLogFile) {
        Add-Content -Path $script:currentLogFile -Value $fullMessage -Encoding UTF8
    }
}

function Set-LogStage {
    param([string]$StageName, [int]$StageNumber)
    $script:currentStage = $StageName
    $logFileName = "{0:D2}_{1}.log" -f $StageNumber, $StageName
    $script:currentLogFile = Join-Path $logRootDir $logFileName
    Write-Log "=== STAGE: $StageName ===" -Color Cyan
}

function Invoke-LoggedCommand {
    param(
        [string]$Description,
        [scriptblock]$Command
    )
    
    Write-Log "[exec] $Description" -Color Yellow
    $output = & $Command 2>&1
    $output | ForEach-Object { 
        Write-Log "  $_" -Color Gray
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "[exec] FAILED with exit code $LASTEXITCODE" -Color Red
        throw "Command failed: $Description"
    }
    
    return $output
}

Write-Log "=== WoWRollback Rebuild & Regenerate ===" -Color Cyan
Write-Log "    Automatic pipeline: DBCTool → AlphaWdtAnalyzer → Viewer generation" -Color DarkGray
Write-Log "    Log directory: $logRootDir" -Color DarkGray
Write-Log ""

# Step 0: Auto-run DBCTool.V2 if needed (handled later after path resolution)

# Step 1: Clean build
Set-LogStage -StageName "build" -StageNumber 1
Write-Log "[1/5] Building solution..." -Color Yellow
$buildOutput = Invoke-LoggedCommand -Description "dotnet build" -Command {
    dotnet build WoWRollback.sln --configuration Release 2>&1
}
Write-Log "✓ Build complete" -Color Green
Write-Log ""

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
Set-LogStage -StageName "cache_preparation" -StageNumber 2
Write-Log "[2/5] Preparing cached Alpha -> LK ADTs..." -Color Yellow

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

# DBCTool paths for AreaTable crosswalk mappings
$dbcToolProject = Join-Path $PSScriptRoot '..\DBCTool.V2\DbcTool.Cli\DbcTool.Cli.csproj'
$dbcToolOutputs = Join-Path $PSScriptRoot '..\DBCTool.V2\dbctool_outputs'
$dbctoolPatchDir = $null
$dbctoolLkDir = Resolve-FirstExistingPath @(
    (Join-Path $PSScriptRoot "..\test_data\3.3.5\tree\DBFilesClient"),
    (Join-Path $PSScriptRoot "test_data\3.3.5\tree\DBFilesClient")
)

# Helper function to run DBCTool.V2 automatically
function Invoke-DBCTool {
    Write-Host "[auto] Running DBCTool.V2 to generate AreaTable crosswalks..." -ForegroundColor Cyan
    
    if (-not (Test-Path $dbcToolProject)) {
        Write-Host "[auto] ✗ DBCTool.V2 project not found at: $dbcToolProject" -ForegroundColor Red
        return $false
    }
    
    # Run DBCTool with default settings
    $dbcArgs = @('run', '--project', $dbcToolProject, '--configuration', 'Release')
    
    Write-Host "[auto] Executing: dotnet run --project DbcTool.Cli.csproj" -ForegroundColor Yellow
    & dotnet @dbcArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[auto] ✓ DBCTool.V2 completed successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[auto] ✗ DBCTool.V2 failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        return $false
    }
}

# Find latest DBCTool session for AreaTable crosswalks
function Get-LatestDBCToolSession {
    if (-not (Test-Path $dbcToolOutputs)) {
        return $null
    }
    
    $latestSession = Get-ChildItem -Path $dbcToolOutputs -Directory | 
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($latestSession) {
        $compareV2Dir = Join-Path $latestSession.FullName 'compare\v2'
        if (Test-Path $compareV2Dir) {
            return $compareV2Dir
        }
    }
    
    return $null
}

$dbctoolPatchDir = Get-LatestDBCToolSession

# Auto-run DBCTool if crosswalks are missing
if (-not $dbctoolPatchDir) {
    Write-Host "[auto] AreaTable crosswalks not found, running DBCTool.V2 automatically..." -ForegroundColor Yellow
    
    if (Invoke-DBCTool) {
        # Try to find session again after running DBCTool
        Start-Sleep -Seconds 1
        $dbctoolPatchDir = Get-LatestDBCToolSession
        
        if ($dbctoolPatchDir) {
            Write-Host "[auto] ✓ DBCTool AreaTable crosswalks generated: $dbctoolPatchDir" -ForegroundColor Green
        } else {
            Write-Host "[auto] ✗ DBCTool ran but crosswalks still not found!" -ForegroundColor Red
            Write-Host "[warn] Continuing without AreaTable mappings - AreaNames will be incorrect!" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[warn] DBCTool failed - continuing without AreaTable mappings" -ForegroundColor Yellow
    }
} else {
    Write-Host "[auto] ✓ Found existing DBCTool AreaTable crosswalks: $dbctoolPatchDir" -ForegroundColor Green
}

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

function Get-ExpectedTileCount {
    param(
        [string]$WdtPath,
        [string]$AlphaToolProject
    )
    
    if (-not (Test-Path $WdtPath)) { return 0 }
    
    try {
        # Use AlphaWdtAnalyzer to read tile count (works for both Alpha and LK WDTs)
        $count = & dotnet run --project $AlphaToolProject --configuration Release -- --count-tiles --input $WdtPath 2>$null
        
        if ($LASTEXITCODE -eq 0 -and $count -match '^\d+$') {
            return [int]$count
        }
        
        Write-Host "  [debug] Could not read tile count from WDT, assuming no validation needed" -ForegroundColor DarkGray
        return 0
    } catch {
        Write-Host "  [warn] Failed to read WDT tile count: $_" -ForegroundColor Yellow
        return 0
    }
}

function Test-CacheComplete {
    param(
        [string]$MapRoot,
        [int]$ExpectedTiles
    )
    
    if (-not (Test-Path $MapRoot)) { return $false }
    
    # Count ADT files in cache
    $adtFiles = Get-ChildItem -Path $MapRoot -Filter "*.adt" -File -ErrorAction SilentlyContinue
    $actualCount = $adtFiles.Count
    
    if ($actualCount -eq 0) {
        return $false
    }
    
    # Allow some tolerance (WDT and WMO-only maps might have fewer)
    # But if we're significantly short, it's incomplete
    if ($ExpectedTiles -gt 0 -and $actualCount -lt $ExpectedTiles) {
        Write-Host "  [warn] Cache incomplete: $actualCount/$ExpectedTiles tiles" -ForegroundColor Yellow
        return $false
    }
    
    return $true
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

    # Find source WDT to check expected tile count
    $sourceWdt = $null
    $expectedTiles = 0
    if ($AlphaRoot) {
        $sourceWdt = Find-WdtPath -Root $AlphaRoot -Version $Version -Map $Map
        if ($sourceWdt) {
            $expectedTiles = Get-ExpectedTileCount -WdtPath $sourceWdt -AlphaToolProject $alphaToolProject
            if ($expectedTiles -gt 0) {
                Write-Host "  [cache] Expected $expectedTiles tiles from WDT" -ForegroundColor DarkGray
            }
        }
    }

    # Smart cache checking: verify all required outputs exist AND are complete
    $lkAdtsExist = Test-Path $mapRoot
    $terrainCsvPath = Join-Path (Join-Path (Join-Path $analysisDir 'csv') $Map) ($Map + '_mcnk_terrain.csv')
    $shadowCsvPath = Join-Path (Join-Path (Join-Path $analysisDir 'csv') $Map) ($Map + '_mcnk_shadows.csv')
    $analysisCsvsExist = (Test-Path $terrainCsvPath) -or (Test-Path $shadowCsvPath)
    
    # Check if cached ADTs are complete (all expected tiles present)
    $cacheComplete = Test-CacheComplete -MapRoot $mapRoot -ExpectedTiles $expectedTiles
    
    # Handle RefreshAnalysis: delete CSVs but keep LK ADTs
    if ($RefreshAnalysis.IsPresent -and (Test-Path $analysisDir)) {
        Write-Host "  [cache] RefreshAnalysis flag set, deleting CSVs for $Version/$Map" -ForegroundColor Yellow
        Remove-Item $analysisDir -Recurse -Force -ErrorAction SilentlyContinue
        $analysisCsvsExist = $false
    }
    
    # Determine if refresh is needed
    if ($RefreshCache.IsPresent) {
        Write-Host "  [cache] RefreshCache flag set, rebuilding $Version/$Map" -ForegroundColor Yellow
        $needsRefresh = $true
    } elseif (-not $lkAdtsExist) {
        Write-Host "  [cache] LK ADTs missing for $Version/$Map, building..." -ForegroundColor Yellow
        $needsRefresh = $true
    } elseif (-not $cacheComplete) {
        Write-Host "  [cache] LK ADT cache incomplete for $Version/$Map, rebuilding..." -ForegroundColor Yellow
        $needsRefresh = $true
    } elseif (-not $analysisCsvsExist) {
        # CSVs missing - check if we can do CSV-only extraction (LK ADTs exist)
        if ($lkAdtsExist -and $cacheComplete) {
            Write-Host "  [cache] Analysis CSVs missing but LK ADTs exist for $Version/$Map" -ForegroundColor Yellow
            Write-Host "  [cache] Will extract CSVs from existing LK ADTs (fast path)" -ForegroundColor Cyan
            $needsRefresh = $false
            $needsCsvExtraction = $true
        } else {
            Write-Host "  [cache] Analysis CSVs missing for $Version/$Map, building..." -ForegroundColor Yellow
            $needsRefresh = $true
            $needsCsvExtraction = $false
        }
    } else {
        $adtCount = (Get-ChildItem -Path $mapRoot -Filter "*.adt" -File -ErrorAction SilentlyContinue).Count
        Write-Host "  [cache] ✓ Reusing cached data for $Version/$Map ($adtCount ADTs + CSVs)" -ForegroundColor Green
        $needsRefresh = $false
        $needsCsvExtraction = $false
    }
    
    if (-not $needsRefresh) {
        # Still need to copy CSVs to rollback_outputs if they're missing there
        $rollbackVersionRoot = Join-Path $PSScriptRoot ('rollback_outputs\' + $Version)
        $rollbackMapCsvDir = Join-Path $rollbackVersionRoot ('csv\' + $Map)
        
        if (-not (Test-Path $rollbackMapCsvDir)) {
            New-Item -Path $rollbackMapCsvDir -ItemType Directory -Force | Out-Null
        }
        
        # Copy terrain CSV if missing in rollback_outputs
        $rollbackTerrainCsv = Join-Path $rollbackMapCsvDir ($Map + '_mcnk_terrain.csv')
        if ((Test-Path $terrainCsvPath) -and -not (Test-Path $rollbackTerrainCsv)) {
            Copy-Item -Path $terrainCsvPath -Destination $rollbackMapCsvDir -Force
            Write-Host "  [cache] Copied cached terrain CSV to rollback_outputs" -ForegroundColor Cyan
        }
        
        # Copy shadow CSV if missing in rollback_outputs
        $rollbackShadowCsv = Join-Path $rollbackMapCsvDir ($Map + '_mcnk_shadows.csv')
        if ((Test-Path $shadowCsvPath) -and -not (Test-Path $rollbackShadowCsv)) {
            Copy-Item -Path $shadowCsvPath -Destination $rollbackMapCsvDir -Force
            Write-Host "  [cache] Copied cached shadow CSV to rollback_outputs" -ForegroundColor Cyan
        }
        
        # Handle CSV-only extraction (RefreshAnalysis with existing LK ADTs)
        if ($needsCsvExtraction) {
            Write-Host "  [cache] Extracting CSVs from existing LK ADTs..." -ForegroundColor Cyan
            
            if (-not $AlphaRoot) {
                throw "AlphaRoot must be provided for CSV extraction"
            }
            
            if (-not (Test-Path $alphaToolProject)) {
                throw "AlphaWDTAnalysisTool project not found at $alphaToolProject"
            }
            
            # NEW: Use LK ADT extraction mode (no Alpha WDT needed!)
            # This reads LK ADTs directly and extracts CSVs
            $toolArgs = @(
                'run','--project',$alphaToolProject,'--configuration','Release','--',
                '--extract-lk-adts', $mapRoot,  # Point directly to LK ADT directory
                '--map', $Map,
                '--out', $versionRoot,
                '--extract-mcnk-terrain',
                '--extract-mcnk-shadows'
            )
            
            Write-Log "  [debug] Running CSV extraction from cached LK ADTs (direct mode)" -Color Magenta
            Write-Log "  [debug] LK ADT directory: $mapRoot" -Color DarkGray
            $csvExtractLog = Join-Path $logRootDir "awdt_csv_${Version}_${Map}.log"
            Write-Log "  [debug] AlphaWDTAnalyzer output: $csvExtractLog" -Color DarkGray
            & dotnet @toolArgs 2>&1 | Tee-Object -FilePath $csvExtractLog | ForEach-Object {
                Write-Log "    [awdt] $_" -Color Gray
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Log "  [ERROR] AlphaWdtAnalyzer LK extraction failed! See: $csvExtractLog" -Color Red
                throw "AlphaWdtAnalyzer LK extraction failed for $Version/$Map"
            }
            
            # Copy newly extracted CSVs to rollback_outputs
            $extractedTerrainCsv = Join-Path $versionRoot "csv\$Map\${Map}_mcnk_terrain.csv"
            if (Test-Path $extractedTerrainCsv) {
                Copy-Item -Path $extractedTerrainCsv -Destination $rollbackMapCsvDir -Force
                Write-Host "  [cache] ✓ Copied extracted terrain CSV to rollback_outputs" -ForegroundColor Green
            }
            
            $extractedShadowCsv = Join-Path $versionRoot "csv\$Map\${Map}_mcnk_shadows.csv"
            if (Test-Path $extractedShadowCsv) {
                Copy-Item -Path $extractedShadowCsv -Destination $rollbackMapCsvDir -Force
                Write-Host "  [cache] ✓ Copied extracted shadow CSV to rollback_outputs" -ForegroundColor Green
            }
        }
        
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
        '--out', $tempExportDir,
        '--export-adt',
        '--export-dir', $tempExportDir,
        '--extract-mcnk-terrain',
        '--extract-mcnk-shadows',
        '--no-web',
        '--profile', 'raw',
        '--no-fallbacks'
    )
    
    # Add DBCTool AreaTable crosswalk parameters if available
    if ($dbctoolPatchDir -and (Test-Path $dbctoolPatchDir)) {
        $toolArgs += '--dbctool-patch-dir'
        $toolArgs += $dbctoolPatchDir
        Write-Host "  [debug] ✓ Using DBCTool AreaTable crosswalks: $dbctoolPatchDir" -ForegroundColor Green
    } else {
        Write-Host "  [warn] ✗ DBCTool AreaTable crosswalks NOT found - AreaIDs will NOT be mapped!" -ForegroundColor Red
        Write-Host "  [warn] Run DBCTool.V2 first to generate AreaTable mappings" -ForegroundColor Yellow
    }
    
    if ($dbctoolLkDir -and (Test-Path $dbctoolLkDir)) {
        $toolArgs += '--dbctool-lk-dir'
        $toolArgs += $dbctoolLkDir
        Write-Host "  [debug] ✓ Using LK DBC directory: $dbctoolLkDir" -ForegroundColor Green
    } else {
        Write-Host "  [warn] ✗ LK DBC directory NOT found: $dbctoolLkDir" -ForegroundColor Yellow
    }

    Write-Log "  [debug] Running AlphaWdtAnalyzer with AreaTable crosswalk support" -Color Magenta
    Write-Log "  [debug] Analysis output dir: $analysisDir" -Color Magenta
    $awdtLog = Join-Path $logRootDir "awdt_full_${Version}_${Map}.log"
    Write-Log "  [debug] AlphaWDTAnalyzer full output: $awdtLog" -Color DarkGray
    & dotnet @toolArgs 2>&1 | Tee-Object -FilePath $awdtLog | ForEach-Object {
        Write-Log "    [awdt] $_" -Color Gray
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "  [ERROR] AlphaWdtAnalyzer conversion failed! See: $awdtLog" -Color Red
        throw "AlphaWdtAnalyzer conversion failed for $Version/$Map"
    }
    
    # Verify terrain CSV was created
   $expectedTerrainCsv = Join-Path $tempExportDir "csv\$Map\${Map}_mcnk_terrain.csv"
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
   
    # Copy terrain CSVs to rollback_outputs for viewer generation
    $rollbackVersionRoot = Join-Path $PSScriptRoot ('rollback_outputs\' + $Version)
    $rollbackMapCsvDir = Join-Path $rollbackVersionRoot ('csv\' + $Map)
    if (-not (Test-Path $rollbackMapCsvDir)) {
        New-Item -Path $rollbackMapCsvDir -ItemType Directory -Force | Out-Null
    }
    
    # Copy MCNK terrain CSV (AlphaWdtAnalyzer outputs to csv/MapName/MapName_mcnk_terrain.csv)
    $terrainCsvDir = Join-Path $tempExportDir "csv\$Map"
    $terrainCsv = Join-Path $terrainCsvDir "${Map}_mcnk_terrain.csv"
    if (Test-Path $terrainCsv) {
        Copy-Item -Path $terrainCsv -Destination $rollbackMapCsvDir -Force
        Write-Host "  [cache] ✓ Copied terrain CSV to rollback_outputs" -ForegroundColor Green
    } else {
        Write-Host "  [warn] Terrain CSV not found at: $terrainCsv" -ForegroundColor Yellow
    }
    
    # Copy MCNK shadow CSV
    $shadowCsv = Join-Path $terrainCsvDir ($Map + '_mcnk_shadows.csv')
    if (Test-Path $shadowCsv) {
        Copy-Item -Path $shadowCsv -Destination $rollbackMapCsvDir -Force
        Write-Host "  [cache] ✓ Copied shadow CSV to rollback_outputs" -ForegroundColor Green
    } else {
        Write-Host "  [debug] Shadow CSV not found (map may not have shadow data)" -ForegroundColor DarkGray
    }
    Remove-Item $tempExportDir -Recurse -Force
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

# Handle RefreshOverlays: delete old overlays BEFORE regenerating
if ($RefreshOverlays.IsPresent) {
    Write-Host "[refresh] Deleting old viewer overlays before regeneration..." -ForegroundColor Yellow
    $existingComparisons = Get-ChildItem -Path "rollback_outputs\comparisons" -Directory -ErrorAction SilentlyContinue
    foreach ($comp in $existingComparisons) {
        $overlayDir = Join-Path $comp.FullName "viewer\overlays"
        if (Test-Path $overlayDir) {
            Remove-Item $overlayDir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  [refresh] Deleted overlays in $($comp.Name)" -ForegroundColor Gray
        }
    }
    Write-Host "[refresh] ✓ Old overlays deleted, will regenerate fresh" -ForegroundColor Green
    Write-Host ""
}

Set-LogStage -StageName "comparison_generation" -StageNumber 3
Write-Log "[3/5] Regenerating comparison data (this may take several minutes)..." -Color Yellow
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

# Step 4: Locate the exact comparison directory from CLI output; fallback to latest with index.json
Set-LogStage -StageName "viewer_setup" -StageNumber 4
Write-Log "[4/5] Locating viewer output..." -Color Yellow

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
# Phase 1: Feature flag to use WoWRollback.Viewer project assets
if ($UseNewViewerAssets) {
    $assetsSrc = Join-Path $PSScriptRoot 'WoWRollback.Viewer\bin\Release\net9.0\assets'
    Write-Host "[PHASE 1] Using WoWRollback.Viewer assets from: $assetsSrc" -ForegroundColor Magenta
} else {
    $assetsSrc = Join-Path $PSScriptRoot 'ViewerAssets'
}

if (Test-Path $assetsSrc) {
    Copy-Item -Path (Join-Path $assetsSrc '*') -Destination $viewerDir -Recurse -Force
    Write-Host "✓ Copied viewer assets from: $assetsSrc" -ForegroundColor Gray
} else {
    Write-Host "Warning: Viewer assets directory not found at $assetsSrc" -ForegroundColor Yellow
    if ($UseNewViewerAssets) {
        Write-Host "Hint: Make sure WoWRollback.Viewer project has been built." -ForegroundColor Yellow
    }
}

# Copy CSV data for sedimentary layers (UniqueID filtering)
Write-Host "Copying CSV data for UniqueID filtering..." -ForegroundColor Cyan
foreach ($ver in $Versions) {
    foreach ($mapName in $Maps) {
        $csvSource = Join-Path $PSScriptRoot "cached_maps\analysis\$ver\$mapName\csv"
        if (Test-Path $csvSource) {
            $csvDest = Join-Path $viewerDir "cached_maps\analysis\$ver\$mapName\csv"
            if (-not (Test-Path $csvDest)) {
                New-Item -Path $csvDest -ItemType Directory -Force | Out-Null
            }
            Copy-Item -Path (Join-Path $csvSource "id_ranges_by_map.csv") -Destination $csvDest -Force -ErrorAction SilentlyContinue
            Copy-Item -Path (Join-Path $csvSource "unique_ids_all.csv") -Destination $csvDest -Force -ErrorAction SilentlyContinue
            Write-Host "  ✓ Copied CSVs for $ver/$mapName" -ForegroundColor DarkGray
        }
    }
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
Write-Host "Viewer output: $viewerDir" -ForegroundColor Cyan
Write-Host "" 

# Step 5: Start server
Set-LogStage -StageName "server" -StageNumber 5
Write-Log "[5/5] Server options..." -Color Yellow
Write-Host "To start the viewer:" -ForegroundColor White
Write-Host "  cd \"$viewerDir\"" -ForegroundColor Yellow
Write-Host "  python -m http.server 8080" -ForegroundColor Yellow
Write-Host "" 
Write-Host "Then open: http://localhost:8080/index.html" -ForegroundColor Green
Write-Host "" 

# === FINAL SUMMARY ===
$summaryFile = Join-Path $logRootDir "_SUMMARY.txt"
$endTime = Get-Date
$duration = $endTime - (Get-Date $timestamp)

$summaryContent = @"
===============================================
WoWRollback Rebuild & Regenerate - SUMMARY
===============================================
Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Duration: $($duration.ToString())
Log Directory: $logRootDir

Maps Processed: $($Maps -join ', ')
Versions: $($Versions -join ', ')
Refresh Flags: $(if ($RefreshCache) {'RefreshCache '})$(if ($RefreshAnalysis) {'RefreshAnalysis '})$(if ($RefreshOverlays) {'RefreshOverlays'})

Viewer Output: $viewerDir
Overlay Files Generated: $overlayCount

Key Logs:
  - Build: $(Join-Path $logRootDir '01_build.log')
  - Cache Prep: $(Join-Path $logRootDir '02_cache_preparation.log')
  - Comparison: $(Join-Path $logRootDir '03_comparison_generation.log')
  - Viewer Setup: $(Join-Path $logRootDir '04_viewer_setup.log')
  - AlphaWDT Logs: $(Join-Path $logRootDir 'awdt_*.log')

To view logs:
  cd "$logRootDir"
  Get-ChildItem *.log | Sort-Object Name

===============================================
"@

Set-Content -Path $summaryFile -Value $summaryContent -Encoding UTF8
Write-Log "" 
Write-Log "=== PIPELINE COMPLETE ===" -Color Green
Write-Log "Duration: $($duration.ToString())" -Color Cyan
Write-Log "Overlay files: $overlayCount" -Color Cyan
Write-Log "Logs saved to: $logRootDir" -Color Cyan
Write-Log "Summary: $summaryFile" -Color Cyan
Write-Log ""

if ($Serve) {
    Write-Log "[5/5] Starting server on http://localhost:8080..." -Color Cyan
    Write-Log "Press Ctrl+C to stop the server" -Color Gray
    Write-Log "" 
    Set-Location $viewerDir
    python -m http.server 8080
} else {
    # Offer to start server interactively
    $response = Read-Host "Start Python HTTP server now? (Y/n)"
    if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
        Write-Log "Starting server on http://localhost:8080..." -Color Cyan
        Write-Log "Press Ctrl+C to stop the server" -Color Gray
        Write-Log "" 
        Set-Location $viewerDir
        python -m http.server 8080
    }
}
