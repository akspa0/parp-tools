param(
    [string]$MapDir = "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368_dev\World\maps\development",
    [string]$ArchiveRoot = "I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft",
    [string]$OutputDir = "I:\parp\parp-tools\wow-viewer\test_data\correlation",
    [string]$ProjectDir = "I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect",
    [string]$MapPrefix = "development",
    [switch]$Force
)

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Find all PM4 files > 100 bytes (filters placeholders)
$pm4Files = Get-ChildItem -LiteralPath $MapDir -Filter "*.pm4" | Where-Object Length -gt 100

$tilesToProcess = @()
foreach ($pm4 in $pm4Files) {
    $base = $pm4.BaseName
    $parts = $base.Split("_")
    if ($parts.Length -lt 3) { continue }
    $x = [int]$parts[1]
    $y = [int]$parts[2]
    $adtName = "${MapPrefix}_${x}_${y}_obj0.adt"
    $adtPath = Join-Path -Path $MapDir -ChildPath $adtName
    if (Test-Path -LiteralPath $adtPath) {
        $tilesToProcess += [PSCustomObject]@{
            PM4Path = $pm4.FullName
            ADTPath = $adtPath
            PM4Name = $pm4.Name
            TileX = $x
            TileY = $y
            PM4SizeKB = [Math]::Round($pm4.Length / 1KB)
        }
    }
}

Write-Host "Found $($tilesToProcess.Count) tiles with PM4 + _obj0.adt"
$tilesToProcess = $tilesToProcess | Sort-Object PM4SizeKB -Descending

# Filter already-processed
$allTiles = Get-ChildItem -LiteralPath $OutputDir -Filter "correlate_*.json" | ForEach-Object { $_.BaseName.Replace("correlate_", "") }
$remaining = if ($Force) { $tilesToProcess } else { $tilesToProcess | Where-Object { "$($_.TileX)_$($_.TileY)" -notin $allTiles } }
Write-Host "Remaining: $($remaining.Count)"

$doneCount = ($tilesToProcess.Count) - $remaining.Count
$totalCount = $tilesToProcess.Count

foreach ($tile in $remaining) {
    $doneCount++
    $outPath = Join-Path -Path $OutputDir -ChildPath "correlate_$($tile.TileX)_$($tile.TileY).json"
    $start = Get-Date
    Write-Host "[$doneCount/$totalCount] Tile ($($tile.TileX),$($tile.TileY)) size=$($tile.PM4SizeKB)KB..." -NoNewline
    
    $output = & "dotnet" run --project $ProjectDir --no-build -c Debug -- pm4 correlate-models `
        --input $tile.PM4Path `
        --placements $tile.ADTPath `
        --archive-root $ArchiveRoot `
        --output $outPath 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  $output" -ForegroundColor DarkRed
    } else {
        $elapsed = [Math]::Round(((Get-Date) - $start).TotalSeconds)
        Write-Host " ${elapsed}s" -ForegroundColor Green
    }
}

Write-Host "`nAll tiles processed. Aggregating results..."
& dotnet run --project $ProjectDir --no-build -c Debug -- pm4 correlate-sweep `
    --map-dir $MapDir `
    --archive-root $ArchiveRoot `
    --corpus-dir $OutputDir `
    --output (Join-Path -Path $OutputDir -ChildPath "sweep_report.json") 2>&1
