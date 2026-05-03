param(
    [string]$OutputDir = "output/ml-training/v11_cache",
    [string]$ScanDir = "output/tmp/v11_scan",
    [switch]$DryRun
)

$RepoRoot = "I:\parp\parp-tools"
$Converter = Join-Path $RepoRoot "wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj"
$ScanDir = Join-Path $RepoRoot $ScanDir
$OutputDir = Join-Path $RepoRoot $OutputDir

New-Item -ItemType Directory -Force -Path $ScanDir | Out-Null

$clients = @(
    @{ Label="3_3_5_12340"; Root="output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft"; Maps=@("Azeroth","Kalimdor","Northrend") }
    @{ Label="4_0_0_11927"; Root="output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft"; Maps=@("Azeroth","Kalimdor","Deepholm","LostIsles") }
    @{ Label="3_0_1_8303";  Root="output/tmp/wowarchive-clients/3_0_1_8303/World of Warcraft"; Maps=@("Northrend") }
    @{ Label="0_7_0_3694";  Root="output/tmp/wowarchive-clients/0_7_0_3694/World of Warcraft"; Maps=@("Azeroth","Kalimdor") }
)

$allScans = @()
foreach ($client in $clients) {
    $clientRoot = Join-Path $RepoRoot $client.Root
    if (!(Test-Path $clientRoot)) {
        Write-Warning "Client root not found: $clientRoot"
        continue
    }

    foreach ($map in $client.Maps) {
        $scanPath = Join-Path $ScanDir "scan_$($client.Label)_$map.json"
        Write-Host "`n=== Scanning $($client.Label) / $map ===" -ForegroundColor Cyan
        if (!$DryRun) {
            & dotnet run --project $Converter -- dataset-scan `
                --client-root $clientRoot --map $map --build $($client.Label) `
                --output $scanPath 2>&1 | ForEach-Object { Write-Host $_ }
        }
        if (Test-Path $scanPath) { $allScans += $scanPath }
    }
}

if ($allScans.Count -eq 0) { Write-Error "No scans produced"; exit 1 }

$mergedPath = Join-Path $ScanDir "merged.json"
Write-Host "`n=== Merging $($allScans.Count) scans ===" -ForegroundColor Cyan
if (!$DryRun) {
    $mergeArgs = @("run", "--project", $Converter, "--", "dataset-merge", "--output", $mergedPath)
    foreach ($s in $allScans) { $mergeArgs += @("--input", $s) }
    & dotnet $mergeArgs 2>&1 | ForEach-Object { Write-Host $_ }
}

$curatedPath = Join-Path $ScanDir "curated.json"
Write-Host "`n=== Curating ===" -ForegroundColor Cyan
if (!$DryRun) {
    & dotnet run --project $Converter -- dataset-curate `
        --input $mergedPath --output $curatedPath --report (Join-Path $ScanDir "curation_report.json") `
        --no-require-minimap --no-require-wdl 2>&1 | ForEach-Object { Write-Host $_ }
}

Write-Host "`n=== Building Cache ===" -ForegroundColor Cyan
if (!$DryRun) {
    & dotnet run --project $Converter -- dataset-build-cache `
        --input $curatedPath --output-dir $OutputDir --overwrite 2>&1 | ForEach-Object { Write-Host $_ }
}

Write-Host "`nDone! Cache: $OutputDir" -ForegroundColor Green
Write-Host "Scans: $ScanDir"
