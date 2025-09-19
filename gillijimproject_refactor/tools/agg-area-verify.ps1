param(
  [Parameter(Mandatory=$true)][string]$ExportRoot,
  [string]$Map
)

$ErrorActionPreference = 'Stop'

$mapsDir = Join-Path $ExportRoot "csv/maps"
if (!(Test-Path $mapsDir)) { throw "Not found: $mapsDir (run the exporter with --verbose to generate verify CSVs)" }

$files = Get-ChildItem -Path $mapsDir -Recurse -Filter "areaid_verify_*.csv" -File
if ($Map) {
  $files = $files | Where-Object { $_.Directory.Name -ieq $Map }
}
if (-not $files) { Write-Host "No verify CSVs found under $mapsDir"; exit 0 }

$agg = @{}
foreach ($f in $files) {
  $mapName = $f.Directory.Name
  try {
    $rows = Import-Csv $f
  } catch {
    Write-Warning "Failed to read $($f.FullName): $_"; continue
  }
  foreach ($r in $rows) {
    if (-not $r.alpha_raw) { continue }
    $alpha = [int]$r.alpha_raw
    if ($alpha -lt 0) { continue } # skip non-present chunks
    $lk = 0
    if ($r.lk_areaid) { [int]::TryParse($r.lk_areaid, [ref]$lk) | Out-Null }
    $key = "$mapName|$alpha"
    if (-not $agg.ContainsKey($key)) {
      $agg[$key] = [PSCustomObject]@{
        map = $mapName
        alpha_raw = $alpha
        alpha_raw_hex = ('0x{0:X8}' -f $alpha)
        area_hi16 = (($alpha -band 0xFFFF0000) -shr 16)
        area_lo16 = ($alpha -band 0x0000FFFF)
        count_chunks = 0
        mapped_any = 0
        sample_lk_areaid = 0
        sample_reason = ''
      }
    }
    $e = $agg[$key]
    $e.count_chunks++
    if ($lk -gt 0) {
      $e.mapped_any = 1
      if ($e.sample_lk_areaid -le 0) { $e.sample_lk_areaid = $lk }
    }
    if (-not $e.sample_reason -and $r.reason) { $e.sample_reason = [string]$r.reason }
  }
}

$outDir = Join-Path $mapsDir "_summary"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$all = $agg.Values | Sort-Object map, alpha_raw
$all | Export-Csv -NoTypeInformation -Path (Join-Path $outDir "alpha_areas_used.csv")
$unknown = $all | Where-Object { $_.mapped_any -eq 0 }
$unknown | Export-Csv -NoTypeInformation -Path (Join-Path $outDir "unknown_area_numbers.csv")

# Also split per map for convenience
$maps = $all | Select-Object -ExpandProperty map -Unique
foreach ($m in $maps) {
  $per = $all | Where-Object { $_.map -ieq $m }
  $per | Export-Csv -NoTypeInformation -Path (Join-Path $outDir ("alpha_areas_used_" + $m + ".csv"))
  ($per | Where-Object { $_.mapped_any -eq 0 }) | Export-Csv -NoTypeInformation -Path (Join-Path $outDir ("unknown_area_numbers_" + $m + ".csv"))
}

Write-Host "Wrote: $outDir\alpha_areas_used.csv"
Write-Host "Wrote: $outDir\unknown_area_numbers.csv"
