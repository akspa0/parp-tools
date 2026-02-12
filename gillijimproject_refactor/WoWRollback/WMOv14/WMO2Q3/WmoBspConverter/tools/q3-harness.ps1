param(
  [Parameter(Mandatory=$true)][string]$Quake3Exe,
  [Parameter(Mandatory=$true)][string]$BspPath,
  [string]$HomePath,
  [int]$TimeoutSec = 20
)
if (-not $HomePath -or $HomePath -eq '') { $HomePath = Join-Path $PSScriptRoot "..\test_output\q3_harness_home" }
$HomePath = [IO.Path]::GetFullPath($HomePath)
$mapName = [IO.Path]::GetFileNameWithoutExtension($BspPath)
$baseq3 = Join-Path $HomePath "baseq3"
$mapsDir = Join-Path $baseq3 "maps"
New-Item -ItemType Directory -Path $mapsDir -Force | Out-Null
$destBsp = Join-Path $mapsDir ($mapName + ".bsp")
Copy-Item -LiteralPath $BspPath -Destination $destBsp -Force
$logFile = Join-Path $baseq3 "qconsole.log"
if (Test-Path $logFile) { Remove-Item $logFile -Force }
$argList = @(
  '+set','dedicated','1',
  '+set','fs_homepath',$HomePath,
  '+set','fs_game','baseq3',
  '+set','sv_pure','0',
  '+set','developer','1',
  '+set','logfile','2',
  '+map',$mapName
)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Quake3Exe
foreach ($a in $argList) { [void]$psi.ArgumentList.Add($a) }
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$p = [System.Diagnostics.Process]::Start($psi)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
  Start-Sleep -Milliseconds 500
  if ($p.HasExited) { break }
}
if (-not $p.HasExited) { try { $p.Kill() } catch {} }
Start-Sleep -Seconds 1
$result = @{
  Map = $mapName
  HomePath = $HomePath
  LogPath = $logFile
  Passed = $false
  Errors = @()
}
if (Test-Path $logFile) {
  $log = Get-Content -Path $logFile -Raw
  $hasSpawn = ($log -match "SV_SpawnServer\s+$mapName") -or ($log -match "CM_LoadMap\(.+maps/$mapName\.bsp")
  $hasFatal = ($log -match "ERROR" -or $log -match "FATAL" -or $log -match "Couldn't load maps/$mapName\.bsp" -or $log -match "BSP different version" -or $log -match "Invalid .+bsp")
  if ($hasSpawn -and -not $hasFatal) { $result.Passed = $true }
  if ($hasFatal) {
    $lines = $log -split "`r?`n"
    $errs = $lines | Where-Object { $_ -match "ERROR|FATAL|Invalid|BSP different version|Couldn''t load maps" } | Select-Object -First 10
    $result.Errors = $errs
  }
} else {
  $result.Errors = @("No log produced")
}
$code = if ($result.Passed) {0} else {1}
$result | ConvertTo-Json -Depth 4
exit $code
