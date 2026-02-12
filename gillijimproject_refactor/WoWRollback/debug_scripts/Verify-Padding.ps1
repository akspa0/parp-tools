#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Verifies Alpha WDT padding fields are zeroed per spec.
.DESCRIPTION
  Checks:
   - MPHD pad[112]
   - MAIN entry pad[4] and flags (expect 0)
   - MHDR pad[36]
   - First tile MCNK:
       - MCNR trailing pad[13]
       - MCLY per-layer pad[2]
.PARAMETER WdtPath
  Path to Alpha WDT
#>
param(
  [Parameter(Mandatory=$true)] [string]$WdtPath
)
if (-not (Test-Path $WdtPath)) { throw "File not found: $WdtPath" }
$bytes = [System.IO.File]::ReadAllBytes($WdtPath)
Write-Host "=== Verify Padding ===" -ForegroundColor Cyan
Write-Host "File: $WdtPath  Size: $($bytes.Length) bytes"

function AllZero([byte[]]$arr) { foreach ($b in $arr) { if ($b -ne 0) { return $false } } return $true }

# Locate top-level chunks quickly
function FindChunk([string]$fcc) {
  for ($i=0; $i + 8 -le $bytes.Length;) {
    $four = [System.Text.Encoding]::ASCII.GetString($bytes, $i, 4)
    $size = [BitConverter]::ToInt32($bytes, $i+4)
    if ($four -eq $fcc) { return @{ Offset=$i; Size=$size } }
    $next = $i + 8 + $size + ((($size -band 1) -eq 1) ? 1 : 0)
    if ($next -le $i) { break }
    $i = $next
  }
  return $null
}

# MPHD
$mphd = FindChunk 'DHPM'
if ($mphd) {
  $pad = $bytes[($mphd.Offset+8+16)..($mphd.Offset+8+127)]
  $ok = AllZero $pad
  Write-Host "MPHD pad[112]: $($ok ? 'OK' : 'NONZERO')"
}

# MAIN
$main = FindChunk 'NIAM'
if ($main) {
  $dataStart = $main.Offset + 8
  $nonZero = 0
  for ($i=0; $i -lt 4096; $i++) {
    $pos = $dataStart + $i*16
    if ($pos + 16 -gt $bytes.Length) { break }
    $flags = [BitConverter]::ToInt32($bytes, $pos+8)
    $pad4 = $bytes[($pos+12)..($pos+15)]
    if ($flags -ne 0) { $nonZero++ }
    if (-not (AllZero $pad4)) { $nonZero++ }
  }
  Write-Host "MAIN flags/pad issues: $nonZero (0 expected)"
}

# First populated tile MHDR
# Scan MAIN for first non-zero offset
$firstTile = 0
$firstMhdr = 0
if ($main) {
  $dataStart = $main.Offset + 8
  for ($i=0; $i -lt 4096; $i++) {
    $pos = $dataStart + $i*16
    $off = [BitConverter]::ToInt32($bytes, $pos)
    if ($off -gt 0) { $firstTile = $i; $firstMhdr = $off; break }
  }
}
if ($firstMhdr -gt 0) {
  $mhdrDataStart = $firstMhdr + 8
  $mhdrPad = $bytes[($mhdrDataStart+28)..($mhdrDataStart+63)]
  $ok = AllZero $mhdrPad
  Write-Host "MHDR pad[36]: $($ok ? 'OK' : 'NONZERO')"

  # Resolve first MCNK and parse header
  $offsInfo = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 0)
  $offsTex  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 4)
  $sizeTex  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 8)
  $offsDoo  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 12)
  $sizeDoo  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 16)
  $offsMob  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 20)
  $sizeMob  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 24)

  $mcINEnd = $mhdrDataStart + $offsInfo + (8 + 4096)
  $mtEXEnd = $mhdrDataStart + $offsTex  + (8 + $sizeTex)
  $mdDFEnd = $mhdrDataStart + $offsDoo  + (8 + $sizeDoo)
  $moDFEnd = $mhdrDataStart + $offsMob  + (8 + $sizeMob)
  $firstMcnk = ($mcINEnd, $mtEXEnd, $mdDFEnd, $moDFEnd | Measure-Object -Maximum).Maximum

  if ($firstMcnk + 8 + 128 -le $bytes.Length) {
    $mcnkDataStart = $firstMcnk + 8
    # MCNK header fields (Alpha): offsNormal at +28; offsLayer at +32; nLayers at +16
    $nLayers   = [BitConverter]::ToInt32($bytes, $mcnkDataStart + 16)
    $offsNorm  = [BitConverter]::ToInt32($bytes, $mcnkDataStart + 28)
    $offsLayer = [BitConverter]::ToInt32($bytes, $mcnkDataStart + 32)
    $baseAfterHeader = $mcnkDataStart + 128

    # MCNR pad[13]
    $mcnrPos = $baseAfterHeader + $offsNorm
    $mcnrEndData = $mcnrPos + (145*3)
    if ($mcnrEndData + 13 -le $bytes.Length) {
      $pad13 = $bytes[$mcnrEndData..($mcnrEndData+12)]
      Write-Host "MCNR pad[13]: $((AllZero $pad13) ? 'OK' : 'NONZERO')"
    } else {
      Write-Host "MCNR pad[13]: OUT-OF-RANGE"
    }

    # MCLY per-layer pad[2]
    $mclyPos = $baseAfterHeader + $offsLayer
    $badLayers = 0
    for ($l=0; $l -lt $nLayers; $l++) {
      $layerPos = $mclyPos + ($l * 16)
      if ($layerPos + 16 -gt $bytes.Length) { break }
      $pad2 = $bytes[($layerPos+14)..($layerPos+15)]
      if (-not (AllZero $pad2)) { $badLayers++ }
    }
    Write-Host "MCLY pad[2] nonzero layers: $badLayers (0 expected)"
  }
}

Write-Host "=== Verify Complete ===" -ForegroundColor Cyan
