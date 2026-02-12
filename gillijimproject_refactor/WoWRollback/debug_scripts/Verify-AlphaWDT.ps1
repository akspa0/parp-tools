#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Verifies Alpha WDT file structure and chunk integrity
.DESCRIPTION
    Checks MPHD, MHDR, MCNK headers and offsets in an Alpha 0.5.3 WDT file
.PARAMETER WdtPath
    Path to the Alpha WDT file to verify
.EXAMPLE
    .\Verify-AlphaWDT.ps1 -WdtPath "project_output\Kalidar_20251016_012513\Kalidar.wdt"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$WdtPath
)

if (-not (Test-Path $WdtPath)) {
    Write-Error "WDT file not found: $WdtPath"
    exit 1
}

$bytes = [System.IO.File]::ReadAllBytes($WdtPath)
Write-Host "=== Alpha WDT Verification ===" -ForegroundColor Cyan
Write-Host "File: $WdtPath"
Write-Host "Size: $([math]::Round($bytes.Length/1MB, 2)) MB"
Write-Host ""

# Check top-level chunks
Write-Host "Top-Level Chunks:" -ForegroundColor Yellow
$pos = 0
for ($i = 0; $i -lt 10; $i++) {
    if ($pos + 8 -gt $bytes.Length) { break }
    $fourcc = [System.Text.Encoding]::ASCII.GetString($bytes[$pos..($pos+3)])
    $size = [BitConverter]::ToInt32($bytes, $pos + 4)
    Write-Host "  [$i] 0x$($pos.ToString('X')): '$fourcc' size=$size"
    $pos += 8 + $size
    if (($size % 2) -eq 1) { $pos++ }
}

# Check MPHD
Write-Host "`nMPHD Structure:" -ForegroundColor Yellow
$mphdPos = 0xC
$mphdData = $bytes[($mphdPos+8)..($mphdPos+8+15)]
$nDoodad = [BitConverter]::ToInt32($mphdData, 0)
$offsDoodad = [BitConverter]::ToInt32($mphdData, 4)
$nMapObj = [BitConverter]::ToInt32($mphdData, 8)
$offsMapObj = [BitConverter]::ToInt32($mphdData, 12)
Write-Host "  nDoodadNames: $nDoodad"
Write-Host "  offsDoodadNames: 0x$($offsDoodad.ToString('X'))"
Write-Host "  nMapObjNames: $nMapObj"
Write-Host "  offsMapObjNames: 0x$($offsMapObj.ToString('X'))"

if ($nDoodad -eq 0 -and $offsDoodad -ne 0) {
    Write-Host "  ⚠ WARNING: count=0 but offset!=0" -ForegroundColor Red
}
if ($nMapObj -eq 0 -and $offsMapObj -ne 0) {
    Write-Host "  ⚠ WARNING: count=0 but offset!=0" -ForegroundColor Red
}

# Check first tile MHDR
Write-Host "`nFirst Tile MHDR:" -ForegroundColor Yellow
$mainPos = 0x94 + 8
$firstTileOffset = 0
for ($i = 0; $i -lt 4096; $i++) {
    $entryPos = $mainPos + ($i * 16)
    if ($entryPos + 4 -gt $bytes.Length) {
        Write-Host "  ⚠ MAIN extends past file end" -ForegroundColor Red
        break
    }
    $offset = [BitConverter]::ToInt32($bytes, $entryPos)
    if ($offset -gt 0) {
        $firstTileOffset = $offset
        Write-Host "  First tile: index $i at 0x$($offset.ToString('X'))"
        break
    }
}

if ($firstTileOffset -gt 0) {
    $mhdrData = $bytes[($firstTileOffset+8)..($firstTileOffset+8+27)]
    $offsInfo = [BitConverter]::ToInt32($mhdrData, 0)
    $offsTex = [BitConverter]::ToInt32($mhdrData, 4)
    $sizeTex = [BitConverter]::ToInt32($mhdrData, 8)
    $offsDoo = [BitConverter]::ToInt32($mhdrData, 12)
    $sizeDoo = [BitConverter]::ToInt32($mhdrData, 16)
    $offsMob = [BitConverter]::ToInt32($mhdrData, 20)
    $sizeMob = [BitConverter]::ToInt32($mhdrData, 24)
    
    $mhdrDataStart = $firstTileOffset + 8
    Write-Host "  offsInfo: $offsInfo -> 0x$(($mhdrDataStart + $offsInfo).ToString('X'))"
    Write-Host "  offsTex: $offsTex -> 0x$(($mhdrDataStart + $offsTex).ToString('X'))"
    Write-Host "  sizeTex: $sizeTex"
    Write-Host "  offsDoo: $offsDoo -> 0x$(($mhdrDataStart + $offsDoo).ToString('X'))"
    Write-Host "  sizeDoo: $sizeDoo"
    Write-Host "  offsMob: $offsMob -> 0x$(($mhdrDataStart + $offsMob).ToString('X'))"
    Write-Host "  sizeMob: $sizeMob"
    
    # Verify chunks at offsets
    $mcinFourCC = [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart + $offsInfo)..($mhdrDataStart + $offsInfo + 3)])
    $mtexFourCC = [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart + $offsTex)..($mhdrDataStart + $offsTex + 3)])
    
    Write-Host "`n  Verification:"
    if ($mcinFourCC -eq "NICM") {
        Write-Host "    ✓ MCIN found at correct offset" -ForegroundColor Green
    } else {
        Write-Host "    ✗ MCIN not found (got '$mcinFourCC')" -ForegroundColor Red
    }
    
    if ($mtexFourCC -eq "XETM") {
        Write-Host "    ✓ MTEX found at correct offset" -ForegroundColor Green
    } else {
        Write-Host "    ✗ MTEX not found (got '$mtexFourCC')" -ForegroundColor Red
    }
}

Write-Host "`n=== Verification Complete ===" -ForegroundColor Cyan
