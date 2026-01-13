#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deep comparison of two Alpha WDT files with structural audits
.DESCRIPTION
    - Top-level chunks table and size deltas
    - MPHD validation (counts/offsets)
    - MAIN grid scan (populated tiles)
    - Name tables (MDNM/MONM) stats and sample names
    - Placements (MDDF/MODF) totals
    - Per-tile MHDR verification and MCNK subchunk audit (first N tiles)
.PARAMETER OriginalPath
    Path to the original/reference Alpha WDT file
.PARAMETER GeneratedPath
    Path to the generated Alpha WDT file to compare
.PARAMETER Tiles
    Number of populated tiles to analyze in depth (default 8)
.PARAMETER SampleNames
    Number of names to sample from MDNM/MONM (default 10)
.PARAMETER Json
    Optional path to write a JSON summary
.EXAMPLE
    .\Compare-AlphaWDTs.ps1 -OriginalPath "...orig.wdt" -GeneratedPath "...gen.wdt" -Tiles 8 -SampleNames 10
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$OriginalPath,
    
    [Parameter(Mandatory=$true)]
    [string]$GeneratedPath,

    [int]$Tiles = 8,
    [int]$SampleNames = 10,
    [string]$Json
)

function Read-BytesStrict([string]$path) {
    if (-not (Test-Path $path)) { throw "File not found: $path" }
    return [System.IO.File]::ReadAllBytes($path)
}

function Get-TopLevelChunks([byte[]]$bytes) {
    $list = @()
    $pos = 0
    while ($pos + 8 -le $bytes.Length) {
        $fourcc = [System.Text.Encoding]::ASCII.GetString($bytes[$pos..($pos+3)])
        $size = [BitConverter]::ToInt32($bytes, $pos + 4)
        $list += [pscustomobject]@{ Offset=$pos; FourCC=$fourcc; Size=$size }
        $pos += 8 + $size
        if (($size % 2) -eq 1) { $pos++ }
        if ($pos -ge $bytes.Length) { break }
    }
    return $list
}

function Get-ChunkByFourCC([object[]]$chunks, [string]$fcc) {
    return $chunks | Where-Object { $_.FourCC -eq $fcc } | Select-Object -First 1
}

function Parse-MPHD([byte[]]$bytes, [int]$offset) {
    $dataPos = $offset + 8
    $nDoodad = [BitConverter]::ToInt32($bytes, $dataPos + 0)
    $offsDoodad = [BitConverter]::ToInt32($bytes, $dataPos + 4)
    $nMapObj = [BitConverter]::ToInt32($bytes, $dataPos + 8)
    $offsMapObj = [BitConverter]::ToInt32($bytes, $dataPos + 12)
    [pscustomobject]@{
        nDoodadNames=$nDoodad; offsDoodadNames=$offsDoodad; nMapObjNames=$nMapObj; offsMapObjNames=$offsMapObj
    }
}

function Count-NullTerminatedStrings([byte[]]$data) {
    $count = 0; for ($i=0; $i -lt $data.Length; $i++) { if ($data[$i] -eq 0) { $count++ } }
    return $count
}

function Read-NullStrings([byte[]]$data, [int]$max) {
    $names = @(); $cur = New-Object System.Collections.Generic.List[byte]
    for ($i=0; $i -lt $data.Length; $i++) {
        if ($data[$i] -ne 0) { $cur.Add($data[$i]) } else {
            $names += [System.Text.Encoding]::ASCII.GetString($cur.ToArray())
            $cur.Clear(); if ($names.Count -ge $max) { break }
        }
    }
    return $names
}

function Scan-MAIN([byte[]]$bytes, $mainChunk) {
    $dataPos = $mainChunk.Offset + 8
    $populated = @()
    for ($i=0; $i -lt 4096; $i++) {
        $entryPos = $dataPos + ($i * 16)
        if ($entryPos + 4 -gt $bytes.Length) { break }
        $tileOffs = [BitConverter]::ToInt32($bytes, $entryPos)
        if ($tileOffs -gt 0) { $populated += $tileOffs }
    }
    return $populated
}

function Analyze-Tiles([byte[]]$bytes, [int[]]$tileOffsets, [int]$limit) {
    $results = @()
    $count = [Math]::Min($limit, $tileOffsets.Count)
    for ($i=0; $i -lt $count; $i++) {
        $tileAbs = $tileOffsets[$i]
        $mhdrDataStart = $tileAbs + 8
        $offsInfo = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 0)
        $offsTex  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 4)
        $sizeTex  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 8)
        $offsDoo  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 12)
        $sizeDoo  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 16)
        $offsMob  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 20)
        $sizeMob  = [BitConverter]::ToInt32($bytes, $mhdrDataStart + 24)

        $mcinFCC = [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart+$offsInfo)..($mhdrDataStart+$offsInfo+3)])
        $mtexFCC = [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart+$offsTex)..($mhdrDataStart+$offsTex+3)])
        $mddfFCC = if ($sizeDoo -gt 0) { [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart+$offsDoo)..($mhdrDataStart+$offsDoo+3)]) } else { "" }
        $modfFCC = if ($sizeMob -gt 0) { [System.Text.Encoding]::ASCII.GetString($bytes[($mhdrDataStart+$offsMob)..($mhdrDataStart+$offsMob+3)]) } else { "" }

        # First MCNK start = start of MHDR.data + max(end of MCIN, MTEX, MDDF, MODF)
        $mcINEnd = $mhdrDataStart + $offsInfo + (8 + 4096)    # 8 header + 4096 data
        $mtEXEnd = $mhdrDataStart + $offsTex  + (8 + $sizeTex)
        $mdDFEnd = $mhdrDataStart + $offsDoo  + (8 + $sizeDoo)
        $moDFEnd = $mhdrDataStart + $offsMob  + (8 + $sizeMob)
        $firstMcnk = ($mcINEnd, $mtEXEnd, $mdDFEnd, $moDFEnd | Measure-Object -Maximum).Maximum

        $mcnkFCC = [System.Text.Encoding]::ASCII.GetString($bytes[$firstMcnk..($firstMcnk+3)])
        $mcnkSize = [BitConverter]::ToInt32($bytes, $firstMcnk + 4)

        $results += [pscustomobject]@{
            TileOffset=$tileAbs; MCIN=$mcinFCC; MTEX=$mtexFCC; MDDF=$mddfFCC; MODF=$modfFCC;
            SizeTex=$sizeTex; SizeDoo=$sizeDoo; SizeMob=$sizeMob; FirstMCNKFourCC=$mcnkFCC; FirstMCNKSize=$mcnkSize
        }
    }
    return $results
}

function Sum-TopLevelByFourCC([object[]]$chunks) {
    ($chunks | Group-Object FourCC | ForEach-Object { [pscustomobject]@{ FourCC=$_.Name; Total= ($_.Group | Measure-Object Size -Sum).Sum } })
}

$original = Read-BytesStrict $OriginalPath
$generated = Read-BytesStrict $GeneratedPath

Write-Host "=== Alpha WDT Comparison ===" -ForegroundColor Cyan
Write-Host "Original:  $OriginalPath ($([math]::Round($original.Length/1MB, 2)) MB)"
Write-Host "Generated: $GeneratedPath ($([math]::Round($generated.Length/1MB, 2)) MB)"
Write-Host ""

if ($original.Length -ne $generated.Length) {
    Write-Host "⚠ File sizes differ!" -ForegroundColor Yellow
    Write-Host "  Original:  $($original.Length) bytes"
    Write-Host "  Generated: $($generated.Length) bytes"
    Write-Host "  Difference: $($generated.Length - $original.Length) bytes"
    Write-Host ""
}

# Top-level chunks compare
Write-Host "Top-Level Chunk Comparison:" -ForegroundColor Yellow
$origChunks = Get-TopLevelChunks $original
$genChunks  = Get-TopLevelChunks $generated

$maxPairs = [Math]::Min($origChunks.Count, $genChunks.Count)
for ($i=0; $i -lt $maxPairs; $i++) {
    $o = $origChunks[$i]; $g = $genChunks[$i]
    $match = ($o.FourCC -eq $g.FourCC) -and ($o.Size -eq $g.Size)
    $status = if ($match) { "✓" } else { "✗" }
    $color = if ($match) { "Green" } else { "Red" }
    Write-Host "  [$i] 0x$($o.Offset.ToString('X')): $status Orig='$($o.FourCC)'($($o.Size)) Gen='$($g.FourCC)'($($g.Size))" -ForegroundColor $color
    if (-not $match) { Write-Host "    First difference in chunk $i" -ForegroundColor Red; break }
}

# Byte-level first difference
Write-Host "`nByte-Level Comparison:" -ForegroundColor Yellow
$maxLen = [Math]::Min($original.Length, $generated.Length)
$firstDiff = -1
for ($i = 0; $i -lt $maxLen; $i++) { if ($original[$i] -ne $generated[$i]) { $firstDiff = $i; break } }
if ($firstDiff -ge 0) {
    Write-Host "  First difference at offset: 0x$($firstDiff.ToString('X')) (byte $firstDiff)"
    $contextStart = [Math]::Max(0, $firstDiff - 16)
    $contextEnd = [Math]::Min($maxLen - 1, $firstDiff + 16)
    Write-Host "`n  Context (16 bytes before/after):"
    Write-Host "    Original:  $([BitConverter]::ToString($original[$contextStart..($contextEnd)]))"
    Write-Host "    Generated: $([BitConverter]::ToString($generated[$contextStart..($contextEnd)]))"
} else { if ($original.Length -eq $generated.Length) { Write-Host "  ✓ Files are identical!" -ForegroundColor Green } else { Write-Host "  Files match up to byte $maxLen, but lengths differ" -ForegroundColor Yellow } }

# MPHD audit
Write-Host "`nMPHD Audit:" -ForegroundColor Yellow
$origMPHD = Get-ChunkByFourCC $origChunks 'DHPM'
$genMPHD  = Get-ChunkByFourCC $genChunks  'DHPM'
if ($null -ne $origMPHD -and $null -ne $genMPHD) {
    $o = Parse-MPHD $original $origMPHD.Offset
    $g = Parse-MPHD $generated $genMPHD.Offset
    Write-Host "  Original: nDoodad=$($o.nDoodadNames) offsDoodad=0x$($o.offsDoodadNames.ToString('X')) nMapObj=$($o.nMapObjNames) offsMapObj=0x$($o.offsMapObjNames.ToString('X'))"
    Write-Host "  Generated: nDoodad=$($g.nDoodadNames) offsDoodad=0x$($g.offsDoodadNames.ToString('X')) nMapObj=$($g.nMapObjNames) offsMapObj=0x$($g.offsMapObjNames.ToString('X'))"
    if ($o.nDoodadNames -eq 0 -and $o.offsDoodadNames -ne 0) { Write-Host "  ⚠ Original: count=0 but offset!=0" -ForegroundColor Red }
    if ($g.nDoodadNames -eq 0 -and $g.offsDoodadNames -ne 0) { Write-Host "  ⚠ Generated: count=0 but offset!=0" -ForegroundColor Red }
}

# MAIN scan and populated tiles
Write-Host "`nMAIN Grid Scan:" -ForegroundColor Yellow
$origMAIN = Get-ChunkByFourCC $origChunks 'NIAM'
$genMAIN  = Get-ChunkByFourCC $genChunks  'NIAM'
$origTiles = @(); $genTiles=@()
if ($origMAIN) { $origTiles = Scan-MAIN $original $origMAIN }
if ($genMAIN)  { $genTiles  = Scan-MAIN $generated $genMAIN }
Write-Host "  Original populated tiles: $($origTiles.Count)"
Write-Host "  Generated populated tiles: $($genTiles.Count)"

# MDNM/MONM audit
Write-Host "`nName Tables (MDNM/MONM):" -ForegroundColor Yellow
$origMDNM = Get-ChunkByFourCC $origChunks 'MNDM'
$genMDNM  = Get-ChunkByFourCC $genChunks  'MNDM'
$origMONM = Get-ChunkByFourCC $origChunks 'MONM'
$genMONM  = Get-ChunkByFourCC $genChunks  'MONM'
if ($origMDNM) { $oData = $original[($origMDNM.Offset+8)..($origMDNM.Offset+7+$origMDNM.Size)]; $oCnt = Count-NullTerminatedStrings $oData; Write-Host "  MDNM Original: $($origMDNM.Size) bytes, ~names=$oCnt"; Write-Host "    Sample: $( (Read-NullStrings $oData $SampleNames) -join ', ' )" }
if ($genMDNM)  { $gData = $generated[($genMDNM.Offset+8)..($genMDNM.Offset+7+$genMDNM.Size)]; $gCnt = Count-NullTerminatedStrings $gData; Write-Host "  MDNM Generated: $($genMDNM.Size) bytes, ~names=$gCnt"; Write-Host "    Sample: $( (Read-NullStrings $gData $SampleNames) -join ', ' )" }
if ($origMONM) { $o2 = $original[($origMONM.Offset+8)..($origMONM.Offset+7+$origMONM.Size)]; $o2Cnt = Count-NullTerminatedStrings $o2; Write-Host "  MONM Original: $($origMONM.Size) bytes, ~names=$o2Cnt"; Write-Host "    Sample: $( (Read-NullStrings $o2 $SampleNames) -join ', ' )" }
if ($genMONM)  { $g2 = $generated[($genMONM.Offset+8)..($genMONM.Offset+7+$genMONM.Size)]; $g2Cnt = Count-NullTerminatedStrings $g2; Write-Host "  MONM Generated: $($genMONM.Size) bytes, ~names=$g2Cnt"; Write-Host "    Sample: $( (Read-NullStrings $g2 $SampleNames) -join ', ' )" }

# Placements totals (sum MHDR sizes across tiles)
Write-Host "`nPlacements Totals (MDDF/MODF):" -ForegroundColor Yellow
function Sum-PlacementSizes([byte[]]$bytes, [int[]]$tileOffsets) {
    $sumD=0; $sumM=0
    foreach ($abs in $tileOffsets) {
        $mhdrDataStart = $abs + 8
        if ($mhdrDataStart + 28 -gt $bytes.Length) { continue }
        $sumD += [BitConverter]::ToInt32($bytes, $mhdrDataStart + 16)
        $sumM += [BitConverter]::ToInt32($bytes, $mhdrDataStart + 24)
    }
    [pscustomobject]@{ SumMDDF=$sumD; SumMODF=$sumM; Total=$sumD+$sumM }
}
if ($origTiles.Count -gt 0) { $oPl = Sum-PlacementSizes $original $origTiles; Write-Host "  Original: MDDF=$($oPl.SumMDDF) MODF=$($oPl.SumMODF) Total=$($oPl.Total)" }
if ($genTiles.Count  -gt 0) { $gPl = Sum-PlacementSizes $generated $genTiles; Write-Host "  Generated: MDDF=$($gPl.SumMDDF) MODF=$($gPl.SumMODF) Total=$($gPl.Total)" }

# Per-tile deep analysis (first N)
Write-Host "`nPer-Tile Deep Analysis (first $Tiles tiles):" -ForegroundColor Yellow
$origTileInfo = if ($origTiles.Count -gt 0) { Analyze-Tiles $original $origTiles $Tiles } else { @() }
$genTileInfo  = if ($genTiles.Count  -gt 0) { Analyze-Tiles $generated $genTiles  $Tiles } else { @() }
for ($i=0; $i -lt [Math]::Min($origTileInfo.Count, $genTileInfo.Count); $i++) {
    $o=$origTileInfo[$i]; $g=$genTileInfo[$i]
    Write-Host "  Tile at 0x$($o.TileOffset.ToString('X')):" -ForegroundColor Cyan
    Write-Host "    Orig: MTEX=$($o.SizeTex) MDDF=$($o.SizeDoo) MODF=$($o.SizeMob) MCNK=$($o.FirstMCNKFourCC)($($o.FirstMCNKSize))"
    Write-Host "    Gen : MTEX=$($g.SizeTex) MDDF=$($g.SizeDoo) MODF=$($g.SizeMob) MCNK=$($g.FirstMCNKFourCC)($($g.FirstMCNKSize))"
}

# Totals by top-level FourCC
Write-Host "`nTop-Level Totals by FourCC:" -ForegroundColor Yellow
$oTotals = Sum-TopLevelByFourCC $origChunks
$gTotals = Sum-TopLevelByFourCC $genChunks
foreach ($o in $oTotals) {
    $g = $gTotals | Where-Object { $_.FourCC -eq $o.FourCC } | Select-Object -First 1
    $gTotal = if ($g) { $g.Total } else { 0 }
    $delta = $gTotal - $o.Total
    Write-Host "  $($o.FourCC): Orig=$($o.Total) Gen=$gTotal Delta=$delta"
}

# Optional JSON summary
if ($Json) {
    $summary = [pscustomobject]@{
        Files = [pscustomobject]@{ Original=$OriginalPath; Generated=$GeneratedPath; SizeOrig=$original.Length; SizeGen=$generated.Length }
        FirstDiff = $firstDiff
        MPHD = [pscustomobject]@{
            Original = if ($origMPHD) { (Parse-MPHD $original $origMPHD.Offset) } else { $null }
            Generated = if ($genMPHD) { (Parse-MPHD $generated $genMPHD.Offset) } else { $null }
        }
        MAIN = [pscustomobject]@{ OrigTiles=$origTiles.Count; GenTiles=$genTiles.Count }
        NameTables = [pscustomobject]@{
            MDNM = [pscustomobject]@{ OrigSize=($origMDNM.Size); GenSize=($genMDNM.Size) }
            MONM = [pscustomobject]@{ OrigSize=($origMONM.Size); GenSize=($genMONM.Size) }
        }
        Placements = [pscustomobject]@{
            Original = $oPl; Generated = $gPl
        }
        PerTile = [pscustomobject]@{ Original=$origTileInfo; Generated=$genTileInfo }
    }
    $summary | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 $Json
    Write-Host "`nSaved JSON summary to $Json" -ForegroundColor Green
}

Write-Host "`n=== Comparison Complete ===" -ForegroundColor Cyan
