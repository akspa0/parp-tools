#!/usr/bin/env pwsh
# Debug script to examine MCNK structure in a real Alpha file

$alphaFile = "..\..\lk_shadowfang\Shadowfang\index.json"
# Actually, let's look for any .wdt file in the shadowfang analysis output
$alphaFile = Get-ChildItem "..\..\lk_shadowfang\shadowfang_053_analysis\Shadowfang\*.wdt" | Select-Object -First 1 -ExpandProperty FullName
if (-not $alphaFile) {
    Write-Host "Trying alternate location..." -ForegroundColor Yellow
    $alphaFile = "..\..\lk_shadowfang\Shadowfang.wdt"
}
if (-not (Test-Path $alphaFile)) {
    Write-Host "Alpha file not found: $alphaFile" -ForegroundColor Red
    exit 1
}

$bytes = [System.IO.File]::ReadAllBytes($alphaFile)
Write-Host "File size: $($bytes.Length) bytes" -ForegroundColor Cyan
Write-Host ""

# Find first MCNK
$mcnkPattern = [byte[]]@(0x4D, 0x43, 0x4E, 0x4B) # "MCNK" forward
for ($i = 0; $i -lt $bytes.Length - 4; $i++) {
    if ($bytes[$i] -eq 0x4D -and $bytes[$i+1] -eq 0x43 -and $bytes[$i+2] -eq 0x4E -and $bytes[$i+3] -eq 0x4B) {
        Write-Host "Found MCNK at offset 0x$($i.ToString('X'))" -ForegroundColor Green
        
        # Read size (next 4 bytes, little-endian)
        $size = [BitConverter]::ToInt32($bytes, $i + 4)
        Write-Host "  Size in header: $size (0x$($size.ToString('X')))" -ForegroundColor Yellow
        
        # Read first few fields of SMChunk header
        $headerStart = $i + 8
        $flags = [BitConverter]::ToUInt32($bytes, $headerStart + 0x00)
        $indexX = [BitConverter]::ToUInt32($bytes, $headerStart + 0x04)
        $indexY = [BitConverter]::ToUInt32($bytes, $headerStart + 0x08)
        $radius = [BitConverter]::ToSingle($bytes, $headerStart + 0x0C)
        $nLayers = [BitConverter]::ToUInt32($bytes, $headerStart + 0x10)
        $nDoodadRefs = [BitConverter]::ToUInt32($bytes, $headerStart + 0x14)
        $offsHeight = [BitConverter]::ToInt32($bytes, $headerStart + 0x18)
        $offsNormal = [BitConverter]::ToInt32($bytes, $headerStart + 0x1C)
        $offsLayer = [BitConverter]::ToInt32($bytes, $headerStart + 0x20)
        
        Write-Host "  IndexX: $indexX, IndexY: $indexY" -ForegroundColor Cyan
        Write-Host "  Radius: $radius" -ForegroundColor Cyan
        Write-Host "  nLayers: $nLayers" -ForegroundColor Cyan
        Write-Host "  offsHeight (MCVT): 0x$($offsHeight.ToString('X'))" -ForegroundColor Yellow
        Write-Host "  offsNormal (MCNR): 0x$($offsNormal.ToString('X'))" -ForegroundColor Yellow
        Write-Host "  offsLayer (MCLY): 0x$($offsLayer.ToString('X'))" -ForegroundColor Yellow
        
        # Calculate where MCVT should be
        $mcvtAbsolute1 = $i + $offsHeight
        $mcvtAbsolute2 = $i + 8 + $offsHeight
        $mcvtAbsolute3 = $i + 8 + 128 + $offsHeight
        
        Write-Host ""
        Write-Host "  If offset is from MCNK start (including FourCC+size):" -ForegroundColor Magenta
        Write-Host "    MCVT at: 0x$($mcvtAbsolute1.ToString('X'))" -ForegroundColor Magenta
        
        Write-Host "  If offset is from after FourCC+size:" -ForegroundColor Magenta
        Write-Host "    MCVT at: 0x$($mcvtAbsolute2.ToString('X'))" -ForegroundColor Magenta
        
        Write-Host "  If offset is from after header:" -ForegroundColor Magenta
        Write-Host "    MCVT at: 0x$($mcvtAbsolute3.ToString('X'))" -ForegroundColor Magenta
        
        Write-Host ""
        Write-Host "  Checking what's at each location:" -ForegroundColor White
        Write-Host "    At 0x$($mcvtAbsolute1.ToString('X')): $([System.Text.Encoding]::ASCII.GetString($bytes[$mcvtAbsolute1..($mcvtAbsolute1+3)]))" -ForegroundColor White
        Write-Host "    At 0x$($mcvtAbsolute2.ToString('X')): $([System.Text.Encoding]::ASCII.GetString($bytes[$mcvtAbsolute2..($mcvtAbsolute2+3)]))" -ForegroundColor White
        Write-Host "    At 0x$($mcvtAbsolute3.ToString('X')): First 4 bytes as float: $([BitConverter]::ToSingle($bytes, $mcvtAbsolute3))" -ForegroundColor White
        
        break
    }
}
