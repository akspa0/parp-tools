$ErrorActionPreference = 'Stop'
# Probe Shadowfang.wdt: MHDR offsets from MAIN (8-byte cells), then walk one ADT's MCIN.
$path = 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\Shadowfang.wdt'
$b = [IO.File]::ReadAllBytes($path)
$mainData = 52 + 8   # NIAM @ 52, data at 60

# Tile (30,25) is the first real tile: idx = 30*64+25 = 1945
$tileIdx = 1945
$mhdrOff = [BitConverter]::ToInt32($b, $mainData + $tileIdx * 8)
Write-Host ("Tile idx {0} MHDR offset: {1}" -f $tileIdx, $mhdrOffset)

# MHDR chunk at that offset
$tag = [Text.Encoding]::ASCII.GetString($b, $mhdrOff, 4)
$sz = [BitConverter]::ToInt32($b, $mhdrOff + 4)
Write-Host ("MHDR chunk '{0}' size {1}" -f $tag, $sz)
$mhdrData = $mhdrOff + 8
$mcinRel = [BitConverter]::ToInt32($b, $mhdrData + 0)
$mtexRel = [BitConverter]::ToInt32($b, $mhdrData + 4)
$mddfRel = [BitConverter]::ToInt32($b, $mhdrData + 12)
$modfRel = [BitConverter]::ToInt32($b, $mhdrData + 20)
Write-Host ("MHDR rel: mcin={0} mtex={1} mddf={2} modf={3}" -f $mcinRel, $mtexRel, $mddfRel, $modfRel)

# MCIN: absolute = mhdrData + mcinRel
$mcinOff = $mhdrData + $mcinRel
$mcinTag = [Text.Encoding]::ASCII.GetString($b, $mcinOff, 4)
$mcinSize = [BitConverter]::ToInt32($b, $mcinOff + 4)
Write-Host ("MCIN chunk '{0}' size {1}" -f $mcinTag, $mcinSize)
$mcinData = $mcinOff + 8
$mcnkCount = 0
$firstOffsets = @()
for ($i = 0; $i -lt 256; $i++) {
    $v = [BitConverter]::ToInt32($b, $mcinData + $i * 16)
    if ($v -ne 0) { $mcnkCount++; if ($firstOffsets.Count -lt 8) { $firstOffsets += $v } }
}
Write-Host ("MCNK offsets nonzero: {0}; first: {1}" -f $mcnkCount, ($firstOffsets -join ','))
# Verify the first MCNK tag at its offset
if ($firstOffsets.Count -gt 0) {
    $t = [Text.Encoding]::ASCII.GetString($b, $firstOffsets[0], 4)
    Write-Host ("First MCNK tag: '{0}'" -f $t)
}
