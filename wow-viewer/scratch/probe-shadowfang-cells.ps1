$ErrorActionPreference = 'Stop'
$path = 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\Shadowfang.wdt'
$b = [IO.File]::ReadAllBytes($path)
$mainData = 52 + 8   # NIAM @ 52, data at 60

# Dump the raw 8-byte cells around the known-real entries (idx 1945..1949)
foreach ($idx in 1944..1950) {
    $v = [BitConverter]::ToInt32($b, $mainData + $idx * 8)
    $v2 = [BitConverter]::ToInt32($b, $mainData + $idx * 8 + 4)
    Write-Host ("idx={0} first={1} second={2}" -f $idx, $v, $v2)
}

# Dump bytes 0..64 of MAIN data:
Write-Host ("MAIN first 64 bytes: " + ((0..63 | ForEach-Object { $b[$mainData + $_] }) -join ','))
# And around idx 1945:
Write-Host ("Cells 1944..1946 raw: " + ((0..23 | ForEach-Object { $b[$mainData + 1944*8 + $_] }) -join ','))
