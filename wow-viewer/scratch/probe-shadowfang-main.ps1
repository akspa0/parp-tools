$ErrorActionPreference = 'Stop'
$path = 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\Shadowfang.wdt'
$b = [IO.File]::ReadAllBytes($path)
# NIAM @ 52, size 32768, data starts at 60
$dataOff = 52 + 8
$count = 0
for ($i = 0; $i -lt 4096; $i++) {
    $v = [BitConverter]::ToInt32($b, $dataOff + $i * 8)
    if ($v -ne 0) {
        $count++
        Write-Host ("entry idx={0} tileX={1} tileY={2} offset={3} second={4}" -f $i, [math]::Floor($i / 64), ($i % 64), $v, [BitConverter]::ToInt32($b, $dataOff + $i * 8 + 4))
    }
}
Write-Host ("8-byte nonzero count: " + $count)
# What the 16-byte stride parser actually sees:
$seen = 0
for ($i = 0; $i -lt 2048; $i++) {
    $v = [BitConverter]::ToInt32($b, $dataOff + $i * 16)
    if ($v -ne 0) { $seen++ }
}
Write-Host ("16-byte-stride nonzero (phantom) count: " + $seen)
# DHPM contents
Write-Host ("DHPM data[0..31]: " + ((0..31 | ForEach-Object { $b[20 + $_] }) -join ','))
