$ErrorActionPreference = 'Stop'
$path = 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\Azeroth.wdt'
if (-not (Test-Path $path)) {
    $found = Get-ChildItem -Path 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache' -Filter '*.wdt' -ErrorAction SilentlyContinue | Select-Object -First 10
    $found | ForEach-Object { Write-Host $_.FullName }
    return
}
$b = [IO.File]::ReadAllBytes($path)
Write-Host ("Size: " + $b.Length)
$off = 0
while ($off + 8 -le $b.Length) {
    $tag = [Text.Encoding]::ASCII.GetString($b, $off, 4)
    $sz = [BitConverter]::ToInt32($b, $off + 4)
    Write-Host ("Chunk '{0}' @ {1} size {2}" -f $tag, $off, $sz)
    if ($tag -eq 'MPHD' -or $tag -eq 'DHPM') {
        $wmo = [BitConverter]::ToInt32($b, $off + 8 + 8)
        Write-Host ("  IsWmoBased (data[8]==2): " + ($wmo -eq 2))
    }
    if ($tag -eq 'MAIN' -or $tag -eq 'NIAM') {
        $stride = 8
        $count8 = 0; $list8 = @()
        for ($i = 0; $i -lt [math]::Floor($sz / 8); $i++) {
            $v = [BitConverter]::ToInt32($b, $off + 8 + $i * $stride)
            if ($v -ne 0) { $count8++; if ($list8.Count -lt 20) { $list8 += ("idx={0} tileX={1} tileY={2}" -f $i, [math]::Floor($i / 64), ($i % 64)) } }
        }
        Write-Host ("  8-byte stride nonzero: $count8")
        $list8 | ForEach-Object { Write-Host ("    " + $_) }
        $count16 = 0; $list16 = @()
        for ($i = 0; $i -lt [math]::Floor($sz / 16); $i++) {
            $v = [BitConverter]::ToInt32($b, $off + 8 + $i * 16)
            if ($v -ne 0) { $count16++; if ($list16.Count -lt 10) { $list16 += ("idx={0} tileX={1} tileY={2}" -f $i, [math]::Floor($i / 64), ($i % 64)) } }
        }
        Write-Host ("  16-byte stride nonzero (what MainAlpha sees): $count16")
        $list16 | ForEach-Object { Write-Host ("    " + $_) }
    }
    if ($sz -le 0) { break }
    $pad = $sz % 2
    $off += 8 + $sz + $pad
}
