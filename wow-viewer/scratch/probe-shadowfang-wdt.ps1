$ErrorActionPreference = 'Stop'
$path = 'I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\Shadowfang.wdt'
$b = [IO.File]::ReadAllBytes($path)
Write-Host ("Size: " + $b.Length)
$off = 0
while ($off + 8 -le $b.Length) {
    $tag = [Text.Encoding]::ASCII.GetString($b, $off, 4)
    $sz = [BitConverter]::ToInt32($b, $off + 4)
    Write-Host ("Chunk '{0}' @ {1} size {2}" -f $tag, $off, $sz)
    if ($tag -eq 'MPHD') {
        $vals = 0..15 | ForEach-Object { $b[$off + 8 + $_] }
        Write-Host ("  MPHD data[0..15]: " + ($vals -join ','))
        $wmo = [BitConverter]::ToInt32($b, $off + 8 + 8)
        Write-Host ("  IsWmoBased (data[8]==2): " + ($wmo -eq 2))
    }
    if ($tag -eq 'MAIN') {
        $count = 0
        for ($i = 0; $i -lt 4096; $i++) {
            $v = [BitConverter]::ToInt32($b, $off + 8 + $i * 16)
            if ($v -ne 0) {
                $count++
                Write-Host ("  nonzero idx={0} tileX={1} tileY={2} val={3}" -f $i, [math]::Floor($i / 64), ($i % 64), $v)
            }
        }
        Write-Host ("  MAIN nonzero count: " + $count)
    }
    if ($sz -le 0) { break }
    $pad = $sz % 2
    $off += 8 + $sz + $pad
}
