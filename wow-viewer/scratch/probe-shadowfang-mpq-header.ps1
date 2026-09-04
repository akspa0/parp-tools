$ErrorActionPreference = 'Stop'
# Dump the Shadowfang.wdt.MPQ header + table info to see how many files it holds
# and whether entries are named or hashed-only.
$path = 'H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft\Data\World\Maps\Shadowfang\Shadowfang.wdt.MPQ'
$b = [IO.File]::ReadAllBytes($path)
Write-Host ("MPQ size: " + $b.Length)

$magic = [Text.Encoding]::ASCII.GetString($b, 0, 3)
$headerSize = [BitConverter]::ToInt32($b, 4)
$archiveSize = [BitConverter]::ToUInt32($b, 8)
$formatVersion = [BitConverter]::ToUInt16($b, 12)
$blockSize = [BitConverter]::ToUInt16($b, 14)
$hashTablePos = [BitConverter]::ToUInt32($b, 16)
$blockTablePos = [BitConverter]::ToUInt32($b, 20)
$hashTableSize = [BitConverter]::ToUInt32($b, 24)
$blockTableSize = [BitConverter]::ToUInt32($b, 28)
Write-Host ("magic={0} headerSize={1} archiveSize={2} version={3} sectorShift={4}" -f $magic, $headerSize, $archiveSize, $formatVersion, $blockSize)
Write-Host ("hashTable @ {0} ({1} entries), blockTable @ {2} ({3} entries)" -f $hashTablePos, $hashTableSize, $blockTablePos, $blockTableSize)

Write-Host ("First 32 bytes of block table (raw): " + ((0..31 | ForEach-Object { $b[$blockTablePos + $_] }) -join ','))
Write-Host ("First 32 bytes of hash table (raw): " + ((0..31 | ForEach-Object { $b[$hashTablePos + $_] }) -join ','))

$text = [Text.Encoding]::ASCII.GetString($b)
foreach ($needle in @('(listfile)', '.adt', 'Shadowfang')) {
    $m = [regex]::Matches($text, [regex]::Escape($needle))
    Write-Host ("'{0}' occurrences: {1}" -f $needle, $m.Count)
}
