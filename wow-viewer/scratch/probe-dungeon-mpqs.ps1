$ErrorActionPreference = 'Continue'
# Find the specific per-map MPQ archives for dungeon maps (0.5.3 stages one MPQ per WDT).
$mapsDir = 'H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft\Data\World\Maps'
Get-ChildItem $mapsDir -Recurse -Filter '*.MPQ' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'Shadow|Wetland|Arathi' } |
    ForEach-Object { Write-Host ("{0}  ({1} KB)" -f $_.FullName, [math]::Round($_.Length/1KB)) }

# Then peek inside the Shadowfang MPQ for its ADT naming scheme.
$sf = Get-ChildItem $mapsDir -Recurse -Filter 'Shadowfang*.MPQ' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($sf) {
    Write-Host ("-- inside " + $sf.FullName)
    $fs = [IO.File]::OpenRead($sf.FullName)
    $bytes = New-Object byte[] ([Math]::Min(2000000, $fs.Length))
    $read = $fs.Read($bytes, 0, $bytes.Length)
    $fs.Close()
    $text = [Text.Encoding]::ASCII.GetString($bytes, 0, $read)
    $hits = [regex]::Matches($text, '[A-Za-z0-9_\\]*Shadowfang[A-Za-z0-9_.\\]*\.adt') |
        ForEach-Object { $_.Value } | Select-Object -Unique -First 30
    if ($hits) { $hits | ForEach-Object { Write-Host ("   " + $_) } }
    else { Write-Host "   (no .adt strings found in first 2 MB — hashed filenames?)" }
}
