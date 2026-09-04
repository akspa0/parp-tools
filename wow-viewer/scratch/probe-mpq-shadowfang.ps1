$ErrorActionPreference = 'Continue'
$clients = Get-ChildItem 'H:\CLIENTS\Vanilla\0.x\0_5_3_3368' -Recurse -Include '*.mpq','*.MPQ' -ErrorAction SilentlyContinue | Sort-Object Length -Descending
Write-Host ("MPQs found: " + $clients.Count)
foreach ($mpq in ($clients | Select-Object -First 8)) {
    Write-Host ("-- " + $mpq.FullName + " (" + [math]::Round($mpq.Length/1MB) + " MB)")
    try {
        $fs = [IO.File]::OpenRead($mpq.FullName)
        $bytes = New-Object byte[] ([Math]::Min(8000000, $fs.Length))
        $read = $fs.Read($bytes, 0, $bytes.Length)
        $fs.Close()
        $text = [Text.Encoding]::ASCII.GetString($bytes, 0, $read)
        $hits = [regex]::Matches($text, 'Shadowfang[^\x00]{0,50}') |
            ForEach-Object { $_.Value } | Select-Object -Unique -First 15
        if ($hits) { $hits | ForEach-Object { Write-Host ("   " + $_) } }
        else { Write-Host "   (no Shadowfang strings in first MBs)" }
    } catch { Write-Host ("   read error: " + $_.Exception.Message) }
}
