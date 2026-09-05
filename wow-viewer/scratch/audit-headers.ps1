$ErrorActionPreference = 'Continue'
Get-ChildItem 'wow-viewer\src\viewer\WoWViewer' -Filter 'ViewerApp*.cs' |
    Select-String -Pattern 'CollapsingHeader\("' |
    ForEach-Object {
        $label = $_.Line -replace '.*CollapsingHeader\("', '' -replace '".*$', ''
        Write-Host ("{0}:{1}: {2}" -f (Split-Path $_.Path -Leaf), $_.LineNumber, $label)
    }