param(
    [string]$BaseClientPath = 'H:\CLIENTS\World of Warcraft Cata beta 11927',
    [string]$BaseBuildLabel = '4.0.0.11927',
    [string]$OverlayMapRoot = (Join-Path $PSScriptRoot '..\test_data\original_development\World\Maps\development'),
    [string]$MinimapRoot = (Join-Path $PSScriptRoot '..\test_data\development\World\Textures\Minimap'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\..\output\tmp\original_development_client_4_0_0_11927'),
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    return [System.IO.Path]::GetFullPath($PathValue)
}

function New-LinkedDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LinkPath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (Test-Path $LinkPath) {
        Remove-Item -Path $LinkPath -Force -Recurse
    }

    try {
        New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
        return 'junction'
    }
    catch {
        New-Item -ItemType SymbolicLink -Path $LinkPath -Target $TargetPath | Out-Null
        return 'symlink'
    }
}

$resolvedBaseClientPath = Resolve-FullPath -PathValue $BaseClientPath
$resolvedOverlayMapRoot = Resolve-FullPath -PathValue $OverlayMapRoot
$resolvedMinimapRoot = Resolve-FullPath -PathValue $MinimapRoot
$resolvedOutputRoot = Resolve-FullPath -PathValue $OutputRoot

$baseDataRoot = Join-Path $resolvedBaseClientPath 'Data'
if (-not (Test-Path $baseDataRoot)) {
    throw "Base client Data folder not found: $baseDataRoot"
}

if (-not (Test-Path $resolvedOverlayMapRoot)) {
    throw "Original development map root not found: $resolvedOverlayMapRoot"
}

if (-not (Test-Path $resolvedMinimapRoot)) {
    throw "Development minimap root not found: $resolvedMinimapRoot"
}

if (Test-Path $resolvedOutputRoot) {
    if (-not $Force) {
        throw "Output root already exists. Re-run with -Force to rebuild: $resolvedOutputRoot"
    }

    Remove-Item -Path $resolvedOutputRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $resolvedOutputRoot 'World\Maps') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $resolvedOutputRoot 'World\Textures') -Force | Out-Null

$dataLinkType = New-LinkedDirectory -LinkPath (Join-Path $resolvedOutputRoot 'Data') -TargetPath $baseDataRoot
$mapLinkType = New-LinkedDirectory -LinkPath (Join-Path $resolvedOutputRoot 'World\Maps\development') -TargetPath $resolvedOverlayMapRoot
$minimapLinkType = New-LinkedDirectory -LinkPath (Join-Path $resolvedOutputRoot 'World\Textures\Minimap') -TargetPath $resolvedMinimapRoot

$metadata = [ordered]@{
    schema_version = 'original-development-overlay.v1'
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    base_build_label = $BaseBuildLabel
    base_client_path = $resolvedBaseClientPath
    base_data_path = $baseDataRoot
    overlay_map_root = $resolvedOverlayMapRoot
    minimap_root = $resolvedMinimapRoot
    links = [ordered]@{
        data = $dataLinkType
        development_map = $mapLinkType
        minimap = $minimapLinkType
    }
}

$metadataPath = Join-Path $resolvedOutputRoot 'original_development_overlay.json'
$metadata | ConvertTo-Json -Depth 5 | Set-Content -Path $metadataPath -Encoding ASCII

Write-Host "Original development overlay staged." -ForegroundColor Green
Write-Host ("  Base client : {0}" -f $resolvedBaseClientPath)
Write-Host ("  Overlay map : {0}" -f $resolvedOverlayMapRoot)
Write-Host ("  Minimap root: {0}" -f $resolvedMinimapRoot)
Write-Host ("  Output root : {0}" -f $resolvedOutputRoot)
Write-Host ("  Metadata    : {0}" -f $metadataPath)