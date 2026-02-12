#!/usr/bin/env pwsh
# Test coordinate transformation with sample data

$json = Get-Content "rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\overlays\Azeroth\tile_r20_c44.json" | ConvertFrom-Json

Write-Host "Tile: r$($json.tile.row), c$($json.tile.col)" -ForegroundColor Cyan
Write-Host "Minimap: $($json.minimap.width)x$($json.minimap.height)" -ForegroundColor Cyan
Write-Host ""

Write-Host "Sample objects:" -ForegroundColor Yellow
$samples = $json.layers[0].kinds[0].points | Select-Object -First 10

foreach ($obj in $samples) {
    $worldX = $obj.world.x
    $worldY = $obj.world.y
    $localX = $obj.local.x
    $localY = $obj.local.y
    $pixelX = $obj.pixel.x
    $pixelY = $obj.pixel.y
    
    # Verify tile calculation
    $tileCol = [Math]::Floor(32 - ($worldX / 533.33333))
    $tileRow = [Math]::Floor(32 - ($worldY / 533.33333))
    
    Write-Host ""
    Write-Host "Asset: $($obj.assetPath.Split('/')[-1])"
    Write-Host "  World: ($worldX, $worldY)"
    Write-Host "  Calculated tile: r${tileRow}, c${tileCol} $(if ($tileRow -eq $json.tile.row -and $tileCol -eq $json.tile.col) { '✓' } else { '✗ WRONG TILE!' })"
    Write-Host "  Local: ($localX, $localY)"
    Write-Host "  Pixel: ($pixelX, $pixelY)"
}

Write-Host ""
Write-Host "Checking for tile assignment errors..." -ForegroundColor Yellow
$wrongTile = 0
$total = $json.layers[0].kinds[0].points.Count

foreach ($obj in $json.layers[0].kinds[0].points) {
    $tileCol = [Math]::Floor(32 - ($obj.world.x / 533.33333))
    $tileRow = [Math]::Floor(32 - ($obj.world.y / 533.33333))
    
    if ($tileRow -ne $json.tile.row -or $tileCol -ne $json.tile.col) {
        $wrongTile++
    }
}

Write-Host ""
if ($wrongTile -eq 0) {
    Write-Host "✓ All $total objects correctly assigned to this tile" -ForegroundColor Green
} else {
    Write-Host "✗ $wrongTile / $total objects assigned to WRONG tile!" -ForegroundColor Red
}
