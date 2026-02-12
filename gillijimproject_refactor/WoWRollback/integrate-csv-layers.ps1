# Quick script to integrate CSV-based Sedimentary Layers
$mainJs = "ViewerAssets\js\main.js"
$content = Get-Content $mainJs -Raw

Write-Host "🔧 Integrating CSV-based Sedimentary Layers..." -ForegroundColor Cyan

# 1. Add import
$content = $content -replace 'import \{ OverlayManager \} from ''\.\/overlays\/overlayManager\.js'';', 
    "import { OverlayManager } from './overlays/overlayManager.js';`nimport { SedimentaryLayersManagerCSV } from './sedimentary-layers-csv.js';"

# 2. Add variable declaration
$content = $content -replace 'let overlayManager = null; \/\/ Terrain overlay manager',
    "let overlayManager = null; // Terrain overlay manager`nlet sedimentaryLayers = null; // UniqueID filter manager (CSV-based)"

# 3. Add initialization after OverlayManager
$content = $content -replace '(overlayManager = new OverlayManager\(map\);)',
    "`$1`n    `n    // Initialize sedimentary layers manager (CSV-based)`n    sedimentaryLayers = new SedimentaryLayersManagerCSV(map, state);"

# 4. Add registration calls (find objectMarkers.addLayer and add after each)
$content = $content -replace '(objectMarkers\.addLayer\(square\);)(\r?\n)',
    "`$1`$2                            `$2                            // Register with sedimentary layers`$2                            if (sedimentaryLayers) {`$2                                sedimentaryLayers.registerMarker(square, obj.uniqueId || 0, row, col);`$2                            }`$2"

$content = $content -replace '(objectMarkers\.addLayer\(circle\);)(\r?\n)(?!.*sedimentary)',
    "`$1`$2                            `$2                            // Register with sedimentary layers`$2                            if (sedimentaryLayers) {`$2                                sedimentaryLayers.registerMarker(circle, obj.uniqueId || 0, row, col);`$2                            }`$2"

$content = $content -replace '(objectMarkers\.addLayer\(marker\);)(\r?\n)(?!.*sedimentary)',
    "`$1`$2                        `$2                        // Register with sedimentary layers`$2                        if (sedimentaryLayers) {`$2                            sedimentaryLayers.registerMarker(marker, obj.uniqueId || 0, row, col);`$2                        }`$2"

# Save
Set-Content $mainJs $content -NoNewline

Write-Host "✅ Integration complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Now refresh your browser (Ctrl+F5) and click 'Load UniqueID Ranges'" -ForegroundColor Yellow
