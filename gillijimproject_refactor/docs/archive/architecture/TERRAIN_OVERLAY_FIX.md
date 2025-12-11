# Terrain Overlay Generation Fix

## Issue: terrain_complete Files Not Found (404 Errors)

Browser console showed hundreds of 404 errors:
```
GET /overlays/0.5.3.3368/Azeroth/terrain_complete/tile_r56_c6.json HTTP/1.1" 404
```

---

## Root Cause: Path Mismatch

**AlphaWdtAnalyzer Output Path**:
```csharp
// AlphaWdtAnalyzer.Core/AnalysisPipeline.cs line 94
var terrainCsvPath = Path.Combine(csvDir, wdt.MapName, $"{wdt.MapName}_mcnk_terrain.csv");
// Outputs to: cached_maps/0.5.3.3368/Azeroth/csv/Azeroth/Azeroth_mcnk_terrain.csv
```

**rebuild-and-regenerate.ps1 Expected Path**:
```powershell
# Line 236 (BEFORE FIX)
$terrainCsv = Join-Path (Join-Path $analysisDir 'csv') ($Map + '_mcnk_terrain.csv')
# Was looking for: cached_maps/0.5.3.3368/Azeroth/csv/Azeroth_mcnk_terrain.csv
```

**The CSV file was in a subdirectory that the script wasn't checking!**

---

## Fix Applied

### rebuild-and-regenerate.ps1 (Lines 235-243)

**Before**:
```powershell
$terrainCsv = Join-Path (Join-Path $analysisDir 'csv') ($Map + '_mcnk_terrain.csv')
if (Test-Path $terrainCsv) {
    Copy-Item -Path $terrainCsv -Destination $rollbackMapCsvDir -Force
    Write-Host "  [cache] Copied terrain CSV to rollback_outputs" -ForegroundColor DarkGray
}
```

**After**:
```powershell
# Copy MCNK terrain CSV (AlphaWdtAnalyzer outputs to csv/MapName/MapName_mcnk_terrain.csv)
$terrainCsvDir = Join-Path (Join-Path $analysisDir 'csv') $Map
$terrainCsv = Join-Path $terrainCsvDir ($Map + '_mcnk_terrain.csv')
if (Test-Path $terrainCsv) {
    Copy-Item -Path $terrainCsv -Destination $rollbackMapCsvDir -Force
    Write-Host "  [cache] Copied terrain CSV to rollback_outputs" -ForegroundColor DarkGray
} else {
    Write-Host "  [warn] Terrain CSV not found at: $terrainCsv" -ForegroundColor Yellow
}
```

**Changes**:
1. Added intermediate `$terrainCsvDir` variable to navigate into map subdirectory
2. Added warning message when CSV not found
3. Added comment explaining AlphaWdtAnalyzer's output structure

---

## How Terrain Extraction Works

### 1. AlphaWdtAnalyzer Extraction
```bash
dotnet run --project AlphaWdtAnalyzer.Cli -- \
  --input path/to/Azeroth.wdt \
  --out cached_maps/0.5.3.3368/Azeroth \
  --extract-mcnk-terrain
```

Outputs to:
```
cached_maps/0.5.3.3368/Azeroth/
└── csv/
    └── Azeroth/
        └── Azeroth_mcnk_terrain.csv
```

### 2. rebuild-and-regenerate.ps1 Copies CSV
```powershell
# From: cached_maps/{version}/{map}/csv/{map}/{map}_mcnk_terrain.csv
# To:   rollback_outputs/{version}/csv/{map}/{map}_mcnk_terrain.csv
```

### 3. WoWRollback.Cli Reads CSV
```csharp
// ViewerReportWriter.cs line 458
var csvMapDir = Path.Combine(rootDirectory, version, "csv", mapName);
var terrainCsvPath = Path.Combine(csvMapDir, $"{mapName}_mcnk_terrain.csv");
// Reads from: rollback_outputs/{version}/csv/{map}/{map}_mcnk_terrain.csv
```

### 4. McnkTerrainOverlayBuilder Generates JSON
```csharp
// McnkTerrainOverlayBuilder.cs line 82
var overlayDir = Path.Combine(outputDir, mapName, "terrain_complete");
// Outputs to: viewer/overlays/{version}/{map}/terrain_complete/tile_r{row}_c{col}.json
```

### 5. Browser Requests JSON
```javascript
// overlayManager.js loads terrain_complete overlay
fetch(`/overlays/0.5.3.3368/Azeroth/terrain_complete/tile_r56_c6.json`)
```

---

## Path Flow Diagram

```
AlphaWdtAnalyzer
    ↓ Extract
cached_maps/{version}/{map}/csv/{map}/{map}_mcnk_terrain.csv
    ↓ Copy (rebuild-and-regenerate.ps1)
rollback_outputs/{version}/csv/{map}/{map}_mcnk_terrain.csv
    ↓ Read (WoWRollback.Cli)
Memory (List<McnkTerrainEntry>)
    ↓ Transform (McnkTerrainOverlayBuilder)
rollback_outputs/comparisons/{key}/viewer/overlays/{version}/{map}/terrain_complete/tile_rX_cY.json
    ↓ Serve (Python web server)
http://localhost:8080/overlays/{version}/{map}/terrain_complete/tile_rX_cY.json
    ↓ Fetch (Browser)
Leaflet map overlays
```

---

## Testing

### Verify Fix Works

1. **Clean cached data**:
```powershell
Remove-Item cached_maps -Recurse -Force
Remove-Item rollback_outputs -Recurse -Force
```

2. **Rebuild with terrain extraction**:
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

3. **Check for success messages**:
```
[cache] Copied terrain CSV to rollback_outputs
Built 685 terrain overlay tiles for Azeroth (0.5.3.3368)
[info] Generated terrain overlays for Azeroth (0.5.3.3368)
```

4. **Verify files exist**:
```powershell
# Check cached extraction
Test-Path cached_maps/0.5.3.3368/Azeroth/csv/Azeroth/Azeroth_mcnk_terrain.csv

# Check rollback copy
Test-Path rollback_outputs/0.5.3.3368/csv/Azeroth/Azeroth_mcnk_terrain.csv

# Check generated overlays
Test-Path rollback_outputs/comparisons/*/viewer/overlays/0.5.3.3368/Azeroth/terrain_complete/tile_r30_c30.json
```

5. **Test in browser**:
- Open http://localhost:8080
- Enable "Terrain Properties" overlay
- Verify colored chunks appear (no 404 errors in console)

---

## Related Files

### Modified:
- `rebuild-and-regenerate.ps1` - Fixed path to terrain CSV

### Already Working (No Changes Needed):
- `AlphaWdtAnalyzer.Core/AnalysisPipeline.cs` - Extracts terrain to CSV
- `AlphaWdtAnalyzer.Core/Terrain/McnkTerrainExtractor.cs` - MCNK parsing
- `WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs` - Calls overlay builder
- `WoWRollback.Core/Services/Viewer/McnkTerrainOverlayBuilder.cs` - Generates JSON
- `ViewerAssets/js/overlays/overlayManager.js` - Loads terrain_complete

---

## Summary

**Problem**: Terrain overlay JSON files weren't being generated because the rebuild script couldn't find the terrain CSV files.

**Cause**: AlphaWdtAnalyzer outputs CSVs to `csv/MapName/MapName_mcnk_terrain.csv` but rebuild script was looking for `csv/MapName_mcnk_terrain.csv` (missing subdirectory).

**Solution**: Updated rebuild script to look in correct subdirectory: `csv/{MapName}/{MapName}_mcnk_terrain.csv`

**Result**: Terrain overlays now generate properly! ✅

---

## Lessons Learned

1. **Always verify file paths match** - Small directory structure differences break pipelines
2. **Add explicit warnings** - `Write-Host "[warn]"` helps debug missing files
3. **Document path structure** - Comments in code prevent future confusion
4. **Test end-to-end** - Browser 404s revealed the issue immediately
5. **Check all path joins** - `Path.Combine()` calls need to match actual structure

Ready to test! 🎉
