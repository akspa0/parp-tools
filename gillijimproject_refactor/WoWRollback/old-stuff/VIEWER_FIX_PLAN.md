# Viewer Fix Plan - Critical Issues

## Issue 1: PNG Tiles Not Loading ❌

**Problem**: Leaflet GridLayer not requesting any PNG minimap tiles.

**Root Cause**: The `createTile()` function logs messages but tiles never appear. Need to debug why Leaflet isn't invoking it.

**Test**: Check browser console for "Creating tile [row,col]" messages.

**Fix**: 
1. Verify GridLayer is actually being added to map
2. Check if bounds [[0,0], [64,64]] are preventing tile display
3. Try simpler L.tileLayer() with URL template

## Issue 2: Missing World Coordinates in Overlay JSON ❌

**Problem**: Objects show world: {x:0, y:0, z:0} in generated JSONs.

**Root Cause**: `AssetTimelineDetailedEntry.WorldX/Y/Z` are not populated.

**Evidence**: 
- `OverlayBuilder.cs` DOES write world coordinates (lines 94-99)
- `CoordinateTransformer` expects these values
- But source data has zeros

**Where coordinates should come from**:
1. Alpha WDT analysis (`AlphaWdtAnalyzer.cs`) extracts UniqueID + coordinates
2. CSV files written with WorldX/Y/Z columns
3. `VersionComparisonService` reads CSVs into timeline entries
4. `AssetTimelineDetailedEntry` should have coordinates populated

**Fix Steps**:
1. ✅ Verify `LkAdtReader` extracts coordinates correctly
2. ✅ Verify `AlphaWdtAnalyzer` writes coordinates to CSV
3. ❌ Check if `VersionComparisonService` reads WorldX/Y/Z columns
4. ❌ Ensure `AssetTimelineDetailedEntry` has WorldX/Y/Z properties

## Issue 3: Redundant JSON Data 📊

**Current Structure** (per object):
```json
{
  "uniqueId": 230658,
  "assetPath": "World\\Doodads\\Kalimdor\\Trees\\StoneTree02.mdx",
  "fileName": "StoneTree02.mdx",
  "fileStem": "StoneTree02",
  "extension": ".mdx",
  "folder": "World/Doodads/Kalimdor/Trees",
  "category": "Doodads",
  "subcategory": "Trees",
  ...
}
```

**Proposed Optimized Structure**:
```json
{
  "uid": 230658,
  "path": "World/Doodads/Kalimdor/Trees/StoneTree02.mdx",
  "type": "M2",  // or "WMO"
  "categories": ["Doodads", "Kalimdor", "Trees"],
  "world": {x, y, z},
  "pixel": {x, y}
}
```

**Space Savings**:
- Remove: `fileName`, `fileStem`, `extension` (derivable from `path`)
- Simplify: `assetPath` → `path`
- Hierarchical: `folder/category/subcategory` → `categories` array
- Shorter keys: `uniqueId` → `uid`

**Estimated reduction**: 30-40% per object

## Issue 4: Wrong Coordinate System 🗺️

**Problem**: Clicking shows tile "60_27" (outside 0-63 range) with coords "60.63, 27.25".

**WoW ADT Coordinate System** (from wowdev.wiki):
- Map is 64x64 ADT tiles (blocks)
- Each block is 533.33333 yards
- Total map: 34133.33312 yards (±17066.66656 from center)
- X+ = North, Y+ = West, Z+ = Up
- Origin at map center
- Formula: `blockIndex = floor((32 - (axis / 533.33333)))`

**Current Issues**:
1. Leaflet CRS.Simple uses [[lat, lng]] = [[Y, X]] coordinates
2. We're not mapping WoW tiles to Leaflet coordinates correctly
3. Bounds should be [[0,0], [64,64]] for 64x64 grid
4. But clicks give values > 63

**Fix**:
```javascript
// Leaflet CRS.Simple coordinate mapping:
// - Leaflet coords [y, x] where y=row (0-63), x=col (0-63)
// - bounds: [[0, 0], [64, 64]]
// - Each tile is 1x1 unit in Leaflet space
// - tileSize: 512px

// Transform to WoW world coords:
function leafletToWowCoords(lat, lng, tileSize = 533.33333) {
    const row = Math.floor(lat);  // 0-63
    const col = Math.floor(lng);  // 0-63
    const localY = lat - row;     // 0-1 within tile
    const localX = lng - col;     // 0-1 within tile
    
    // WoW coordinate system
    const worldX = (32 - row) * tileSize + (0.5 - localY) * tileSize;
    const worldY = (32 - col) * tileSize + (0.5 - localX) * tileSize;
    
    return {worldX, worldY, tileRow: row, tileCol: col};
}
```

## Issue 5: Object Overlay Alignment 🎯

**Requirements**:
1. Minimap tiles must align to strict grid
2. Objects must overlay at correct pixel positions
3. Coordinate transform: WoW world → tile-local → pixel

**Current Transform** (`CoordinateTransformer.cs`):
```csharp
public static (double localX, double localY) ComputeLocalCoordinates(
    double worldX, double worldY, int tileRow, int tileCol)
{
    const double tileSize = 533.33333;
    double tileCenterX = (32.0 - tileRow - 0.5) * tileSize;
    double tileCenterY = (32.0 - tileCol - 0.5) * tileSize;
    double localX = (worldX - tileCenterX) / tileSize + 0.5;
    double localY = (worldY - tileCenterY) / tileSize + 0.5;
    return (localX, localY);
}

public static (double pixelX, double pixelY) ToPixels(
    double localX, double localY, int width, int height)
{
    double pixelX = localX * width;
    double pixelY = (1.0 - localY) * height; // Flip Y
    return (pixelX, pixelY);
}
```

**This looks correct!** The transform:
1. ✅ Centers tile at map center (32, 32)
2. ✅ Converts world → local [0,1]
3. ✅ Converts local → pixels
4. ✅ Flips Y for rendering

**Test**:
- If worldX/Y/Z are correct in JSON
- Objects should overlay properly

## Priority Actions

### P0 - Critical (blocks everything):
1. **Fix PNG tile loading** - Without tiles, nothing else matters
2. **Debug GridLayer** - Add extensive logging
3. **Try fallback L.tileLayer()** - Simpler approach

### P1 - High (needed for usability):
4. **Verify coordinate pipeline** - Trace WorldX/Y/Z from ADT → CSV → JSON
5. **Fix click coordinates** - Must be 0-63 range
6. **Test object overlay** - Once coords are correct

### P2 - Medium (optimization):
7. **Optimize JSON structure** - Remove redundant fields
8. **Improve categories** - Hierarchical folder structure
9. **Add kit/subkit metadata** - For filtering/search

## Next Steps

1. Run coordinate extraction test:
   ```bash
   .\test-lk-coords.ps1
   ```

2. Check if CSVs have WorldX/Y/Z:
   ```bash
   Get-Content rollback_outputs\0.5.3.3368\Kalimdor\*_assetledger.csv | Select -First 5
   ```

3. Check if JSONs have world coords:
   ```bash
   Get-Content rollback_outputs\comparisons\...\viewer\overlays\Kalimdor\tile_r30_c30.json
   ```

4. Fix GridLayer or switch to simpler tile layer

5. Test object rendering once coordinates are correct
