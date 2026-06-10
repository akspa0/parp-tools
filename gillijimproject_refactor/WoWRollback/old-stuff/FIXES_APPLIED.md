# Fixes Applied - Viewer Tile Loading & Coordinates

## Changes Made

### 1. Fixed PNG Tile Loading ✅

**Problem**: Leaflet wasn't requesting any PNG tiles, only overlay JSONs.

**Solution**: Switched from custom `GridLayer` to standard `L.tileLayer()` with URL template.

**Changes**:
- `ViewerAssets/js/main.js` - `updateTileLayer()` function
- Now uses simple URL template: `minimap/{version}/{map}/{map}_{x}_{y}.png`
- Added extensive logging to track tile requests
- Added event handlers for `tileload`, `tileerror`, `tileloadstart`

**Test**: Refresh browser and check console for:
```
→ Requesting tile [30,30]: minimap/0.5.3.3368/Kalimdor/Kalimdor_30_30.png
✓ Loaded tile [30,30]
```

### 2. Fixed Click Coordinates ✅

**Problem**: Clicks showed tiles like "60_27" (outside valid 0-63 range).

**Solution**: Added clamping to ensure coordinates stay within 0-63.

**Changes**:
- `handleMapClick()` now clamps row/col to `Math.max(0, Math.min(63, value))`
- Shows both tile coords and Leaflet coords for debugging
- Added console logging for tile availability

**Test**: Click anywhere - should show tile in 0-63 range.

### 3. Created Coordinate Extraction Script ✅

**Problem**: No workflow to regenerate data with coordinates from converted LK ADTs.

**Solution**: Created comprehensive regeneration script.

**New File**: `regenerate-with-coordinates.ps1`

**Features**:
- Builds solution
- Analyzes each version/map with `--converted-adt-dir` flag
- Verifies coordinates in generated CSVs
- Generates comparison with viewer
- Shows summary of coordinate extraction success
- Auto-starts viewer

**Usage**:
```powershell
.\regenerate-with-coordinates.ps1
```

**Parameters**:
```powershell
.\regenerate-with-coordinates.ps1 `
    -ConvertedAdtDir "path\to\output_dirfart2\World\Maps" `
    -Versions @("0.5.3.3368", "0.5.5.3494") `
    -Maps @("Kalimdor", "Azeroth")
```

### 4. Fixed Map Initialization ✅

**Changes**:
- Set proper bounds `[[0,0], [64,64]]` for 64x64 tile grid
- Added `maxBounds` and `maxBoundsViscosity` to prevent scrolling outside map
- Improved initial view with `fitBounds()`

## Next Steps

### Immediate Testing

1. **Run regeneration script**:
   ```powershell
   cd WoWRollback
   .\regenerate-with-coordinates.ps1
   ```

2. **Check console output** for:
   - ✓ Coordinate verification (should show non-zero coords)
   - ✓ Tile loading messages
   - ⚠ Any warnings about missing converted ADTs

3. **Test in browser**:
   - Open http://localhost:8080/index.html
   - Check browser console for tile load messages
   - Verify tiles appear on map
   - Click tiles and check coordinate display

### Expected Results

**Browser Console**:
```
Loading 156 tiles for Kalimdor, version 0.5.3.3368
Tile URL template: minimap/0.5.3.3368/Kalimdor/Kalimdor_{x}_{y}.png
→ Requesting tile [30,30]: minimap/0.5.3.3368/Kalimdor/Kalimdor_30_30.png
✓ Loaded tile [30,30]
...
```

**Network Tab**:
```
GET /minimap/0.5.3.3368/Kalimdor/Kalimdor_30_30.png → 200 OK
GET /minimap/0.5.3.3368/Kalimdor/Kalimdor_30_31.png → 200 OK
...
```

**Sidebar Click Info**:
```
Tile: 30_30
Coord: Tile [30,30] | Leaflet (30.45, 30.67)
```

**Overlay JSON** (if coordinates extracted correctly):
```json
{
  "world": {
    "x": 15990.69,
    "y": 16191.14,
    "z": 42.67
  },
  "pixel": {
    "x": 234.56,
    "y": 156.78
  }
}
```

### If Tiles Still Don't Load

**Debug Steps**:

1. Check if PNG files exist:
   ```powershell
   Test-Path "rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\minimap\0.5.3.3368\Kalimdor\Kalimdor_30_30.png"
   ```

2. Check browser Network tab:
   - Are PNG requests being made?
   - What's the response status?
   - Check the actual URL being requested

3. Check console for errors:
   - JavaScript errors?
   - Tile load failures?

4. Verify Leaflet is loading:
   - Check for Leaflet library errors
   - Verify `L` global object exists

### If Coordinates Are Still (0,0,0)

**Troubleshooting**:

1. **Verify converted ADTs exist**:
   ```powershell
   Test-Path "output_dirfart2\World\Maps\Kalimdor\Kalimdor_30_30.adt"
   ```

2. **Check CSV has coordinates**:
   ```powershell
   Get-Content "rollback_outputs\0.5.3\Kalimdor\*_assetledger.csv" | Select -First 3
   ```
   Should show `WorldX,WorldY,WorldZ` columns with non-zero values.

3. **Re-run with explicit converted ADT path**:
   ```powershell
   .\WoWRollback.Cli\bin\Release\net9.0\WoWRollback.Cli.exe analyze-alpha-wdt `
       --wdt-file "test_data\0.5.3\tree\World\Maps\Kalimdor\Kalimdor.wdt" `
       --converted-adt-dir "i:\full\path\to\output_dirfart2\World\Maps\Kalimdor" `
       --out rollback_outputs
   ```

## Remaining Work

### JSON Optimization (Not Critical)

**Goal**: Reduce JSON file size by removing redundant data.

**Changes Needed**:
- Modify `OverlayBuilder.cs` to remove `fileName`, `fileStem`, `extension`
- Restructure `folder/category/subcategory` into `categories` array
- Shorten keys: `uniqueId` → `uid`, `assetPath` → `path`

**Estimated Savings**: 30-40% per object

**Priority**: Medium (do after coordinate extraction works)

### WoW World Coordinate Transform (Future)

**Goal**: Convert Leaflet clicks to actual WoW world coordinates.

**Formula** (from ADT docs):
```javascript
function leafletToWowWorld(lat, lng) {
    const tileSize = 533.33333;
    const row = Math.floor(lat);
    const col = Math.floor(lng);
    const localY = lat - row;
    const localX = lng - col;
    
    const worldX = (32 - row - 0.5 + localY) * tileSize;
    const worldY = (32 - col - 0.5 + localX) * tileSize;
    
    return {worldX, worldY, row, col};
}
```

**Priority**: Low (nice to have for debugging)
