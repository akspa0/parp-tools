# WoWRollback Viewer - Complete Fix Plan

## Current Status (2025-09-30)

### ✅ What Works
1. **Object coordinates are correct** - World coordinates extracted from LK ADTs with proper transformation (`17066.66656 - rawCoord`)
2. **Object overlay JSONs generated** - Files in `rollback_outputs/comparisons/0_5_3_3368_vs_0_5_5_3494/viewer/overlays/`
3. **Object markers display** - Blue dots show on map
4. **8x8 grid lazy loading implemented** - Only loads objects for visible tiles
5. **Data regenerated** - All coordinate transforms applied correctly

### ❌ What's Broken
1. **Map is upside down** - Feralas appears on wrong side (east instead of west)
2. **PNG tiles don't load** - Leaflet coordinate system doesn't map to file paths
3. **Wrong directory served** - Script serves `0_5_3_3368` (single version) instead of `0_5_3_3368_vs_0_5_5_3494` (comparison)
4. **Only shows 1 version, 1 map** - Should show 2 versions, 2 maps

## Root Cause Analysis

### Issue 1: Coordinate System Mismatch

**ADT Coordinate System (from wowdev.wiki):**
```
     0,0 -----> X (East)
      |
      |
      v Y (South)
   
   Northwest = (0,0)
   Southeast = (63,63)
```

**Leaflet CRS.Simple:**
```
   Southwest = (0,0)
   Northeast = (maxX, maxY)
   
   Y increases UPWARD (north)
```

**The Problem:**
- ADT: Row 0 = North edge, Row 63 = South edge
- Leaflet: Y=0 = South edge, Y=max = North edge
- **They're inverted!**

### Issue 2: Tile Loading
Leaflet's GridLayer `coords.x` and `coords.y` don't map 1:1 to our game tiles. The coordinate transformation with `CRS.Simple` requires proper scaling.

### Issue 3: Server Directory
The `serve-viewer.ps1` script picks the first directory alphabetically, which is `0_5_3_3368` (single version) instead of `0_5_3_3368_vs_0_5_5_3494` (comparison).

## Solution Plan

### Phase 1: Fix Directory Selection (5 minutes)

**File:** `serve-viewer.ps1`

**Change:**
```powershell
# Current (WRONG):
$comparisonDirs = Get-ChildItem $comparisonsDir -Directory | Where-Object { $_.Name -like "*_vs_*" }

# Should be:
$comparisonDirs = Get-ChildItem $comparisonsDir -Directory | 
    Where-Object { $_.Name -like "*_vs_*" } | 
    Sort-Object LastWriteTime -Descending

# Use most recent comparison
$viewerDir = Join-Path $comparisonDirs[0].FullName "viewer"
```

**Test:** Run `.\serve-viewer.ps1` and verify URL shows "2 version(s) | 2 map(s)"

### Phase 2: Fix Coordinate System (15 minutes)

**Problem:** Need to flip Y-axis to match ADT system

**Option A: Transform in Leaflet (simpler)**
```javascript
// In initializeMap()
const WoWCRS = L.extend({}, L.CRS.Simple, {
    transformation: new L.Transformation(1, 0, -1, 64)
});

map = L.map('map', { crs: WoWCRS, ... });
```

**Option B: Transform in data generation (more correct)**

In `CoordinateTransformer.cs`:
```csharp
// Current:
var row = (int)Math.Floor(HalfTiles - (worldY / TileSpanYards));

// Should be:
// If Y+ = South in world coords, and row 0 = North, formula is correct
// But we need to flip in pixel coords:
var py = ClampUnit(localY) * height;  // Remove the (1.0 - localY) flip
```

**Recommendation:** Use Option B - fix the data generation so coordinates match ADT spec exactly.

### Phase 3: Fix PNG Tile Loading (30 minutes)

**Problem:** Leaflet tile coordinates don't map to `MapName_COL_ROW.png` filenames

**Solution:** Custom transformation in GridLayer

```javascript
const CustomTileLayer = L.GridLayer.extend({
    createTile: function(coords, done) {
        const tile = document.createElement('img');
        
        // Calculate game tile coordinates from Leaflet coords
        // At zoom level z, Leaflet divides space into 2^z tiles per axis
        // We want zoom where 64 game tiles = viewport
        
        // For CRS.Simple with our bounds [0,64]:
        // At zoom 0: whole map is 1 tile (not useful)
        // We need to set minNativeZoom/maxNativeZoom appropriately
        
        const gameTileSize = 1; // Each game tile = 1 map unit
        const col = Math.floor(coords.x);
        const row = Math.floor(coords.y);
        
        if (col < 0 || col > 63 || row < 0 || row > 63) {
            tile.style.display = 'none';
            done(null, tile);
            return tile;
        }
        
        const url = `minimap/${state.selectedVersion}/${state.selectedMap}/${state.selectedMap}_${col}_${row}.png`;
        
        tile.onload = () => done(null, tile);
        tile.onerror = () => {
            tile.style.backgroundColor = '#2a2a2a';
            done(null, tile);
        };
        tile.src = url;
        
        return tile;
    }
});

tileLayer = new CustomTileLayer({
    tileSize: 512,      // Each tile is 512px
    noWrap: true,
    minNativeZoom: 0,
    maxNativeZoom: 0    // No scaling, 1:1 mapping
});
```

### Phase 4: Alternative - SVG Tiles (IF PNG fails)

**If PNG loading continues to fail, consider generating SVG tiles:**

**Pros:**
- Vector graphics scale perfectly at any zoom
- Can embed object markers directly in tiles
- Smaller file size for sparse tiles
- Can style with CSS

**Cons:**
- Need to implement SVG generation from ADT data
- More complex rendering

**Implementation:**
1. Create `SvgTileGenerator.cs` to convert ADT terrain to SVG paths
2. Export as `MapName_COL_ROW.svg` instead of PNG
3. Update viewer to load SVG tiles

**This is a fallback option if PNG tiles can't be fixed.**

## Step-by-Step Execution

### Step 1: Verify Data Integrity
```powershell
cd WoWRollback

# Check comparison exists
ls rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\index.json

# Verify index shows 2 versions, 2 maps
cat rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\index.json

# Check PNG tiles exist
ls rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\minimap\0.5.3.3368\Azeroth\*.png | Select-Object -First 5
```

### Step 2: Fix Server Script
1. Edit `serve-viewer.ps1` as shown in Phase 1
2. Test: `.\serve-viewer.ps1`
3. Verify browser shows "2 version(s) | 2 map(s)"
4. Verify map dropdown shows both Kalimdor and Azeroth

### Step 3: Fix Coordinates
**Choose one approach and commit to it:**

**Approach A (Quick):** Accept current coordinate system, just fix the flip
- Remove `(1.0 - localY)` flip in `ToPixels()`
- Test with browser

**Approach B (Correct):** Align everything to ADT spec
- Review wowdev.wiki ADT coordinate system
- Ensure tile calculation matches: `tileRow = floor((32 - (worldY / 533.33333)))`
- Ensure pixel calculation doesn't double-flip

### Step 4: Fix Tile Loading
1. Implement CustomTileLayer as shown in Phase 3
2. Add console logging to see what tiles are requested
3. Verify tile URLs match actual file paths
4. Test with browser developer tools Network tab

### Step 5: Test & Validate
1. Open browser to comparison viewer
2. Check console for errors
3. Verify PNG tiles load (or see 404s with correct paths)
4. Verify object positions match minimap geography
5. Verify both maps selectable
6. Verify both versions selectable

## Debug Checklist

When things don't work, check:

- [ ] Browser console shows no errors
- [ ] Network tab shows correct PNG paths being requested
- [ ] `index.json` loaded correctly
- [ ] Overlay JSONs loading (check Network tab)
- [ ] Map dropdown populated from `index.json`
- [ ] Version dropdown populated from `index.json`
- [ ] Leaflet map initialized without errors
- [ ] Tile layer added to map
- [ ] Object markers layer added to map

## Key Files Reference

### C# (Data Generation)
- `WoWRollback.Core/Services/LkAdtReader.cs` - Reads coordinates from LK ADTs
- `WoWRollback.Core/Services/CoordinateTransformer.cs` - Converts to tile/pixel coords
- `WoWRollback.Core/Services/ViewerJsonGenerator.cs` - Generates overlay JSONs

### JavaScript (Viewer)
- `ViewerAssets/js/main.js` - Map initialization, tile loading
- `ViewerAssets/js/state.js` - Manages versions/maps
- `ViewerAssets/js/overlayLoader.js` - Loads overlay JSONs
- `ViewerAssets/index.html` - Main viewer page

### Data Files
- `rollback_outputs/comparisons/{comparison}/viewer/index.json` - Versions/maps list
- `rollback_outputs/comparisons/{comparison}/viewer/overlays/{map}/tile_r{row}_c{col}.json` - Objects per tile
- `rollback_outputs/comparisons/{comparison}/viewer/minimap/{version}/{map}/{map}_{col}_{row}.png` - Tile images

## Coordinate System Reference (from wowdev.wiki)

```
Map Coordinate System:
- Map center: (0, 0) in world coordinates
- Map bounds: ±17066.66656 yards
- Tile size: 533.33333 yards
- Total tiles: 64x64 grid

Tile Index Formula:
  tileCol = floor(32 - (worldX / 533.33333))
  tileRow = floor(32 - (worldY / 533.33333))

Tile Position:
  Tile (0, 0) = Northwest corner
  Tile (63, 63) = Southeast corner

Positive Directions:
  X+ = East
  Y+ = South (IMPORTANT!)
```

## Success Criteria

Viewer is "working" when:
1. ✅ Correct directory served (comparison, not single version)
2. ✅ Both maps selectable
3. ✅ Both versions selectable
4. ✅ PNG tiles load and display
5. ✅ Objects positioned correctly on tiles
6. ✅ Geography matches expected (e.g., Feralas on west side of Kalimdor)
7. ✅ Lazy loading keeps memory usage reasonable (<500MB)
8. ✅ Zoom levels allow viewing individual objects

## Next Steps for New Session

1. Start with Phase 1 (fix directory selection) - quick win
2. Test thoroughly before moving to Phase 2
3. For coordinate system (Phase 2), pick ONE approach and stick with it
4. For tile loading (Phase 3), add lots of console logging to debug
5. Consider SVG tiles (Phase 4) only if PNG approach completely fails

## Notes

- The coordinate confusion comes from multiple nested systems (world → tile → local → pixel)
- The data generation is mostly correct now (as of 2025-09-30 regeneration)
- The main issues are viewer-side (Leaflet coordinate mapping)
- Don't regenerate data unless absolutely necessary - focus on viewer fixes first
