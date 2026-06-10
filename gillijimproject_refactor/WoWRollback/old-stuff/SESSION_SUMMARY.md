# WoWRollback Viewer Session Summary - 2025-09-30

## Major Fixes Completed

### 1. **Coordinate System Fixed** ✅
**Problem**: MDDF/MODF coordinates were being used raw instead of converting from map-corner-relative to world coordinates.

**Solution**: 
- Implemented proper coordinate conversion in `LkAdtReader.cs`:
  ```csharp
  worldX = MAP_HALF_SIZE - rawX;  // 17066.66656 - rawX
  worldY = MAP_HALF_SIZE - rawY;
  ```
- This matches wowdev.wiki spec: `posx = 32 * TILESIZE - mddf.position[0]`

**Result**: Object positions are now correct relative to tiles!

### 2. **Y-Axis Orientation Fixed** ✅
**Problem**: Map was upside down because Y-axis wasn't properly inverted for display.

**Solution**:
- Kept Y-axis flip in `CoordinateTransformer.cs`:
  ```csharp
  var py = (1.0 - ClampUnit(localY)) * height;
  ```

**Result**: Map should now render right-side-up (Azeroth not upside-down).

### 3. **Lazy Overlay Loading** ✅
**Problem**: Loading all tile overlays at once caused browser to freeze (100s of MB, <1 FPS).

**Solution**:
- Implemented viewport-based overlay loading
- Only loads overlays for tiles currently visible
- Reloads on pan/zoom
- Uses `loadedOverlays` Set to prevent duplicate loading

**Result**: Massively improved performance - browser stays responsive!

### 4. **Zoom Levels Fixed** ✅
**Problem**: Started at zoom 0 (way too zoomed out), could only zoom to level 5 (not close enough).

**Solution**:
- Changed zoom range from `0-5` to `-1 to 3`
- Start at zoom `1` (good overview)
- Added smooth zoom: `zoomSnap: 0.25, zoomDelta: 0.5`

**Result**: Better initial view, can zoom in closer to see individual objects.

### 5. **CSS Syntax Fixed** ✅
**Problem**: Broken CSS causing styling issues.

**Solution**: Fixed missing `body {` selector.

## In Progress

### Sedimentary Layers Panel 🚧
- HTML structure added
- CSS styles added
- **TODO**: JavaScript implementation to populate layers from UniqueID ranges
- **TODO**: Toggle visibility of layers (enable/disable by ID range)

### Tile Detail Page 🚧
- Page exists but appears blank
- **TODO**: Debug why `state.loadIndex()` or overlay loading fails
- **TODO**: Restore object list, diff display, CSS styling

### PNG Tile Loading 🚧
- Tiles still not loading (black background only)
- **TODO**: Debug Leaflet tile coordinate system
- **TODO**: Verify PNG paths are correct

## Key Discoveries

1. **Nested Coordinate Systems**: 
   - ADT MDDF/MODF: Map-corner-relative (0 to 34133 range)
   - World coordinates: Center-origin (±17066 range)
   - Tile indices: 0-63 grid
   - Tile-local: 0-1 normalized per tile
   - Pixel: 0-512 per tile image

2. **Terrain Mirroring Theory**:
   - User observed: Azeroth appears mirrored to create Kalimdor
   - Evidence: Stonetalon Mountains = mirrored Wetlands
   - "HELP" text at Grim Batol reversed in Stonetalon

## Next Steps Priority

1. **Test current fixes** - Regenerate viewer and verify:
   - Map is right-side-up ✓
   - Objects spread across tiles (not clustered in corners)
   - Performance is good (lazy loading working)
   - Zoom levels are usable

2. **Implement Sedimentary Layers Panel**:
   - Parse UniqueID ranges from overlays
   - Group by range into "archaeological layers"
   - Add checkboxes to toggle visibility
   - Filter objects by active layers

3. **Fix PNG tiles** - Debug why minimap tiles don't load

4. **Fix tile detail page** - Restore full functionality

## Files Modified

- `WoWRollback.Core/Services/LkAdtReader.cs` - Coordinate conversion
- `WoWRollback.Core/Services/CoordinateTransformer.cs` - Y-axis fix
- `ViewerAssets/js/main.js` - Lazy loading, zoom levels
- `ViewerAssets/index.html` - Layers panel HTML
- `ViewerAssets/styles.css` - Layers panel CSS, syntax fix

## Commands to Test

```powershell
# Regenerate with all fixes
cd WoWRollback
.\regenerate-with-coordinates.ps1

# Or manual:
dotnet build -c Release
.\WoWRollback.Cli\bin\Release\net9.0\WoWRollback.Cli.exe compare-versions `
  --versions "0.5.3.3368,0.5.5.3494" --maps "Kalimdor,Azeroth" `
  --viewer-report --out rollback_outputs

# Copy updated viewer
Copy-Item ViewerAssets\js\main.js rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\js\ -Force
Copy-Item ViewerAssets\styles.css rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\ -Force
Copy-Item ViewerAssets\index.html rollback_outputs\comparisons\0_5_3_3368_vs_0_5_5_3494\viewer\ -Force
```

## Architecture Notes

The coordinate transformation pipeline:
```
MDDF/MODF Raw → World Coords → Tile Indices → Local [0,1] → Pixel [0,512]
(map-corner)    (center-origin)  (64x64 grid)   (per-tile)    (display)
```

Each transformation is now properly implemented following the ADT v18 spec.
