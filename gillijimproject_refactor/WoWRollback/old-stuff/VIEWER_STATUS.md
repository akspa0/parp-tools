# Viewer Status - What Works and What Doesn't

## ✅ Working

1. **Coordinate extraction** - World X/Y/Z are being extracted from converted LK ADTs
2. **CSV generation** - `assets_alpha_*.csv` files contain `world_x,world_y,world_z` columns
3. **Overlay JSON generation** - Viewer JSONs have world/local/pixel coordinates
4. **Object markers** - Blue dots appear on the map

## ❌ Broken

### 1. PNG Tiles Not Loading (CRITICAL)

**Issue**: Minimap PNG tiles don't render, only black background with blue dots.

**Root Cause**: Leaflet tile coordinate system mismatch.

**Current State**: Using `GridLayer` with `minNativeZoom: 6, maxNativeZoom: 6` but tiles still don't load.

**Next Steps**: Need to debug why `createTile()` isn't being called or why tiles aren't displaying.

### 2. Object Positioning Wrong

**Issue**: All objects on a tile appear at the same position (top-right corner).

**Symptoms**:
- Stratholme appears at bottom instead of top
- All objects clustered in corners
- Pixel coordinates not being honored

**Possible Causes**:
1. **Coordinate system mismatch** - ADT spec says:
   - Positive X = North
   - Positive Y = West
   - Origin at map center
   - Tile formula: `floor((32 - (axis / 533.33333)))`

2. **Y-axis flip** - WoW Y increases downward in-game but coordinates might be inverted

3. **Wrong tile assignment** - Objects may be assigned to wrong tiles, so their local coordinates are outside [0,1] range and get clamped

### 3. Tile Detail Page Broken

**Issue**: Clicking a tile opens a detail page with only tile name and minimap image, no object details.

**Expected**: Should show list of objects, diff info, version comparison.

## 🔍 Debugging Steps

### For PNG Tiles:

1. Open browser console
2. Check for `createTile()` calls
3. Check Network tab for PNG requests
4. Verify PNG files exist at expected paths

### For Object Positioning:

1. Pick a tile with objects (e.g., one with Stratholme)
2. Check overlay JSON for that tile
3. Verify world coordinates match ADT formula:
   ```
   tileCol = floor(32 - (worldX / 533.33333))
   tileRow = floor(32 - (worldY / 533.33333))
   ```
4. Check if objects are on CORRECT tile
5. If on wrong tile, local coords will be outside [0,1] and clamp to corners

## 📋 Priority Fixes

### P0 - Must Fix Now

1. **Get PNG tiles loading** - Without tiles, viewer is useless
2. **Fix object positioning** - Follow ADT spec exactly, no interpretation

### P1 - Important

3. **Verify tile assignment** - Objects must be on correct tile
4. **Fix coordinate transform** - May need to invert Y-axis
5. **Restore tile detail page** - Show object lists and diffs

### P2 - Nice to Have

6. **Optimize data size** - 100s of MB is too much, need clustering/LOD
7. **Remove redundant JSON fields** - File/folder info can be derived
8. **Add coordinate system debugging** - Show transforms visually

## 🎯 ADT Coordinate Spec (MUST FOLLOW)

From wowdev.wiki:

- Map is 64x64 tiles
- Each tile is 533.33333 yards
- Map center is (0, 0)
- Map bounds: ±17066.66656 yards
- **Positive X points NORTH**
- **Positive Y points WEST**
- Tile index: `floor((32 - (axis / 533.33333)))`

### Example:

For worldX = 20446.4, worldY = 13617.0:
```
tileCol = floor(32 - (20446.4 / 533.33333)) = floor(32 - 38.34) = floor(-6.34) = -7 ???
```

**WAIT** - That's negative! This means the coordinates are WRONG or the formula interpretation is wrong.

Let me recalculate following the spec EXACTLY:

If X+ = North and origin is at center:
- North edge (X = +17066) → tile 0
- South edge (X = -17066) → tile 63
- Formula: `tileRow = floor((32 - (X / 533.33333)))`

For X = +17066: `floor(32 - 32) = 0` ✓
For X = -17066: `floor(32 - (-32)) = 64` → clamped to 63 ✓

For X = 20446.4: `floor(32 - 38.34) = -6` → OUT OF BOUNDS!

**CONCLUSION**: The world coordinates being extracted are WRONG. They're outside the map bounds!

## 🚨 REAL PROBLEM FOUND

The world coordinates in the CSVs are **NOT in WoW world coordinate system**. They might be:
- Raw ADT-relative coordinates
- Different coordinate space
- Not properly transformed from LK ADT format

**Action Required**: Check `LkAdtReader.ReadMddf()` and `ReadModf()` to see what coordinate system they return.
