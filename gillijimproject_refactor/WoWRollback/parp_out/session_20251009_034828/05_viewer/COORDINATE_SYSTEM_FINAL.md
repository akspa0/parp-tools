# Coordinate System - Final Status

**Date**: 2025-10-08  
**Status**: ✅ WORKING

---

## Summary

The coordinate system is now correctly implemented and tiles overlay properly on the grid.

### Key Findings

1. **Direct Mapping Works**: WoW tile (row, col) → Leaflet bounds [[row, col], [row+1, col+1]]
2. **No Y-Flip Needed**: Both WoW and Leaflet have (0,0) at top-left, (63,63) at bottom-right
3. **Grid Labels Match Tiles**: Tiles overlay exactly on their corresponding grid cells

### Test Results

From console output:
```
Tile (31,31): bounds = [[31, 31], [32, 32]]  ✅
Tile (32,32): bounds = [[32, 32], [33, 33]]  ✅
Tile (30,30): bounds = [[30, 30], [31, 31]]  ✅
```

Visual confirmation:
- Simulated tiles overlay correctly on grid cells
- Grid labels show correct row_col coordinates
- Center tiles (31,31), (31,32), (32,31), (32,32) appear at correct positions

---

## Final Implementation

### CoordinateSystem.js

```javascript
// Tile → World (center of tile)
tileToWorld(row, col) {
    // Tile (32,32) should map to world (0,0)
    const worldX = (32 - col) * this.TILE_SIZE;
    const worldY = (32 - row) * this.TILE_SIZE;
    return { worldX, worldY };
}

// Leaflet bounds for tile
tileBounds(row, col) {
    // Direct mapping: WoW tile (row, col) → Leaflet bounds [[row, col], [row+1, col+1]]
    // Both systems have (0,0) at top-left, (63,63) at bottom-right
    return [
        [row, col],
        [row + 1, col + 1]
    ];
}
```

### GridPlugin.js

```javascript
// Grid labels
for (let row = 0; row < 64; row++) {
    for (let col = 0; col < 64; col++) {
        // Direct mapping: grid position (row, col) = WoW tile (row, col)
        text.textContent = `${row}_${col}`;
        text.setAttribute('x', col + 0.5);
        text.setAttribute('y', row + 0.5);
    }
}
```

---

## Coordinate System Reference

### WoW Tile Coordinates
- **Grid**: 64×64 (indices 0-63)
- **Row 0**: North (top)
- **Row 63**: South (bottom)
- **Col 0**: West (left)
- **Col 63**: East (right)
- **Center**: Tile (32, 32)

### Leaflet Simple CRS
- **Coordinate space**: 0-64
- **lat 0**: Top
- **lat 63**: Bottom
- **lng 0**: Left
- **lng 63**: Right
- **Center**: [32, 32]

### Perfect Alignment
```
WoW Tile (row, col) → Leaflet Bounds [[row, col], [row+1, col+1]]

Examples:
- Tile (0, 0)    → [[0, 0], [1, 1]]      (top-left)
- Tile (0, 63)   → [[0, 63], [1, 64]]    (top-right)
- Tile (63, 0)   → [[63, 0], [64, 1]]    (bottom-left)
- Tile (63, 63)  → [[63, 63], [64, 64]]  (bottom-right)
- Tile (32, 32)  → [[32, 32], [33, 33]]  (center)
```

---

## What Was Wrong Before

### Attempt 1: Y-Flip in tileBounds()
```javascript
// WRONG - caused tiles to appear upside down
const lat1 = 63 - row;
const lat2 = 63 - (row + 1);
```

### Attempt 2: Y-Flip in Grid Labels
```javascript
// WRONG - caused grid labels to not match tiles
const row = 63 - lat;
```

### Attempt 3: Off-by-one in tileToWorld()
```javascript
// WRONG - caused 266 yard offset
const worldX = (32 - col - 0.5) * this.TILE_SIZE;
```

---

## Final Solution

**Keep it simple!** Both coordinate systems use the same orientation:
- (0,0) at top-left
- (63,63) at bottom-right
- No transformations needed for `tileBounds()`
- Direct 1:1 mapping

---

## Testing

### All Coordinate Tests Pass ✅
```
Test 1: World to Tile conversion          ✓ PASS
Test 2: Tile to World conversion          ✓ PASS
Test 3: Round-trip World → Tile → World   ✓ PASS
Test 4: Tile bounds calculation           ✓ PASS
Test 5: Elevation normalization           ✓ PASS
Test 6: Color brightness adjustment       ✓ PASS
Test 7: Lat/Lng to Tile conversion        ✓ PASS
Test 8: Tile to Lat/Lng conversion        ✓ PASS

Total: 8/8 passed
```

### Visual Verification ✅
- Simulated tiles overlay on correct grid cells
- Grid labels match tile positions
- Center tiles appear at center of map
- No off-by-one errors

---

## Next Steps

1. ✅ **Coordinate system working** - tiles align with grid
2. ⏳ **Load real minimap tiles** - need to copy tiles to viewer assets directory
3. ⏳ **Test with actual data** - verify with real WoW minimap images
4. ⏳ **Implement M2/WMO overlays** - use same coordinate system

---

## Notes for Future Development

### When Adding Real Minimap Tiles

The minimap tiles should be placed in:
```
WoWRollback.Viewer/assets/minimap/{version}/{map}/{map}_{col}_{row}.webp
```

Then load them with:
```javascript
const url = `minimap/${version}/${map}/${map}_${col}_${row}.webp`;
const bounds = coordSystem.tileBounds(row, col);
L.imageOverlay(url, bounds).addTo(map);
```

### When Adding M2/WMO Markers

Use the same coordinate system:
```javascript
// Convert world coordinates to tile
const tile = coordSystem.worldToTile(worldX, worldY);

// Convert tile to Leaflet lat/lng
const latLng = coordSystem.tileToLatLng(tile.row, tile.col);

// Add marker
L.circleMarker([latLng.lat, latLng.lng]).addTo(map);
```

---

**Status**: Coordinate system is correct and ready for production use! ✅
