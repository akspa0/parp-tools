# Viewer Critical Fixes Needed

## Status: INCOMPLETE - Core Functionality Missing

### 1. Missing Placement Coordinates (CRITICAL)
**Problem**: AlphaWDTAnalyzer doesn't extract world coordinates from MODF/MDDF chunks.

**Current State**:
- `AlphaWdtAnalyzer.Core.PlacementRecord` has: Type, AssetPath, MapName, TileX, TileY, UniqueId
- **Missing**: WorldX, WorldY, WorldZ (actual placement positions)
- WoWRollback fills these with 0.0f placeholders (line 55-57 in AlphaWdtAnalyzer.cs)

**Impact**:
- No objects appear on minimap overlays (all at 0,0,0)
- Diff system cannot detect "moved" objects
- Viewer shows 0 objects because coordinates are invalid

**Fix Required**:
1. Extend `AlphaWdtAnalyzer.Core.AdtScanner` to read MODF/MDDF placement data:
   - MODF (WMO placements): Position (X,Y,Z), Rotation, NameId, UniqueId
   - MDDF (M2 placements): Position (X,Y,Z), Rotation, NameId, UniqueId
2. Add Position fields to `PlacementRecord`
3. Update WoWRollback's `AlphaWdtAnalyzer.cs` to use real coordinates

**Files to Modify**:
- `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/Models.cs` - Add position to PlacementRecord
- `AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/AdtScanner.cs` - Read MODF/MDDF chunks
- `WoWRollback/WoWRollback.Core/Services/AlphaWdtAnalyzer.cs` - Remove placeholder 0.0f values

**Reference Implementation**:
- `src/gillijimproject-csharp/WowFiles/Mddf.cs` - Has MDDF chunk reader
- `src/gillijimproject-csharp/WowFiles/Modf.cs` - Has MODF chunk reader

---

### 2. Visual Minimap Grid (UX Issue)
**Problem**: Viewer shows text boxes instead of visual minimap grid.

**Current State**:
- Tiles displayed as text cards: "30_30 | 2 version(s)"
- User expected: Full continent map with clickable minimap tiles

**Fix Required**:
1. Load and display minimap images in grid layout
2. Tiles should show actual minimap PNGs, not text
3. Click on visual tile to open detail view

**Implementation**:
- Modify `main.js` `createTileCard()` to include `<img>` tag
- Load minimap from per-version path: `minimap/{version}/{map}/{map}_{col}_{row}.png`
- CSS grid layout to maintain tile positions

---

### 3. Diff System Not Working
**Problem**: Diff functionality exists but doesn't display properly.

**Current State**:
- `OverlayDiffBuilder.cs` generates diff JSON
- Tile viewer has diff UI controls
- But objects aren't visible due to #1 (missing coordinates)

**Fix Dependencies**:
- Requires #1 to be fixed first (need real coordinates for diff detection)
- Movement threshold detection needs valid X,Y,Z values

---

### 4. Overlay Data Structure Mismatch (FIXED)
**Status**: ✓ Fixed in latest tile.js

**Was**: JavaScript expected `layers[].objects`  
**Actually**: JSON has `layers[].kinds[].points`  
**Fix**: Updated `tile.js` line 152 to use `flatMap(kind => kind.points)`

---

## Priority Order

1. **FIRST**: Fix placement coordinate extraction (#1)
   - Without this, viewer shows 0 objects and is non-functional
   - Requires modifying AlphaWDTAnalysisTool to read MODF/MDDF

2. **SECOND**: Visual minimap grid (#2)
   - UX improvement once data is working

3. **THIRD**: Verify diff system (#3)
   - Will work once #1 is fixed

---

## Immediate Next Steps

1. Read MODF/MDDF chunk format from Alpha ADT files
2. Extend AlphaWdtAnalyzer.Core to extract positions
3. Test with single tile to verify coordinates
4. Regenerate viewer data with real coordinates
5. Verify objects appear in tile detail view
