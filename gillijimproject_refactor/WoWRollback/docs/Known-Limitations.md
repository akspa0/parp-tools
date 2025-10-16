# Known Limitations - Alpha WDT Converter

## Current Status (Updated 2025-10-15)

✅ **Working**: Dungeon maps (e.g., RazorfenDowns) - **FULLY FUNCTIONAL!**  
⚠️ **Partial**: Outdoor maps (e.g., Kalidar) - Top-level loads, per-tile offset issue

---

## ✅ Fully Working Maps

### RazorfenDowns ✅
- **Type**: Dungeon
- **WMO Names**: 1 (from WDT + tiles)
- **Tiles**: 24
- **Result**: **FULLY LOADS IN 0.5.3 ALPHA CLIENT!**
- **Terrain**: ✅ Displays correctly
- **Character**: ✅ Loads and renders
- **Environment**: ✅ Skybox and effects work
- **Status**: Production ready!

---

## ⚠️ Partially Working Maps

### Kalidar ⚠️
- **Type**: Outdoor map
- **WMO Names**: 3 (from per-tile ADTs)
- **Tiles**: 55
- **Top-Level**: ✅ Loads successfully
- **Issue**: ❌ Per-tile MODF offset incorrect
- **Error**: `index (0x4D484452)` when loading tiles
- **Cause**: `MHDR.offsMob` pointing to wrong location

**Technical Details**:
- 0.6.0 Kalidar.wdt has `MWMO size=0` (no WMO names in WDT)
- WMO names are in individual `Kalidar_xx_yy.adt` files
- Our converter only reads terrain (MCNK) from ADTs, not objects
- Alpha client requires valid MONM/MODF structures even for terrain-only maps

---

## Root Cause

The converter is **terrain-only** and deliberately skips object data:
- ✅ Reads: MCNK terrain chunks
- ❌ Skips: MWMO/MODF (WMOs), MMDX/MDDF (M2s)

For maps where WMO data is in the **top-level WDT** (dungeons), this works because:
1. We read MWMO from WDT
2. We write MONM with correct names
3. We write empty MODF (no placements)
4. Client accepts this

For maps where WMO data is in **per-tile ADTs** (outdoor maps), this fails because:
1. WDT has no MWMO data
2. We write empty MONM
3. Client tries to read WMO data and gets garbage
4. Crash in `CMap::CreateMapObjDef`

---

## Workaround

**For testing**: Use dungeon maps that have WMO data in top-level WDT:
- RazorfenDowns ✅
- Other instance maps (likely to work)

**For outdoor maps**: Full object conversion required (future work)

---

## Future Work

To support outdoor maps like Kalidar, we need to:

### 1. Read Per-Tile Object Data
```
For each tile ADT:
  - Read MWMO chunk (WMO names)
  - Read MODF chunk (WMO placements)
  - Read MMDX chunk (M2 names)
  - Read MDDF chunk (M2 placements)
```

### 2. Merge Object Data
```
- Collect all unique WMO names from all tiles
- Collect all unique M2 names from all tiles
- Build global MONM/MDNM string tables
- Remap placement indices to global tables
```

### 3. Write Per-Tile Placements
```
For each Alpha tile:
  - Write MODF with WMO placements
  - Write MDDF with M2 placements
  - Update MHDR.offsMob/offsDoo
  - Update MHDR.sizeMob/sizeDoo
```

### 4. Handle Coordinate Conversion
```
- Convert WMO positions from LK to Alpha
- Convert M2 positions from LK to Alpha
- Handle rotation/scale differences
```

This is a significant amount of work beyond terrain-only conversion.

---

## Recommendation

For now, focus on:
1. ✅ Terrain conversion (working)
2. ✅ Dungeon maps with top-level WMOs (working)
3. ⏳ Object conversion (future enhancement)

The terrain-only converter successfully demonstrates the core WDT format conversion and fixes all critical bugs (MHDR.offsInfo, MCNK.radius, MCNK.mclqOffset, MPHD.nMapObjNames).

Full object support can be added incrementally as a separate feature.
