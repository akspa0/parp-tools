# Progress

## ✅ Working

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working (DisplayInfo + Extra).
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **DBC Service**: ✅ Automates variation mapping for Alpha archaeology.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK, works 100%.
- **WMO v14/v17**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).
- **V8 Binary Export**: ✅ `.bin` format implemented.

## ⚠️ Partial / In Progress

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer. Mapping MDX sequences to M2 animations.

### LK 3.3.5 / Cata 4.0.0 ADT Processing

| Component | Status | Notes |
|-----------|--------|-------|
| Minimap TRS | ✅ FIXED | Column order + coordinate padding |
| JSON height_min/max | ✅ FIXED | MCIN-based parsing working |
| JSON heights[] array | ✅ FIXED | 256 chunks populated |
| Heightmap PNG | 🔧 FIX APPLIED | Removed posZ addition - untested |

## ❌ Broken

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| 0.5.3 Alpha MDX | ✅ Working | Geometry, UVs, and Skins (DBC) resolved correctly |
| OBJ Split Export | ✅ Working | Verified with fat textures and creature variations |
| LK/Cata ADT | ✅ Working | Heights correctly extracted via MCIN |

## Key Technical Insight

**Alpha 0.5.3 MDX Archaeology:**
Unlike Retail/M2 formats, Alpha MDX `GEOS` sub-chunks (VRTX, TVRT, etc) are often separated by variable null padding. Robust parsing requires scanning for the next UTF-8 chunk tag rather than relying on fixed offsets. Additionally, `UVAS` (TVRT) data in Version 1300 is stored as raw float pairs immediately following the Count field, differing from standard WC3/Later-WoW specs.
