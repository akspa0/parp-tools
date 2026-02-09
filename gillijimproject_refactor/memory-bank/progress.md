# Progress

## ✅ Working

### MdxViewer (3D World Viewer)
- **Alpha WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **Standard WDT+ADT (3.3.5)**: ✅ Split ADT files from MPQ/IDataSource
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent (Feb 9)
- **WMO liquid (MLIQ)**: ✅ Type detection, 90° CCW rotation fix, tile visibility (Feb 9)
- **WMO transparent textures**: ✅ Alpha test/blend per BlendMode (Feb 9)
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate (Feb 9)
- **MDX rendering**: ✅ Two-pass opaque/transparent, blend modes 0-6, correct orientation
- **MDX GEOS parsing**: ✅ BIDX/BWGT peek-ahead validation (Feb 9)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **MCLQ ocean liquid**: ✅ Inline liquid from MCNK header flags
- **Async tile streaming**: ✅ AOI-based lazy loading with background threads
- **Frustum culling**: ✅ View-frustum + bounding box culling
- **Minimap overlay**: ✅ From minimap tile images

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working.
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK.
- **WMO v14/v17 converter**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).

## ⚠️ Partial / In Progress

### MdxViewer — MDX Rendering Quality
- **MDX alpha discard**: Wrong — uses boolean 0/1 instead of proper thresholds (0.75 for AlphaKey, 1/255 for transparent)
- **MDX per-geoset color/alpha**: Only static alpha used; animated GeosetAnims not wired
- **MDX particles/ribbons**: Not implemented
- **MDX texture UV animation**: Not implemented
- **MDX billboard bones**: Not implemented
- **WMO lighting**: v14-16 grayscale lightmap + v17 MOCV vertex colors not implemented

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- Our renderer uses raw file coords with Z-up (Camera up = Vector3.UnitZ)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`, `axis2 = heights[idx]`
- Tile visibility: bit 3 (0x08) = hidden (from noggit SMOLTile.liquid & 0x8)
- Liquid type: `(groupLiquid - 1) & 3` for basic type

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW uses right-handed coords (X=North, Y=West, Z=Up) with Direct3D (CW winding)
- OpenGL uses CCW winding for front faces
- **Fix**: Reverse triangle winding at GPU upload (swap v1↔v2) + 180° Z rotation in placement
- Model vertices pass through raw — NO axis swap at vertex level

### Alpha 0.5.3 MDX Archaeology
Alpha MDX `GEOS` sub-chunks use Tag(4)+Count(4)+Data layout. `UVAS` Count=1 in Version 1300 contains raw UV data directly. `BIDX` and `BWGT` chunks require peek-ahead validation for 1-byte vs 4-byte stride detection.
