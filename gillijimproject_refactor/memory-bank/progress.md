# Progress

## ✅ Working

### MdxViewer (3D World Viewer)
- **Alpha WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **Standard WDT+ADT (3.3.5)**: ✅ Split ADT files from MPQ/IDataSource
- **WMO v14 rendering**: ✅ Correct geometry orientation (winding fix Feb 9)
- **MDX rendering**: ✅ Geometry + BLP textures, correct orientation
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

### MdxViewer Next Features
- **Liquid types**: Only ocean renders; rivers/lakes/magma/slime not yet visible
- **WMO interior liquid**: MLIQ chunk not yet parsed/rendered
- **MDX animations/bones**: No skeletal animation system yet
- **Lighting**: Basic hardcoded lighting only; no DBC-driven lights
- **Skybox**: Procedural gradient only; no game-data skyboxes

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW uses right-handed coords (X=North, Y=West, Z=Up) with Direct3D (CW winding)
- OpenGL uses CCW winding for front faces
- **Fix**: Reverse triangle winding at GPU upload (swap v1↔v2) + 180° Z rotation in placement
- Model vertices pass through raw — NO axis swap at vertex level
- Terrain positions: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`

### Alpha 0.5.3 MDX Archaeology
Alpha MDX `GEOS` sub-chunks use Tag(4)+Count(4)+Data layout. `UVAS` Count=1 in Version 1300 contains raw UV data directly. `BIDX` and `BWGT` chunks require peek-ahead validation for 1-byte vs 4-byte stride detection.
