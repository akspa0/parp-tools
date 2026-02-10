# Progress

## ✅ Working

### MdxViewer (3D World Viewer)
- **Alpha WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent
- **WMO doodad culling**: ✅ Distance (500u) + cap (64) + nearest-first sort + fog passthrough (Feb 10)
- **WMO liquid (MLIQ)**: ✅ GroupLiquid=15 → magma, type detection, positioning (Feb 10)
- **WMO transparent textures**: ✅ Alpha test/blend per BlendMode
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate
- **MDX rendering**: ✅ Two-pass opaque/transparent, alpha cutout for trees, fog skip for untextured
- **MDX GEOS parsing**: ✅ IsValidGeosetTag() peek-ahead prevents footer misread (Feb 10)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **MCLQ ocean liquid**: ✅ Inline liquid from MCNK header flags
- **Directional tile streaming**: ✅ Camera heading tracking, forward lookahead, priority-sorted queue (Feb 10)
- **Frustum culling**: ✅ View-frustum + distance + fade, relaxed MDX thresholds (Feb 10)
- **AreaID lookup**: ✅ Low 16-bit extraction + low byte fallback for MapID mismatch (Feb 10)
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors (Feb 10)
- **Replaceable Textures**: ✅ DBC CDI variant validation against MPQ + model dir scan fallback (Feb 10)
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

### MdxViewer — Rendering Quality & Lighting
- **MDX textures magenta**: ROOT CAUSE UNKNOWN — needs aggressive diagnostic logging
- **Terrain liquid type**: Lava still green — diagnostic logging added for mcnkFlags analysis
- **Water plane MDX rotation**: Flat water MDX models tilted wrong
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

### Reverted Changes (Caused Regressions)
- ❌ WMO fog skip for untextured fragments — broke WMO rendering entirely
- ❌ MDX rotation axis swap (X↔Y) — caused fence tilt issues
- ❌ MDX rotation negation — caused tree geometry to mirror/stretch into sky

## Key Technical Insights

### WMO Doodad Performance (Feb 10, 2026)
- WMO doodads were the primary rendering bottleneck — each rendered individually with own draw call
- Fix: Distance cull (500u), nearest-first sort, max 64 per WMO per frame, fog passthrough
- WmoRenderer.cs stores `LocalPosition` per doodad for fast world-space distance computation

### Alpha 0.5.3 AreaID Packing (Feb 10, 2026)
- MCNK `Unknown3` (offset 0x38) stores AreaID in low 16 bits
- High 16 bits may contain other data causing wrong AreaTable lookups
- Fallback: try low byte (0xFF mask) if 16-bit value doesn't match current MapID

### Terrain Liquid Type from MCNK Flags (Feb 10, 2026)
- Bits 4-5 of mcnkFlags encode liquid type: 0=water, 1=ocean, 2=magma, 3=slime
- Bit 3 (0x08) = ocean override flag
- Diagnostic logging added to verify flag values for lava areas

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`, `axis2 = heights[idx]`
- Tile visibility: bit 3 (0x08) = hidden (from noggit SMOLTile.liquid & 0x8)
- GroupLiquid=15 always → magma (old WMO "green lava" type)

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW: right-handed (X=North, Y=West, Z=Up), Direct3D CW winding
- OpenGL: CCW winding for front faces
- **Fix**: Reverse winding at GPU upload + 180° Z rotation in placement
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` — NO axis swap (swap was wrong)
- Model vertices pass through raw — NO axis swap at vertex level

### Replaceable Texture Resolution (Feb 10, 2026)
- Root cause: CreatureDisplayInfo has multiple entries per ModelID (e.g., Goblin model shared by Goblins + Fire Elementals)
- displayIndex=0 picked wrong CDI variant → wrong TextureVariation (e.g., LobstrokBlack for GoblinShredder)
- Fix: Try ALL CDI variants, validate each resolved texture exists in MPQ via FileExists/FindInFileSet
- If no DBC variant validates, fall through to model directory scan (Strategy 2)
- Key: `ReplaceableTextureResolver.SetDataSource()` wired in ViewerApp.cs

### Alpha 0.5.3 MDX Archaeology
Alpha MDX `GEOS` sub-chunks use Tag(4)+Count(4)+Data layout. `UVAS` Count=1 in Version 1300 contains raw UV data directly. `BIDX` and `BWGT` chunks require peek-ahead validation for 1-byte vs 4-byte stride detection.
