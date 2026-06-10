# Format Unknowns & Research Targets

Prioritized list of unresolved format questions that need Ghidra investigation.

---

## Priority 1 — Blocking Current Work

### U-001: Alpha MCNK Header Full Layout
**Status**: Partially resolved — class structure documented
**Affects**: Terrain parsing correctness
**What we know** (from Ghidra 0.5.3):
- MCNK magic: `0x4D434E4B` (forward byte order, not reversed)
- `CMapChunk` class with 145 vertices/normals (81 outer + 64 inner)
- File format is monolithic WDT with inline MCNK chunks
- SMChunk header starts at offset +8 from chunk magic (after 4-byte magic + 4-byte size)
- MCLY subchunk follows MCNK header at ~+1164 bytes
- Header includes `sizeShadow` field for texture calculations
- Located primary parser: `CMapChunk::SyncLoad` @ 0x00698d90
**Still unknown**: Complete SMChunk field-by-field layout, exact header size, all flag meanings, absolute vs relative offset fields
**Ghidra target**: Decompile additional field accessors in CMapChunk methods to complete SMChunk structure
**See also**: [`specifications/outputs/053/task-01-mcnk-header.md`](specifications/outputs/053/task-01-mcnk-header.md)

### U-002: Alpha WDT MAIN Chunk Entry Layout
**Status**: Working but needs verification
**Affects**: Tile enumeration
**What we know**: Column-major indexing (tileX*64+tileY), entries contain ADT file offsets
**Unknown**: Exact entry size and all fields. Are there flags beyond the offset?
**Ghidra target**: 0.5.3 — find WDT MAIN reader, check entry structure.

### U-003: MCAL 4-bit Alpha Pixel Ordering
**Status**: ✅ RESOLVED — Row-major confirmed
**Affects**: Texture blending quality
**Resolution**: Ghidra analysis of `CMapChunk::UnpackAlphaBits` @ `0x0069a621` shows a single linear loop from 0→4095 (0x1000). This is row-major: pixels 0-63 = row 0, 64-127 = row 1, etc. Our viewer's `AlphaTerrainAdapter.ExtractAlphaMaps` linear expansion is CORRECT. The VLM exporter's `AlphaMapService` column-major indexing is WRONG and should be fixed.
**Bonus**: Discovered `CWorld::alphaMipLevel` — mip level 0 = 64×64 (2048 bytes packed), mip level 1 = 32×32 (reads from 33-wide source buffer with +0x21 stride).

### U-004: WMO v14 Geometry Handedness
**Status**: Renders mirrored
**Affects**: WMO orientation (TODO #13)
**What we know**: We apply Scale(1,-1,1) to correct the X↔Y coordinate swap, but WMOs still appear mirrored.
**Unknown**: Does the WMO v14 format store geometry in a different handedness than terrain? Is there an additional mirror needed for WMO local-space vertices?
**Ghidra target**: 0.5.3 — find WMO rendering path, check vertex transform matrices.

### U-005: MCLQ Liquid Data Layout
**Status**: Partially resolved — class structure known, chunk parser still needed
**Affects**: Liquid rendering (TODOs #10, #11)
**What we know** (from Ghidra 0.5.3):
- `CChunkLiquid` class size = 0x338 (824 bytes)
- 4 liquid types: Water(0), Ocean(1), Magma(2), Slime(3) — `type < LQ_LAST` where LQ_LAST=4
- Height range: `CFloatRange height` (min/max floats)
- Flow vectors: `SWFlowv flowvs[2]`
- Up to 4 liquid instances per chunk (`liquids[4]` in `CMapChunk`)
- MCLQ = terrain liquid, MLIQ = WMO liquid (separate systems)
- Estimated 9×9 vertex grid (~9 bytes/vertex = float height + byte flags)
**Still unknown**: Exact MCLQ chunk binary layout, vertex data format, flag meanings
**Ghidra target**: 0.5.3 — search for 0x4D434C51 constant to find MCLQ chunk parser directly.
**Key functions found**: `AllocChunkLiquid`@0x691860, `AddChunkLiquid`@0x66b120, `RenderLiquid_0`@0x69e4b0, `PrepareRenderLiquid`@0x66a590, `QueryLiquidStatus`@0x664e70, `GetLiquidTexture`@0x6736b0

---

## Priority 2 — Needed for Format Conversion

### U-006: Alpha MDDF/MODF Field Differences vs LK
**Status**: Mostly known
**Affects**: Format conversion accuracy
**What we know**: Alpha uses 36-byte MDDF (same as LK) and 64-byte MODF. Scale is uint16 / 1024.
**Unknown**: Are there any Alpha-specific flags or fields that differ from LK? The padding bytes at end of MODF — are they always zero?
**Ghidra target**: 0.5.3 vs 3.3.5 — compare MDDF/MODF reader functions.

### U-007: Alpha MDX Format Complete Spec
**Status**: ✅ RESOLVED — Complete loading pipeline documented
**Affects**: MDX↔M2 conversion fidelity
**Resolution**: Ghidra analysis of `BuildModelFromMdxData` @ `0x00421fb0` reveals complete 15-step loading sequence:
1. Global properties (VERS/MODL)
2. Textures (TEXS) - null-terminated path strings  
3. Materials (MTLS)
4. Geosets (GEOS) - geometry data
5. Attachments (ATCH) - attachment points for effects/weapons
6. Animation (BONE/SEQS) - bones + keyframe animation
7. Ribbon emitters (RIBB) - trail effects
8. Particle emitters (PRE2) - Warcraft 3-style particles
9. Bone matrices
10. Hit test data (collision shapes)
11. Lights (LITE)
12. Collision shapes (CLID)
13. Bounding extents
14. Pivot points (PIVT)
15. Cameras (CAMS)
**Model flags**: 0x20=complex_model, 0x100=no_anim, 0x200=no_lights, 0x20=hit_test, 0x80000000=full_alpha
**Bonus**: ASCII MDL format also supported (same structure, text representation) via `BuildModelFromMdlData` @ 0x004235b0
**See also**: [`specifications/outputs/053/task-05-mdx-model-loading.md`](specifications/outputs/053/task-05-mdx-model-loading.md)

### U-008: Alpha WMO v14 Complete Chunk Inventory
**Status**: Partially resolved — class structure documented
**Affects**: WMO v14↔v17 conversion
**What we know** (from Ghidra 0.5.3):
- WMO v14 is monolithic single file (not split like v17+)
- `CMapObj` supports up to 384 groups (0x180 max)
- Group management via `TSList<CMapObjGroup>` + `TSFixedArray` pointers
- `CMapObjDef` stores placement data (MODF - position, rotation, scale, bounds)
- `CMapObjGroup` handles group data (MOGP - vertices, normals, UVs, triangles, BSP, materials)
- MLIQ chunks for WMO liquids (separate from terrain MCLQ) @ 0x008a2930
- Forward FourCC byte order (MLIQ = 0x4D4C4951)
- Ambient color + AABB bounding box stored in `CMapObj`
- Key functions: `CMapObj` @ 0x00693190, `CMapObjGroup` @ 0x0068b610, `CMapObjDef` @ 0x006ac280
**Still unknown**: Doodad set placement format, complete BSP tree layout, fog/lighting chunk details, vertex transformation pipeline, handedness corrections
**Ghidra target**: Find WMO file loader, trace MOHD/MOGP parsing, examine vertex transformation code
**See also**: [`specifications/outputs/053/task-04-wmo-v14-rendering.md`](specifications/outputs/053/task-04-wmo-v14-rendering.md)

### U-009: Alpha BLP Texture Format Differences
**Status**: ✅ RESOLVED — Standard BLP1, no Alpha-specific variants
**Affects**: Texture conversion
**Resolution**: Ghidra analysis of `LoadBlpMips` @ `0x0046f820` confirms standard BLP1 format:
- Magic: 0x31504C42 ("BLP1")
- Color encodings: JPEG(0), Palette(1), DXT(2), ARGB8888(3)
- DXT1/DXT3/DXT5 compression supported
- Alpha depths: 0, 1, 4, 8 bits
- Up to 16 mipmap levels with box filtering
- Hardware fallback to ARGB1555/ARGB4444/RGB565 if DXT unsupported
- Source: `D:\build\buildWoW\engine\Source\BLPFile\blp.cpp`
- Our SereniaBLPLib usage is correct for Alpha textures.

---

## Priority 3 — PM4 and Advanced

### U-010: PM4 Coordinate System Offset
**Status**: Known to be wrong
**Affects**: PM4 tile viewer overlay
**What we know**: PM4 coordinates are close to ADT world coordinates but have a systematic offset.
**Unknown**: The exact transform from PM4 vertex positions to WoW world coordinates. The offset direction and magnitude.
**Ghidra target**: N/A (PM4 is server-side, not in client binaries). Use Pm4AdtCorrelator with known development map data.

### U-011: PM4 MSLK Object Hierarchy
**Status**: Partially decoded
**Affects**: PM4 object extraction
**What we know**: MSLK entries have ObjectTypeFlags, ParentIndex, geometry references. Values 1-18 for type flags.
**Unknown**: Complete meaning of all ObjectTypeFlags values. How ParentIndex chains relate to physical object boundaries.
**Ghidra target**: N/A (server-side). Use statistical analysis across PM4 files.

### U-012: MCNK MCCV Vertex Colors
**Status**: Parsed but not rendered
**Affects**: Terrain rendering fidelity
**What we know**: MCCV stores per-vertex BGRA colors for terrain tinting.
**Unknown**: Exact blending mode — is it multiplicative or additive? Applied before or after texture blending?
**Ghidra target**: 0.5.3 — find terrain rendering shader/path, check MCCV application.

### U-013: 0.5.5 / 0.6.0 Format Deltas
**Status**: Unknown
**Affects**: Multi-version support
**What we know**: 0.5.5 and 0.6.0 are transitional builds. Some format changes happened.
**Unknown**: Which specific chunks changed between 0.5.3, 0.5.5, and 0.6.0?
**Ghidra target**: 0.5.5 and 0.6.0 — diff chunk reader functions against 0.5.3.

---

## Discovery Log

Track findings as they're made:

| ID | Date | Finding | Source |
|----|------|---------|--------|
| D-001 | 2026-02-08 | MCAL alpha maps are row-major (linear loop 0→4095) | `CMapChunk::UnpackAlphaBits` @ 0x0069a621 |
| D-002 | 2026-02-08 | Alpha mip level system: level 0=64×64, level 1=32×32 | `CWorld::alphaMipLevel` global |
| D-003 | 2026-02-08 | CChunkLiquid class = 0x338 bytes, 4 liquid types, up to 4 per chunk | `CMap::AllocChunkLiquid` @ 0x00691860 |
| D-004 | 2026-02-08 | WMO v14 max 384 groups (0x180), MLIQ separate from MCLQ | `CMapObj` constructor @ 0x00693190 |
| D-005 | 2026-02-08 | MDX loading: 15-step pipeline, simple+complex paths, all chunks identified | `BuildModelFromMdxData` @ 0x00421fb0 |
| D-006 | 2026-02-08 | BLP1 standard format, DXT1/3/5, 0/1/4/8-bit alpha, 16 mip levels | `LoadBlpMips` @ 0x0046f820 |
| D-007 | 2026-02-08 | ASCII MDL format also supported (WC3 text format) | `BuildSimpleModelFromMdlData` @ 0x00424370 |
| D-008 | 2026-02-08 | MCNK verification uses forward FourCC (0x4D434E4B) | `CMapChunk::SyncLoad` @ 0x00698d90 |
| D-009 | 2026-02-09 | MCNK header at +8 offset, MCLY follows at ~+1164, monolithic WDT format | `CMapChunk::SyncLoad` @ 0x00698d90 |
| D-010 | 2026-02-09 | CMapChunk has 145 verts/normals (81+64), 256 planes, up to 4 liquid instances | `CMapChunk` constructor @ 0x00698510 |
| D-011 | 2026-02-09 | Liquid texture paths: XTextures/{river,slime,lava,ocean}/pattern.%d.blp | String refs @ 0x0089f0d0-14c |
| D-012 | 2026-02-09 | WMO v14 max 384 groups, forward FourCC, MLIQ for WMO liquids | `CMapObj` @ 0x00693190 |
| D-013 | 2026-02-09 | MDX model flags: 0x20=complex, 0x100=no_anim, 0x200=no_lights, 0x80000000=full_alpha | `BuildModelFromMdxData` @ 0x00421fb0 |
| D-014 | 2026-02-09 | BLP hardware fallback: DXT1→RGB565/ARGB1555, DXT3/5→ARGB4444 if DXT unsupported | `LoadBlpMips` @ 0x0046f820 |
