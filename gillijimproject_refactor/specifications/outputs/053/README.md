# WoW Alpha 0.5.3 (Build 3368) - Ghidra Reverse Engineering Results

## Overview

This directory contains the results of reverse engineering analysis of WoWClient.exe from Alpha 0.5.3 (build 3368) using Ghidra. This is the only WoW build with a PDB file available, making it invaluable for understanding early WoW formats.

**Binary**: WoWClient.exe (Alpha 0.5.3 build 3368)  
**Architecture**: x86 (32-bit)  
**PDB**: Available (with function names and type information)

## Research Tasks Completed

### Unknown Format Investigations

#### [U-001: Alpha MCNK Complete Header](u001-mcnk-complete-header.md)
**Status**: High Confidence (Complete)

Complete field-by-field MCNK header layout extracted from CMapChunk::Create.

**Key Findings**:
- Header size: 136 bytes (0x88)
- MCLY follows at +0x48C (1164 bytes)
- Flags identify liquid types (bits 2-5)
- All offsets are relative to MCNK start
- Position stored at +0x88 as 3 floats

#### [U-002: WDT MAIN Entry Layout](u002-wdt-main-entry-layout.md)
**Status**: High Confidence (Complete)

WDT MAIN chunk entry structure and monolithic format documented.

**Key Findings**:
- Entry size: 16 bytes (4096 entries × 16 = 65536 bytes)
- Column-major indexing: `entries[tileX * 64 + tileY]`
- Fields: flags, asyncId, offset, size
- Monolithic format with inline MCNK chunks
- MPHD header: 128 bytes, MODF: 64 bytes per entry

#### [U-004: WMO v14 Handedness Transform](u004-wmo-handedness-transform.md)
**Status**: Medium-High Confidence (Partial)

WMO transformation pipeline analyzed—no mirroring in transforms.

**Key Findings**:
- Standard right-handed coordinate system
- No axis swaps or negative scales in transform code
- Rotation around Z-axis (vertical)
- Mirroring likely in geometry data or render state, not transforms
- Transform matrix at offset +0x90, inverse at +0xD0

#### [U-005: MCLQ Complete Format](u005-mclq-complete-format.md)
**Status**: High Confidence (Complete)

Complete MCLQ inline liquid data format.

**Key Findings**:
- Inline storage (not a separate chunk)
- 804 bytes (0x324) per liquid instance
- 9×9 vertex grid (81 vertices × 8 bytes)
- 4×4 tile grid (16 floats)
- 2 flow vector structures (40 bytes each)
- Type determined by MCNK flag bits 2-5

#### [U-006: MDDF/MODF Alpha Format](u006-mddf-modf-alpha-format.md)
**Status**: High Confidence (Complete)

MDDF and MODF formats are identical to LK—no Alpha-specific differences.

**Key Findings**:
- MDDF: 36 bytes (same as LK)
- MODF: 64 bytes (same as LK)
- Scale: uint16 / 1024
- Rotation: C3Vector (3 floats) - Euler angles
- Padding: 2 bytes at end of MODF, always zero
- Direct binary compatibility with LK

## Research Tasks Completed

### [Task 1: MCNK Header Structure](task-01-mcnk-header.md)
**Status**: Medium Confidence (Partial)

Documented the MCNK (Map Chunk) header structure used in Alpha's monolithic WDT format.

**Key Findings**:
- MCNK magic: `0x4D434E4B` (forward byte order)
- Class structure: `CMapChunk` with 145 vertices/normals
- Confirmed monolithic WDT format with inline terrain chunks
- Located primary parser: `CMapChunk::SyncLoad` @ 0x00698d90

**Functions Analyzed**:
- `CMapChunk::CMapChunk` @ 0x00698510 (constructor)
- `CMapChunk::SyncLoad` @ 0x00698d90 (MCNK parser)

### [Task 2: MCAL Alpha Map Pixel Order](task-02-mcal-pixel-order.md)
**Status**: High Confidence (Complete)

Determined the pixel ordering for 4-bit packed alpha maps in MCAL chunks.

**Key Findings**:
- ✅ **Row-major order confirmed**
- 4096 pixels (64×64) stored linearly: rows then columns
- 2048 bytes total (2 4-bit values per byte)
- Single linear loop (not nested row/column), but semantically row-major

**Functions Analyzed**:
- `CMapChunk::UnpackAlphaBits` @ 0x0069a621 (alpha unpacking)

### [Task 3: MCLQ Liquid Data Structure](task-03-mclq-liquid-structure.md)
**Status**: Medium Confidence (Partial)

Documented the MCLQ liquid chunk structure for terrain water/lava/slime.

**Key Findings**:
- `CChunkLiquid` class size: 0x338 bytes (824 bytes)
- 4 liquid types: Water, Ocean, Magma, Slime
- Height range stored as min/max floats
- Likely 9×9 vertex grid (~9 bytes per vertex)
- Distinction between MCLQ (terrain) and MLIQ (WMO)

**Functions Analyzed**:
- `CMap::AllocChunkLiquid` @ 0x00691860 (allocation)
- `CWorldScene::AddChunkLiquid` @ 0x0066b120 (scene management)

### [Task 4: WMO v14 Rendering Path](task-04-wmo-v14-rendering.md)
**Status**: Medium Confidence (Partial)

Analyzed WMO (World Map Object) v14 structure and rendering pipeline.

**Key Findings**:
- WMO v14 is monolithic (single .wmo file, not split like v17+)
- Supports up to 384 groups (0x180) per WMO
- MLIQ chunks for WMO liquids (vs MCLQ for terrain)
- Forward FourCC byte order confirmed

**Functions Analyzed**:
- `CMapObj` @ 0x00693190 (WMO constructor)
- `CMapObjGroup` @ 0x0068b610 (group handler)
- `CMapObjDef` @ 0x006ac280 (placement definition)

### [Task 5: MDX Model Loading](task-05-mdx-model-loading.md)
**Status**: High Confidence (Complete)

Fully documented the MDX (Warcraft 3-style) model format loading pipeline.

**Key Findings**:
- Complete 15-step loading sequence identified
- All MDX chunks documented: VERS, MODL, TEXS, MTLS, GEOS, ATCH, BONE, RIBB, PRE2, LITE, CLID, PIVT, CAMS
- Flag-based loading (animations, lights, hit test data)
- Particle emitters (Type 2) and ribbon emitters
- Bone/skeleton system with keyframe animation

**Functions Analyzed**:
- `BuildModelFromMdxData` @ 0x00421fb0 (primary loader)
- 15+ chunk reader functions identified

### [Task 6: BLP Texture Loading](task-06-blp-texture-loading.md)
**Status**: High Confidence (Complete)

Verified BLP1 texture format handling with complete format detection.

**Key Findings**:
- BLP1 format confirmed (magic: `0x31504C42`)
- DXT1/3/5 compression support
- Hardware fallback logic for non-DXT GPUs
- Alpha depths: 0, 1, 4, 8 bits
- Up to 16 mipmap levels
- Complete format conversion table documented

**Functions Analyzed**:
- `LoadBlpMips` @ 0x0046f820 (primary loader)
- `CreateBlpTexture` @ 0x004717f0 (texture creation)

## Key Format Differences from Later WoW

| Feature | Alpha 0.5.3 | Later WoW (LK+) |
|---------|-------------|-----------------|
| **WDT Format** | Monolithic (ADT data inline) | Separate ADT files |
| **FourCC Order** | Forward (MCNK = 0x4D434E4B) | Reversed (KNCM) |
| **WMO Version** | v14 (monolithic .wmo) | v17+ (split root + groups) |
| **Model Format** | MDX (Warcraft 3) | M2 (WoW-specific) |
| **Vertex Order** | Non-interleaved (81 outer + 64 inner) | Interleaved |
| **WDT Indexing** | Column-major (tileX*64+tileY) | Row-major |
| **Liquid Chunks** | MCLQ (simple, 4 types) | MH2O (complex, expanded types) |
| **BLP Format** | BLP1 | BLP1 → BLP2 (Cata+) |

## Coordinate System

WoW uses:
- **X = North**
- **Y = West** 
- **Z = Up**

File positions stored as **(X, Z, Y)** — Z (height) in the middle.

Constants:
- `MapOrigin = 17066.66666` (world coord of tile 0,0)
- `ChunkSize = 533.33333` (world units per tile)

## Usage

Each task document follows this structure:

1. **Overview** - Summary of the analysis
2. **Key Findings** - Main discoveries
3. **Functions Analyzed** - Decompiled code with addresses
4. **Structure Definitions** - C-style struct definitions
5. **Cross-References** - Related functions and data
6. **Confidence Level** - Assessment of findings
7. **Differences** - Comparison with later WoW versions

## Function Address Reference

| Function | Address | Purpose |
|----------|---------|---------|
| `CMapChunk::SyncLoad` | 0x00698d90 | MCNK chunk parser |
| `CMapChunk::Create` | 0x00698e99 | MCNK chunk creation (complete header) |
| `CMapChunk::UnpackAlphaBits` | 0x0069a621 | MCAL alpha unpacking |
| `CMapChunk::CreateNormals` | 0x00699b60 | MCNR normal processing |
| `CMap::AllocChunkLiquid` | 0x00691860 | Liquid allocation |
| `CMap::LoadWdt` | 0x0067fde0 | WDT file loader (MAIN/MPHD/MODF) |
| `CMap::CreateMapObjDef` | 0x00680f50 | MODF WMO placement |
| `CMap::CreateDoodadDef` | 0x00680300 | MDDF doodad placement |
| `CMapObj` | 0x00693190 | WMO constructor |
| `CWorldScene::RenderMapObjDefGroups` | 0x0066e030 | WMO rendering pipeline |
| `CWorldScene::RenderChunks` | 0x0066de50 | Terrain chunk rendering |
| `BuildModelFromMdxData` | 0x00421fb0 | MDX model loader |
| `LoadBlpMips` | 0x0046f820 | BLP texture loader |

## String References

Useful error/debug strings found:

| String | Address | Context |
|--------|---------|---------|
| `"iffChunk->token=='MCNK'"` | 0x008a126c | MCNK verification |
| `"iffChunk.token=='MVER'"` | (inline) | WDT MVER chunk check |
| `"iffChunk.token=='MPHD'"` | (inline) | WDT MPHD chunk check |
| `"iffChunk.token=='MAIN'"` | (inline) | WDT MAIN chunk check |
| `"iffChunk.token=='MODF'"` | (inline) | WDT MODF chunk check |
| `"alphaPixels"` | 0x008a13a8 | Alpha map validation |
| `"normals"` | (inline) | Normal data validation |
| `"liquid"` | 0x0089e4ec | Liquid system |
| `"pIffChunk->token == 'MLIQ'"` | 0x008a2930 | WMO liquid detection |
| `"fileName"` | (inline) | File name validation |
| `"mapObjDef"` | (inline) | WMO placement validation |
| `"doodadDef"` | (inline) | Doodad placement validation |
| `".mdx"` | 0x008b1a68 | Model file extension |
| `".blp"` | 0x008389b0 | Texture file extension |
| `"D:\build\buildWoW\engine\Source\BLPFile\blp.cpp"` | 0x0085ad04 | BLP source file |

## Source Files Referenced

From error messages and asserts:
- `"D:\build\buildWoW\WoW\Source\..."` - Main client code
- `"D:\build\buildWoW\engine\Source\BLPFile\blp.cpp"` - BLP texture loader

## Next Steps

To further complete the documentation:

1. **MCNK Header**: Decompile additional field accessors to complete SMChunk structure
2. **MCLQ**: Find actual MCLQ chunk parser (search for 0x4D434C51 constant)
3. **WMO**: Trace vertex transformation and rendering pipeline
4. **Palette BLP**: Investigate paletted texture loading path
5. **MDX Chunks**: Decompile individual chunk readers for exact binary formats

## Tools Used

- **Ghidra** - Binary analysis and decompilation
- **MCP Ghidra Server** - Programmatic access to Ghidra analysis
- **PDB Symbols** - Function names and type information

## Credits

Analysis performed using the MCP Ghidra bridge with Claude AI, leveraging the availability of PDB symbols in Alpha 0.5.3 build 3368.
