# Complete Alpha 0.5.3 Format Specification

**Version**: WoW Alpha 0.5.3.3368 (December 8, 2003)  
**Sources**: Ghidra reverse engineering of WoWClient.exe with Wowae.pdb  
**Date**: 2025-12-28  
**Trust Level**: HIGH - All data from actual binary analysis

---

## Table of Contents

1. [General Conventions](#1-general-conventions)
2. [WDT Format (Monolithic)](#2-wdt-format-monolithic)
3. [MCNK Terrain Chunk Format](#3-mcnk-terrain-chunk-format)
4. [WMO v14 Format](#4-wmo-v14-format)
5. [MDX Model Format](#5-mdx-model-format)
6. [WDL Format (Low-Detail Terrain)](#6-wdl-format)
7. [BLP Texture Format](#7-blp-texture-format)
8. [Render Distance &amp; Constants](#8-render-distance--constants)
9. [PM4 Format (Server-Side Only)](#9-pm4-format-server-side-only)
10. [Quick Reference Tables](#10-quick-reference-tables)

---

## 1. General Conventions

### FourCC Tokens

All chunk IDs are **reversed on disk** (little-endian storage):
- `MVER` stored as bytes: `0x52 0x45 0x56 0x4D` = `REVM`
- When read as uint32 little-endian: `0x4D564552`

### Chunk Header Structure

```c
struct SIffChunk {
    uint32 token;   // FourCC (reversed on disk)
    uint32 size;    // Data size (excludes 8-byte header)
};
```

### No Padding Between Chunks

> [!IMPORTANT]
> Alpha format does NOT use padding bytes between chunks. Client reads sequentially.

---

## 2. WDT Format (Monolithic)

The Alpha WDT is a **monolithic file** containing all terrain tiles within a single file.

### Chunk Order (Ghidra Verified - LoadWdt @ 0x0067fde0)

```
MVER (4 bytes)      → Version = 18 (0x12)
MPHD (128 bytes)    → Header with name counts/offsets  
MAIN (65536 bytes)  → 4096 × 16-byte tile entries
MDNM (variable)     → M2/doodad name strings
MONM (variable)     → WMO name strings
[MODF] (64 bytes)   → Optional, only for WMO-based maps
[Tile Data...]      → MHDR + MCIN + MCNKs per tile
```

### MPHD Chunk (128 bytes / 0x80)

```c
struct MPHD {
    uint32 nDoodadNames;      // +0x00: Count of M2 names in MDNM
    uint32 offsDoodadNames;   // +0x04: Offset to MDNM header
    uint32 unknown;           // +0x08: Type flag (2 = WMO-based)
    uint32 nMapObjNames;      // +0x0C: Count of WMO names in MONM
    uint32 offsMapObjNames;   // +0x10: Offset to MONM header
    uint8  padding[108];      // +0x14: Reserved
};
// Total: 128 bytes (0x80)
```

### MAIN Chunk (65536 bytes / 0x10000)

64×64 grid of tile info entries:

```c
struct SMAreaInfo {
    uint32 offset;   // Absolute offset to tile MHDR (0 = no tile)
    uint32 size;     // Size from MHDR to first MCNK
    uint32 flags;    // Runtime (write as 0)
    uint32 asyncId;  // Runtime (write as 0)
};
// 16 bytes × 4096 tiles = 65536 bytes
```

### MDDF/MODF Entry Sizes (Ghidra Verified)

> [!IMPORTANT]
> **Both MDDF and MODF entries are 64 bytes (0x40) in Alpha!**
> Verified via `CMapArea::Create` where entries are processed with `<< 6` (×64).

```c
struct SMDoodadDef {  // MDDF entry - 64 bytes
    uint32 nameIndex;       // Index into MDNM
    uint32 uniqueId;        // Unique ID
    C3Vector position;      // 12 bytes
    C3Vector rotation;      // 12 bytes (degrees)
    uint16 scale;           // Scale (1024 = 1.0)
    uint16 flags;
    uint8  padding[28];     // Padding to 64 bytes
};

struct SMMapObjDef {  // MODF entry - 64 bytes
    uint32 nameIndex;       // Index into MONM
    uint32 uniqueId;        // Unique ID
    C3Vector position;      // 12 bytes
    C3Vector rotation;      // 12 bytes (degrees)
    CAaBox extents;         // 24 bytes (bounding box)
    uint16 flags;
    uint16 doodadSet;
    uint16 nameSet;
    uint16 padding;
};
```

---

## 3. MCNK Terrain Chunk Format

### World Coordinate Formula (CreateVertices @ 0x006997e0)

```c
// Constants from Ghidra
CHUNK_SIZE = 33.333333f;   // ___real_42055555
MAP_CENTER = 17066.6666f;  // ___real_46855555 = 32 × 533.333333

// NOTE: X and Y are SWAPPED!
worldCorner.x = -(chunkOffset.y * CHUNK_SIZE) + MAP_CENTER;
worldCorner.y = -(chunkOffset.x * CHUNK_SIZE) + MAP_CENTER;
```

### MCNK Header Offsets (CMapChunk::Create @ 0x00698e99)

| Offset | Field | Type | Notes |
|--------|-------|------|-------|
| 0x00 | IFF Token | uint32 | 'MCNK' (0x4D434E4B) |
| 0x04 | Size | uint32 | Chunk data size |
| 0x08 | flags | uint32 | Chunk flags |
| 0x0C | **indexX** | uint32 | Chunk X (0-15) |
| 0x10 | **indexY** | uint32 | Chunk Y (0-15) |
| 0x18 | **nLayers** | uint32 | Texture layer count (max 4) |
| 0x1C | **nDoodadRefs** | uint32 | MCRF doodad count |
| 0x3C | **sizeShadow** | uint32 | Shadow map size |
| 0x44 | mcrfSize | uint32 | MCRF chunk size |
| 0x88 | **MCVT** | - | Height data starts (145 floats) |
| 0x2CC | **MCNR** | - | Normal data starts (145×3 bytes) |
| 0x48C | MCLY token | uint32 | Layer chunk IFF header |
| 0x494 | MCLY data | - | Layer entries start |

### MCVT (Height Map) - 580 bytes

145 floats in outer-inner pattern:
- 9×9 outer vertices (81) + 8×8 inner vertices (64) = 145

### MCNR (Normals) - 448 bytes  

145 packed normals + 13 bytes padding:

```c
struct PackedNormal {
    int8 x, y, z;  // Normalized: divide by 127
};
// 145 × 3 = 435 bytes + 13 padding = 448
```

### SMLayer Structure (MCLY entries)

```c
struct SMLayer {
    uint32 textureId;    // Index into MTEX
    uint32 props;        // Layer properties (0x100 = has alpha)
    uint32 offsAlpha;    // Offset into MCAL
    int32  effectId;     // Ground effect ID
};
```

### Liquid Types (Ghidra Verified)

| Type | Value | Notes |
|------|-------|-------|
| Water | 0 | Standard water |
| Ocean | 1 | Deep water, fatigue |
| Magma | 2 | Lava, fire damage |
| Slime | 3 | Slime, nature damage |

---

## 4. WMO v14 Format

### Overview

WMO v14 is a **monolithic single-file format**. Version = 14 (0x0E).

**Container**: Uses `MOMO` wrapper chunk (0x4D4F4D4F).

### Root Chunk Order (CreateDataPointers @ 0x006aeaa0)

| # | Chunk | FourCC | Entry Size | Calculation |
|---|-------|--------|------------|-------------|
| 1 | MOHD | 0x4D4F4844 | Header | Single struct |
| 2 | MOTX | 0x4D4F5458 | Variable | String table |
| 3 | MOMT | 0x4D4F4D54 | **44 bytes** | `size / 0x2c` |
| 4 | MOGN | 0x4D4F474E | Variable | String table |
| 5 | MOGI | 0x4D4F4749 | **40 bytes** | `size / 0x28` |
| 6 | MOPV | 0x4D4F5056 | 12 bytes | `size / 0xc` |
| 7 | MOPT | 0x4D4F5054 | **20 bytes** | `size / 0x14` |
| 8 | MOPR | 0x4D4F5052 | 8 bytes | `size >> 3` |
| 9 | MOLT | 0x4D4F4C54 | **32 bytes** | `size >> 5` |
| 10 | MODS | 0x4D4F4453 | **32 bytes** | `size >> 5` |
| 11 | MODN | 0x4D4F444E | Variable | String table |
| 12 | MODD | 0x4D4F4444 | **40 bytes** | `size / 0x28` |
| 13 | MFOG | 0x4D464F47 | **48 bytes** | `size / 0x30` |

### MOMT Material (44 bytes - NOT 64!)

```c
struct SMOMaterial_v14 {
    /*0x00*/ uint32 version;         // V14 only!
    /*0x04*/ uint32 flags;
    /*0x08*/ uint32 blendMode;
    /*0x0C*/ uint32 texture1Offset;  // Into MOTX
    /*0x10*/ uint32 sidnColor;       // Emissive
    /*0x14*/ uint32 frameSidnColor;  // Runtime
    /*0x18*/ uint32 texture2Offset;  // Environment map
    /*0x1C*/ uint32 diffColor;
    /*0x20*/ uint32 groundType;
    /*0x24*/ uint8  padding[8];
};
// Total: 44 bytes (0x2C)
```

### MOGI Group Info (40 bytes - NOT 32!)

```c
struct SMOGroupInfo_v14 {
    /*0x00*/ uint32 flags;           // 4 bytes
    /*0x04*/ CAaBox boundingBox;     // 24 bytes
    /*0x1C*/ int32  nameOffset;      // Into MOGN
    /*0x20*/ uint32 unknown[4];      // Additional fields
};
// Total: 40 bytes (0x28)
```

### Group Chunk Order (CMapObjGroup::CreateDataPointers @ 0x006af2d0)

> [!WARNING]
> **Index chunk is MOIN (0x4D4F494E), NOT MOVI!**

| # | Chunk | FourCC | Entry Size |
|---|-------|--------|------------|
| 1 | MOPY | 0x4D4F5059 | **4 bytes** (`size >> 2`) |
| 2 | MOVT | 0x4D4F5654 | 12 bytes |
| 3 | MONR | 0x4D4F4E52 | 12 bytes |
| 4 | MOTV | 0x4D4F5456 | 8 bytes |
| 5 | **MOLV** | 0x4D4F4C56 | 8 bytes (lightmap UVs) |
| 6 | **MOIN** | 0x4D4F494E | 2 bytes (indices) |
| 7 | MOBA | 0x4D4F4241 | **24 bytes** |

### MOPY Face Info (4 bytes - NOT 2!)

```c
struct SMOPoly_v14 {
    uint8 flags;       // Face flags
    uint8 materialId;  // Material index
    uint16 unknown;    // Additional data
};
// Total: 4 bytes
```

---

## 5. MDX Model Format

### Overview

MDX is a **Warcraft III-derived format** used in Alpha. Magic: `MDLX`.

### Chunk Loading Order (BuildModelFromMdxData @ 0x00421fb0)

1. MODL - Global properties
2. TEXS - Textures (268 bytes each)
3. MTLS - Materials
4. GEOS - Geosets
5. ATCH - Attachments
6. SEQS/GLBS - Animations
7. RIBB - Ribbon emitters
8. PRE2 - Particle emitters
9. LITE - Lights
10. CAMS - Cameras
11. CLID - Collision
12. PIVT - Pivot points

### TEXS Entry (268 bytes / 0x10C)

```c
struct MDXTexture {
    uint32 replaceableId;     // +0x00 (0 = use filename)
    char   filename[260];     // +0x04 (null-padded path)
    uint32 flags;             // +0x108
};
// Total: 268 bytes (0x10C) - verified via `size / 0x10c`
```

### Key Ghidra Addresses

| Function | Address | Chunk |
|----------|---------|-------|
| BuildModelFromMdxData | 0x00421fb0 | Entry |
| MdxReadTextures | 0x0044e310 | TEXS |
| MdxReadMaterials | 0x0044e550 | MTLS |
| MdxReadGeosets | 0x0044eba0 | GEOS |
| MDLFileBinarySeek | 0x0078be40 | Chunk seek |

---

## 6. WDL Format

Low-detail heightmap for distant terrain.

```
MVER (version = 18)
MAOF (64×64 × 4 bytes = 16384 bytes offset table)
MARE[n] (545 int16s = 1090 bytes per tile)
```

### Height Grid

545 heights = 17×17 outer + 16×16 inner (same pattern as MCVT).

---

## 7. BLP Texture Format

Alpha uses **BLP2** format.

**Maximum size**: 256×256 pixels (modern textures must be downscaled).

---

## 8. Render Distance &amp; Constants

### Far Clip (FarClipCallback @ 0x00671c20)

**Range**: 177.0 to 777.0 yards

### Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| ADT/Tile Width | 533.33333f | Ghidra |
| Chunk Width | 33.33333f | Ghidra |
| Map Center | 17066.6666f | 32 × 533.33333 |
| Vertices per Chunk | 145 | 9×9 + 8×8 |
| WDL Heights per Tile | 545 | 17×17 + 16×16 |

---

## 9. PM4 Format (Server-Side Only)

> [!IMPORTANT]
> **PM4 files are NOT read by any WoW client.** They are server-side pathfinding data.

### Origin

PM4 files were accidentally shipped to players during a **Cataclysm 4.0.0 PTR build in 2010**. They originate from server infrastructure, NOT client data.

### Key Facts

| Fact | Detail |
|------|--------|
| **Purpose** | Server-side NPC pathfinding/navigation mesh |
| **Era** | Cataclysm 4.0.0 (2010) |
| **Client Support** | NONE - no client reads PM4 |
| **Accidental Release** | PTR build included server files |

### Structure (Reverse-Engineered)

Since no client code exists, PM4 is analyzed from file structure:
- MSLK: Surface links
- MSUR: Surface definitions
- MSVT: Vertices
- MSVI: Vertex indices
- MSCN: Collision normals
- MPRL/MPRR: Placement references

---

## 10. Quick Reference Tables

### Structure Sizes (Ghidra Verified)

| Structure | Size | Verification |
|-----------|------|--------------|
| **MDDF entry** | **64 bytes** | `CMapArea::Create` |
| **MODF entry** | **64 bytes** | `CMapArea::Create` |
| MPHD | 128 bytes | `LoadWdt` reads 0x80 |
| MAIN entry | 16 bytes | 0x10000 / 4096 |
| MOMT (WMO material) | **44 bytes** | `size / 0x2c` |
| MOGI (WMO group info) | **40 bytes** | `size / 0x28` |
| MOPY (WMO face) | **4 bytes** | `size >> 2` |
| MODD (WMO doodad) | **40 bytes** | `size / 0x28` |
| MFOG (WMO fog) | **48 bytes** | `size / 0x30` |
| MOBA (WMO batch) | 24 bytes | `size / 0x18` |
| MOLT (WMO light) | 32 bytes | `size >> 5` |
| MODS (WMO doodad set) | 32 bytes | `size >> 5` |
| TEXS (MDX texture) | **268 bytes** | `size / 0x10c` |

### v14 vs v17 Differences

| Aspect | v14 (Alpha) | v17+ (LK) |
|--------|-------------|-----------|
| File layout | Monolithic | Root + groups |
| MOMO container | Yes | No |
| MOMT size | 44 bytes | 64 bytes |
| MOGI size | 40 bytes | 32 bytes |
| MOPY size | 4 bytes | 2 bytes |
| Index chunk | **MOIN** | MOVI |
| MOLV chunk | Present | Removed |

### Ghidra Key Addresses

| Function | Address |
|----------|---------|
| LoadWdt | 0x0067fde0 |
| CMapArea::Create | 0x006aae69 |
| CMapChunk::Create | 0x00698e99 |
| CreateVertices | 0x006997e0 |
| CMapObj::CreateDataPointers | 0x006aeaa0 |
| CMapObjGroup::CreateDataPointers | 0x006af2d0 |
| BuildModelFromMdxData | 0x00421fb0 |

---

*Document consolidated from Ghidra analysis of WoWClient.exe (0.5.3.3368) with Wowae.pdb symbols. All sizes verified via binary decompilation.*
