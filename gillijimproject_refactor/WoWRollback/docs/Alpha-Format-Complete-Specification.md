# Complete Alpha 0.5.3 Format Specification

**Version**: WoW Alpha 0.5.3.3368 (December 8, 2003)  
**Sources**: Ghidra reverse engineering + existing parser code  
**Date**: 2025-12-28

---

## Table of Contents

1. [General Conventions](#1-general-conventions)
2. [WDT Format (Monolithic)](#2-wdt-format-monolithic)
3. [MCNK Terrain Chunk Format](#3-mcnk-terrain-chunk-format)
4. [WMO v14 Format](#4-wmo-v14-format)
5. [WDL Format (Low-Detail Terrain)](#5-wdl-format)
6. [MDX Model Format](#6-mdx-model-format)
7. [BLP Texture Format](#7-blp-texture-format)
8. [Render Distance & LOD](#8-render-distance--lod)

---

## 1. General Conventions

### FourCC Tokens

All chunk IDs are **reversed on disk** (little-endian storage):
- `MVER` is stored as `REVM` (bytes: 0x52 0x45 0x56 0x4D)
- When read as uint32 little-endian: `0x4D564552`

### Chunk Header Structure

```c
struct SIffChunk {
    uint32 token;   // FourCC (reversed on disk)
    uint32 size;    // Data size (excludes header)
};
// Total: 8 bytes
```

### No Padding Between Chunks

**Critical**: Alpha format does NOT use padding bytes between chunks, unlike later versions.

---

## 2. WDT Format (Monolithic)

The Alpha WDT is a **monolithic file** containing all terrain tiles embedded within a single file.

### Chunk Order (Ghidra Verified - Sequential Reading)

```
MVER (4 bytes)
MPHD (128 bytes)
MAIN (65536 bytes)
MDNM (variable)
MONM (variable)
[MODF] (optional, only for WMO-based maps)
[Tile Data...] (MHDR + MCIN + MCNKs for each tile)
```

### MVER Chunk

```c
struct MVER {
    uint32 version;  // Always 18 (0x12) for Alpha
};
```

### MPHD Chunk (128 bytes)

```c
struct MPHD {
    uint32 nDoodadNames;      // Count of M2 names in MDNM
    uint32 offsDoodadNames;   // Absolute offset to MDNM chunk header
    uint32 nMapObjNames;      // Count of WMO names in MONM
    uint32 offsMapObjNames;   // Absolute offset to MONM chunk header
    // Remaining 112 bytes: padding/reserved
};
```

**Important**: Offsets point to chunk **headers** (FourCC), not data. However, Ghidra analysis shows the client reads chunks **sequentially** and does NOT seek to these offsets during WDT loading.

### MAIN Chunk (65536 bytes)

64×64 grid of tile info entries:

```c
struct SMAreaInfo {
    uint32 offset;   // Absolute offset to tile MHDR (0 = no tile)
    uint32 size;     // Size from MHDR to first MCNK (NOT total)
    uint32 flags;    // Runtime flag (write as 0)
    uint32 pad;      // Padding (write as 0)
};
// 16 bytes × 4096 tiles = 65536 bytes
```

### MDNM Chunk (M2/Doodad Names)

Null-terminated ASCII strings for M2 model paths:
```
"World\...\model1.mdx\0"
"World\...\model2.mdx\0"
"\0"  // Trailing null terminator
```

### MONM Chunk (WMO Names)

Null-terminated ASCII strings for WMO paths:
```
"World\...\building.wmo\0"
"\0"  // Trailing null terminator
```

**Critical**: Even empty chunks must contain at least 1 byte (trailing null).

### Tile Data

For each tile with `offset != 0` in MAIN:

```
MHDR (64 bytes header)
MCIN (256 entries × 16 bytes = 4096 bytes)
MCNK[0..255] (256 terrain chunks)
```

---

## 3. MCNK Terrain Chunk Format

### MCNK Header (128 bytes)

```c
struct McnkAlphaHeader {
    /*0x00*/ uint32 flags;
    /*0x04*/ uint32 indexX;          // Chunk X index (0-15)
    /*0x08*/ uint32 indexY;          // Chunk Y index (0-15)
    /*0x0C*/ uint32 unknown1;
    /*0x10*/ uint32 nLayers;         // Texture layer count
    /*0x14*/ uint32 m2Number;        // Doodad reference count
    /*0x18*/ uint32 mcvtOffset;      // Relative to MCNK data start
    /*0x1C*/ uint32 mcnrOffset;
    /*0x20*/ uint32 mclyOffset;
    /*0x24*/ uint32 mcrfOffset;
    /*0x28*/ uint32 mcalOffset;
    /*0x2C*/ uint32 mcalSize;
    /*0x30*/ uint32 mcshOffset;
    /*0x34*/ uint32 mcshSize;
    /*0x38*/ uint32 unknown3;        // Possibly AreaID
    /*0x3C*/ uint32 wmoNumber;       // WMO reference count
    /*0x40*/ uint32 holes;           // Hole bitmap
    /*0x44*/ uint32 groundEffectsMap1;
    /*0x48*/ uint32 groundEffectsMap2;
    /*0x4C*/ uint32 groundEffectsMap3;
    /*0x50*/ uint32 groundEffectsMap4;
    /*0x54*/ uint32 unknown6;
    /*0x58*/ uint32 unknown7;
    /*0x5C*/ uint32 mcnkChunksSize;  // Size of sub-chunks (minus header)
    /*0x60*/ uint32 unknown8;
    /*0x64*/ uint32 mclqOffset;      // Liquid data offset
    /*0x68*/ uint32 unused1;
    /*0x6C*/ uint32 unused2;
    /*0x70*/ uint32 unused3;
    /*0x74*/ uint32 unused4;
    /*0x78*/ uint32 unused5;
    /*0x7C*/ uint32 unused6;
};
```

### MCNK Sub-Chunks

| Chunk | Size | Has Header | Description |
|-------|------|------------|-------------|
| MCVT | 580 | NO | Height values (145 floats) |
| MCNR | 448 | NO | Normals (145 × 3 bytes + padding) |
| MCLY | variable | YES | Texture layers (16 bytes each) |
| MCRF | variable | YES | Doodad/WMO references |
| MCSH | variable | YES | Shadow map |
| MCAL | variable | NO | Alpha maps |
| MCLQ | variable | YES | Liquid data |

### MCVT (Height Map)

145 float values in outer-inner pattern:
- 9×9 outer vertices + 8×8 inner vertices = 81 + 64 = 145

### MCNR (Normals)

145 packed normals (3 bytes each: X, Y, Z as signed bytes):
```c
struct PackedNormal {
    int8 x, y, z;  // Normalized: divide by 127
};
```

### MCLQ (Liquid Data)

```c
struct MCLQ {
    float minHeight;
    float maxHeight;
    // 9×9 vertex data (81 vertices)
    struct {
        float height;
        // ...
    } vertices[81];
    // 8×8 tile flags
    uint8 tileFlags[64];
};
```

**Liquid Types** (from Ghidra):
- 0: Water
- 1: Ocean
- 2: Magma/Lava
- 3: Slime

---

## 4. WMO v14 Format

### Overview

WMO v14 is a **monolithic single-file format** (unlike v17+ which splits root/groups).

**Version**: 14 (0x0E) - confirmed via Ghidra `fileHeader.version == 0xE`

### Top-Level Structure

```
MVER (version = 14)
MOMO (container chunk)
  ├── MOHD (header)
  ├── MOTX (textures)
  ├── MOMT (materials)
  ├── MOGN (group names)
  ├── MOGI (group info)
  ├── MOSB (skybox)
  ├── MOVV (visible vertices)
  ├── MOVB (visible batches)
  ├── MOLT (lights)
  ├── MODS (doodad sets)
  ├── MODN (doodad names)
  ├── MODD (doodad definitions)
  └── MOGP[n] (group data - one per group)
```

### MOHD (Header)

```c
struct MOHD_v14 {
    uint32 nTextures;
    uint32 nGroups;
    uint32 nPortals;
    uint32 nLights;
    uint32 nDoodadNames;
    uint32 nDoodadDefs;
    uint32 nDoodadSets;
    CArgb ambColor;
    uint32 wmoID;
    CAaBox boundingBox;
    uint32 flags;
};
```

### MOMT (Materials) - 44 bytes per entry (NOT 64!)

```c
struct WMO_Material_v14 {
    /*0x00*/ uint32 version;       // V14 only!
    /*0x04*/ uint32 flags;
    /*0x08*/ uint32 blendMode;
    /*0x0C*/ uint32 texture1Offset; // Offset into MOTX
    /*0x10*/ uint32 sidnColor;      // Emissive color
    /*0x14*/ uint32 frameSidnColor; // Runtime
    /*0x18*/ uint32 texture2Offset;
    /*0x1C*/ uint32 diffColor;
    /*0x20*/ uint32 groundType;
    /*0x24*/ uint8  padding[8];
};
// Total: 44 bytes (v17+ uses 64 bytes)
```

### MOGI (Group Info) - 32 bytes per entry

```c
struct SMOGroupInfo_v14 {
    uint32 flags;
    CAaBox boundingBox;     // 24 bytes
    int32  nameOffset;      // Offset into MOGN
};
```

### MOGP (Group Data)

Each MOGP contains:

```
MOGP header (64 bytes)
├── MOVT (vertices: 12 bytes each - x,y,z floats)
├── MOVI (indices: 2 bytes each - uint16)
├── MONR (normals: 12 bytes each)
├── MOTV (UVs: 8 bytes each - u,v floats)
├── MOPY (face info: 2 bytes each - flags, materialId)
├── MOBA (render batches: 24 bytes each in v14)
├── MOBN (BSP nodes)
└── MOBR (BSP face indices)
```

### MOBA (Render Batches) - 24 bytes per entry (v14)

```c
struct MOBA_v14 {
    uint8  lightMap;       // +0
    uint8  texture;        // +1 (material index)
    uint8  boundingBox[12]; // +2
    uint16 startIndex;     // +14 (into MOVI)
    uint16 numIndices;     // +16
    uint16 minIndex;       // +18
    uint16 maxIndex;       // +20
    uint8  flags;          // +22
    uint8  padding;        // +23
};
```

### MOPY (Face Info) - 2 bytes per face

```c
struct MOPY_entry {
    uint8 flags;
    uint8 materialId;
};
```

---

## 5. WDL Format

Low-detail terrain heightmap for distant rendering.

### Chunk Structure

```
MVER (version = 18)
MAOF (offset table: 64×64 uint32s = 16384 bytes)
MARE[n] (height data per tile: 545 int16s = 1090 bytes each)
```

### MAOF (Area Low Offsets)

```c
uint32 offsets[64][64];  // Absolute offsets to MARE chunks (0 = no data)
```

### MARE (Area Low Entry)

```c
int16 heights[545];  // 17×17 outer + 16×16 inner grid
```

**Height Grid**: Same pattern as MCVT but lower resolution (one value per MCNK chunk instead of per vertex).

---

## 6. MDX Model Format

Warcraft III-like model format used in Alpha.

### Chunk Structure

```
MDLX (magic)
VERS (version)
MODL (model info)
SEQS (sequences)
GLBS (global sequences)
TEXS (textures)
MTLS (materials)
GEOS (geosets)
GEOA (geoset animations)
BONE (bones)
PIVT (pivot points)
ATCH (attachments)
CAMS (cameras)
LITE (lights)
PRE2 (particle emitters v2)
RIBB (ribbon emitters)
EVTS (events)
CLID (collision)
```

### Loading Flow (from Ghidra)

```c
void BuildModelFromMdxData(uchar *data, uint size, CModelComplex *model, ...) {
    MdxLoadGlobalProperties(data, size, &flags, shared);  // MODL chunk
    MdxReadTextures(data, size, flags, model, status);     // TEXS
    MdxReadMaterials(data, size, flags, model, shared);    // MTLS
    MdxReadGeosets(data, size, flags, model, shared);      // GEOS
    MdxReadAttachments(data, size, flags, model, ...);     // ATCH
    MdxReadAnimation(data, size, model, flags);            // SEQS/GLBS
    MdxReadRibbonEmitters(data, size, model, shared);      // RIBB
    MdxReadEmitters2(data, size, flags, model, ...);       // PRE2
    MdxReadLights(data, size, model);                      // LITE
    MdxReadCameras(data, size, &model->m_cameras);         // CAMS
    // ...
}
```

---

## 7. BLP Texture Format

Alpha uses **BLP2** format (not BLP1).

### Constraints

- **Maximum size**: 256×256 pixels
- Modern textures (512+) must be downscaled

### Loading (from Ghidra)

```c
void CreateBlpTexture(char *filename, int flags) {
    SFile::Open(filename, &file);
    CTexture *texture = SMemAlloc(0x14c, "HTEXTURE", ...);
    texture->asyncObject = AsyncFileReadCreateObject();
    texture->asyncObject->callback = AsyncCreateBlpTextureCallback;
    AsyncFileReadObject(texture->asyncObject);
}
```

---

## 8. Render Distance & LOD

### Far Clip Range (from Ghidra)

```c
// FarClipCallback @ 0x00671c20
if (val >= 177.0f && val <= 777.0f) {
    CWorld::SetFarClip(val);
    return true;
}
ConsoleWrite("FarClip must be in range 177.0 to 777.0", DEFAULT_COLOR);
```

**Range**: **177 to 777 yards**

### Key Constants

| Constant | Value | Notes |
|----------|-------|-------|
| ADT Width | 533.33 yards | One terrain tile |
| Min FarClip | 177.0 yards | ~1/3 ADT |
| Max FarClip | 777.0 yards | ~1.5 ADTs |
| Chunk vertices | 145 | Per MCNK |
| WDL heights | 545 | Per tile (17×17 + 16×16) |

### Terrain LOD System

```
[Within FarClip?]
    ├─ Yes → Full ADT (MCNK with textures)
    └─ No  → WDL heightmap (CMapAreaLow, untextured)
```

### Fog System

Linear blend between `fogStart` and `fogEnd` to hide LOD transitions:

```c
float ComputeFogBlend(FogData *fog, float distance) {
    if (distance < fog->fogStart) return 1.0f;  // No fog
    return 1.0f - (distance - fogStart) / (fogEnd - fogStart);
}
```

---

## Quick Reference: Sizes

| Structure | Alpha Size | LK Size | Notes |
|-----------|-----------|---------|-------|
| MPHD | 128 bytes | 32 bytes | Alpha has more fields |
| MOMT (WMO material) | 44 bytes | 64 bytes | No shader field in v14 |
| MOBA (WMO batch) | 24 bytes | 24 bytes | Same |
| MCNK header | 128 bytes | 128 bytes | Different field meanings |
| MOGI (WMO group info) | 32 bytes | 32 bytes | Same |

---

## Files This Applies To

- `*.wdt` - World Definition Tables (monolithic)
- `*.wdl` - World Detail Level (low-res heightmap)
- `*.wmo` - World Map Objects (v14 monolithic)
- `*.mdx` - Models (WC3-like format)
- `*.blp` - Textures (BLP2, max 256×256)

---

*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) with Wowae.pdb symbols and existing parser code.*
