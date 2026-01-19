# WMO Format Specification

**Complete World Map Object (WMO) format documentation across all WoW versions.**

---

## Version History

| Client | WMO Version | Key Features |
|:---|:---|:---|
| Alpha 0.5.3 | v14 | Embedded lightmaps, 44-byte MOMT, 40-byte MOGI |
| Alpha 0.5.5 | v14 | Identical to 0.5.3 |
| Alpha 0.6.0 | v16 | Hybrid: v17 root headers, v14 group features |
| WotLK 3.3.5a | v17 | Standard modern format |

> [!NOTE]
> **Ghidra Analysis (WoW 4.0.0.11927)**: Key WMO classes decompiled:
> - `CMapObj` - Main WMO handler (parsing, rendering)
> - `CMapObjDef` - WMO placement/instance
> - `CMapObjGroup` - Individual group rendering
> - DBC: `DBFilesClient\WMOAreaTable.dbc` - WMO area definitions

---

# Part 1: Root File Structure

## MOHD (Header) - 64 bytes

```c
struct SMOHeader {
  uint32_t nTextures;     // Texture count
  uint32_t nGroups;       // Group count
  uint32_t nPortals;      // Portal count
  uint32_t nLights;       // Light count
  uint32_t nDoodadNames;  // Doodad name count
  uint32_t nDoodadDefs;   // Doodad definition count
  uint32_t nDoodadSets;   // Doodad set count
  CArgb ambColor;         // Ambient color
  uint32_t wmoID;         // WMOAreaTable ID
#if VERSION >= 0.6.0
  CAaBox bounding_box;    // 24 bytes
  uint16_t flags;
  uint16_t numLod;
#else // v14
  uint8_t padding[0x1c];
#endif
};
```

## MOMT (Materials)

| Version | Entry Size | Notes |
|:---|:---|:---|
| v14 (0.5.3) | **44 bytes** (0x2C) | Smaller, has `version` field at start |
| v16+ | **64 bytes** (0x40) | Standard modern size |

```c
struct SMOMaterial {
#if VERSION == v14
  uint32_t version;           // v14 only
#endif
  uint32_t flags;             // Material flags
#if VERSION >= v16
  uint32_t shader;            // Shader index
#endif
  uint32_t blendMode;         // EGxBlend
  uint32_t texture_1;         // MOTX offset
  CImVector sidnColor;        // Emissive color
  CImVector frameSidnColor;   // Runtime emissive
  uint32_t texture_2;         // Second texture
  CImVector diffColor;        // Diffuse color
  uint32_t ground_type;       // TerrainType
#if VERSION >= v16
  uint32_t texture_3;
  uint32_t color_2;
  uint32_t flags_2;
  uint32_t runTimeData[4];
#endif
};
```

## MOGI (Group Info)

| Version | Entry Size | Notes |
|:---|:---|:---|
| v14 (0.5.3) | **40 bytes** (0x28) | Has offset/size fields |
| v16+ | **32 bytes** (0x20) | Standard modern size |

```c
struct SMOGroupInfo {
#if VERSION == v14
  uint32_t offset;       // Embedded group offset
  uint32_t size;         // Embedded group size
#endif
  uint32_t flags;        // Group flags (matches MOGP)
  CAaBox bounding_box;   // 24 bytes
  int32_t nameoffset;    // Offset in MOGN
};
```

## MOLT (Lights)

| Version | Entry Size |
|:---|:---|
| v14 | **32 bytes** (0x20) |
| v16+ | **48 bytes** (0x30) |

---

# Part 2: Group File Structure (MOGP)

## MOGP Header

### v14 Header (0.5.3/0.5.5)
```c
struct SMOGxBatch {
  uint16_t vertStart;
  uint16_t vertCount;
  uint16_t batchStart;
  uint16_t batchCount;
};

struct MOGP_v14 {
  uint32_t groupName;
  uint32_t descriptiveGroupName;
  uint32_t flags;
  CAaBox boundingBox;           // 24 bytes
  uint32_t portalStart;         // Note: uint32, not uint16
  uint32_t portalCount;
  uint8_t fogIds[4];
  uint32_t groupLiquid;
  SMOGxBatch intBatch[4];       // v14-only inline batches
  SMOGxBatch extBatch[4];
  uint32_t uniqueID;
  uint8_t padding[8];
};
```

### v17 Header (3.3.5a) - 68 bytes
```c
struct SMOGroupHeader {
  /*0x00*/ uint32_t groupName;
  /*0x04*/ uint32_t descriptiveGroupName;
  /*0x08*/ uint32_t flags;
  /*0x0C*/ CAaBox boundingBox;         // 24 bytes
  /*0x24*/ uint16_t portalStart;
  /*0x26*/ uint16_t portalCount;
  /*0x28*/ uint16_t transBatchCount;   // Transparent batches
  /*0x2A*/ uint16_t intBatchCount;     // Interior batches
  /*0x2C*/ uint16_t extBatchCount;     // Exterior batches
  /*0x2E*/ uint16_t padding;
  /*0x30*/ uint8_t fogIds[4];
  /*0x34*/ uint32_t groupLiquid;
  /*0x38*/ uint32_t uniqueID;
  /*0x3C*/ uint32_t flags2;
  /*0x40*/ uint32_t unk;
};
```

> [!IMPORTANT]
> `transBatchCount + intBatchCount + extBatchCount` MUST equal total batches in MOBA.

---

## Group Flags

| Flag | Hex | Description |
|:---|:---|:---|
| HAS_BSP | 0x0001 | MOBN/MOBR present |
| HAS_LIGHTMAP | 0x0002 | MOLM/MOLD (v14 only) |
| HAS_VERTEX_COLORS | 0x0004 | MOCV present |
| EXTERIOR | 0x0008 | Outdoor group |
| EXTERIOR_LIT | 0x0040 | Use exterior lighting |
| HAS_LIGHTS | 0x0200 | MOLR present |
| HAS_DOODADS | 0x0800 | MODR present |
| HAS_WATER | 0x1000 | MLIQ present |
| INTERIOR | 0x2000 | Indoor group |

---

## MOBA (Render Batches)

### v14 Batch - 24 bytes
```c
struct SMOBatch_v14 {
  uint8_t lightMap;      // MOLM index
  uint8_t texture;       // MOMT index
  int16_t bx, by, bz;    // Bounding min
  int16_t tx, ty, tz;    // Bounding max
  uint16_t startIndex;   // MOVI start (16-bit!)
  uint16_t count;        // Index count
  uint16_t minIndex;     // First vertex
  uint16_t maxIndex;     // Last vertex
  uint8_t flags;
  uint8_t padding;
  uint8_t unknown[8];    // Always zero
};
```

### v17 Batch - 24 bytes
```c
struct SMOBatch {
  /*0x00*/ int16_t bx, by, bz;    // Bounding min
  /*0x06*/ int16_t tx, ty, tz;    // Bounding max
  /*0x0C*/ uint32_t startIndex;   // MOVI start (32-bit!)
  /*0x10*/ uint16_t count;
  /*0x12*/ uint16_t minIndex;
  /*0x14*/ uint16_t maxIndex;
  /*0x16*/ uint8_t flags;
  /*0x17*/ uint8_t material_id;
};
```

> [!CAUTION]
> Bounding box values are used for batch-level culling. Incorrect values cause geometry to disappear randomly.

---

## MOBN (BSP Tree Nodes) - 16 bytes per node

```c
enum BSPFlags {
  Flag_XAxis = 0x0,
  Flag_YAxis = 0x1,
  Flag_ZAxis = 0x2,
  Flag_AxisMask = 0x3,
  Flag_Leaf = 0x4,
  Flag_NoChild = 0xFFFF
};

struct CAaBspNode {
  /*0x00*/ uint16_t flags;      // 0=YZ, 1=XZ, 2=XY, 4=Leaf
  /*0x02*/ int16_t negChild;    // -1 for no child
  /*0x04*/ int16_t posChild;    // -1 for no child
  /*0x06*/ uint16_t nFaces;     // Face count (leaves)
  /*0x08*/ uint32_t faceStart;  // MOBR index
  /*0x0C*/ float planeDist;     // Split distance
}; // Total: 16 bytes
```

> [!CAUTION]
> The BSP node is **exactly 16 bytes**. Writing 14 bytes corrupts all subsequent chunk reads.

---

## Lighting Modes (v14)

v14 WMOs support two mutually exclusive lighting modes:

### Embedded Lightmaps (Flag 0x2)
- `MOLM`: Lightmap info headers
- `MOLD`: Lightmap data (32KB + 4 bytes per map)
- `MOLV`: Lightmap UV coordinates

### Vertex Colors (Flag 0x4)
- `MOCV`: RGBA vertex colors

> [!NOTE]
> v16+ clients do NOT support embedded lightmaps. Use MOCV only.

---

## Required Chunk Order (v17)

Groups must contain chunks in this order:
1. MOGP (header)
2. MOPY (triangle flags)
3. MOVI (vertex indices)
4. MOVT (vertices)
5. MONR (normals)
6. MOTV (UVs)
7. MOBA (batches)
8. [MOLR] (light refs)
9. [MODR] (doodad refs)
10. [MOBN] (BSP nodes)
11. [MOBR] (BSP faces)
12. [MOCV] (vertex colors)
13. [MLIQ] (liquid)

---

# Part 3: Converter Requirements

## Writing v14 (for Alpha 0.5.3/0.5.5)
- MOMT: 44 bytes per entry
- MOGI: 40 bytes per entry (with offset/size)
- Support MOLM/MOLD if preserving lightmaps
- MOBA: 16-bit startIndex

## Writing v17 (for WotLK 3.3.5a)
- MOMT: 64 bytes per entry
- MOGI: 32 bytes per entry
- MOBN: **16 bytes per node** (not 14!)
- Batch counts must be set correctly in MOGP
- No lightmap support (MOCV only)
