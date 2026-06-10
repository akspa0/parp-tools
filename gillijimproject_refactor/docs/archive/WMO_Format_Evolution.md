# WMO Format Evolution (Alpha 0.5.3 - 0.6.0)

**Based on Reverse Engineering of WoW Alpha Client Binaries**

This document details the critical format changes in World Map Objects (WMO) across early World of Warcraft Alpha builds. This research is vital for maintaining the `WoWMapConverter` tool and ensuring correct asset conversion.

## Version Timeline

| Client Version | WMO Version | Key Characteristic | Status |
| :--- | :--- | :--- | :--- |
| **Alpha 0.5.3** | **v14** | Embedded Lightmaps, Small Headers | **Supported** |
| **Alpha 0.5.5** | **v14** | (Identical to 0.5.3) | **Supported** |
| **Alpha 0.6.0** | **v16** (Hybrid) | No Embedded LMs, Hybrid Headers | **Documented** |
| **Alpha 0.7.0+** | **v17** | Modern Format (Standard) | Standard |

---

## 1. Alpha 0.5.3 / 0.5.5 (Format v14)

**File Spec**: `v14`
**Status**: Fully analyzed and supported.

### Root File Structure
The Root WMO file uses smaller, older header variants compared to modern (v17) files.

| Chunk | Token | Size of Entry | Notes |
| :--- | :--- | :--- | :--- |
| **MOMT** | `0x4D4F4D54` | **0x2C** (44 bytes) | Materials. Smaller than v17 (0x40). |
| **MOGI** | `0x4D4F4749` | **0x28** (40 bytes) | Group Info. Larger than v17 (0x20). |
| **MOLT** | `0x4D4F4C54` | **0x20** (32 bytes) | Lights. Smaller than v17 (0x30). |

### Group File Structure (MOGP)
The Group file supports embedded lightmaps, a feature removed in later versions.

*   **MOPY (Polys)**: Present. Explicit list of triangle faces.
*   **MOIN (Indices)**: Standard index list (uint16).
*   **Embedded Lightmaps**:
    *   Flag `0x0002` in MOGP header enables this.
    *   Chunks: `MOLM` (Info) and `MOLD` (Data).
    *   **Data Size**: `0x8004` (32KB + 4 bytes header) per lightmap.
    *   **MOLV**: Explicit Lightmap UV coordinates chunk.
*   **Vertex Colors**:
    *   Flag `0x0004` enables `MOCV` chunk.
    *   Standard RGBA format.

---

## 2. Alpha 0.6.0 (Format v16 - "The Hybrid")

**File Spec**: `v16` (Transitional)
**Status**: Analyzed. Represents the transition point between v14 architecture and v17 data structures.

### Key Changes
1.  **Root File Modernized**: The Root file structure jumps immediately to v17 standards.
2.  **Embedded Lightmaps Removed**: The engine no longer supports `MOLM`/`MOLD`. Lightmaps are likely externalized (BLP) or disabled.
3.  **Geometry Hybrid**: The Group file keeps the old `MOPY` (Polys) chunk but adopts the modern `MOVI` (Indices) token.

### Root File Structure (v17-style)

| Chunk | Token | Size of Entry | Change vs v14 |
| :--- | :--- | :--- | :--- |
| **MOMT** | `0x4D4F4D54` | **0x40** (64 bytes) | **Expanded** to modern size. |
| **MOGI** | `0x4D4F4749` | **0x20** (32 bytes) | **Shrunk** to modern size. |
| **MOLT** | `0x4D4F4C54` | **0x30** (48 bytes) | **Expanded** to modern size. |

### Group File Structure (Hybrid)

| Chunk | Token | Entry Size | Notes |
| :--- | :--- | :--- | :--- |
| **MOPY** | `0x4D4F5059` | **4 bytes** | Polygon Materials/Flags. |
| **MOVI** | `0x4D4F5649` | **2 bytes** | **Vertex Indices** (uint16). Replaces `MOIN`. |
| **MOLV** | `MOLV` | - | **Removed**. No lightmap UVs in main sequence. |
| **MOBA** | `0x4D4F4241` | **32 bytes** | **Expanded**. Material ID is at **offset 0x17** (1 byte). |

### Lighting & Flags
*   **Flag 0x2 (Embedded LMs)**: **No longer checked**. The code blocks for `MOLM`/`MOLD` are removed.
*   **Flag 0x4 (Vertex Colors)**: **Still supported**. `MOCV` chunk is read if flag is set.

---

## 3. Summary of Converter Requirements

To support these versions, the converter must handle:

1.  **v14 (0.5.3/0.5.5)**:
    *   Write `MOMT` as 0x2C bytes.
    *   Write `MOGI` as 0x28 bytes.
    *   Support embedding lightmaps via `MOLD` if preserving lighting (or strip them).

2.  **v16 (0.6.0)**:
    *   Write `MOMT` as 0x40 bytes (v17 style).
    *   Write `MOGI` as 0x20 bytes (v17 style).
    *   **Do not** write embedded lightmaps (`MOLD`/`MOLM`).
    *   Use `MOVI` token instead of `MOIN`.
    *   Keep writing `MOPY` chunk.

3.  **WDT Compatibility**:
    *   WDT format (MDDF/MODF) is **identical** across 0.5.3, 0.5.5, and 0.6.0.

---

## 4. WMO v17 (Standard) - Reference Specification

**File Spec**: `v17`  
**Used in**: WoW 3.3.5a (Wrath of the Lich King) and later.

This section documents the **definitive v17 format** based on wowdev.wiki and Ghidra analysis of the 3.3.5a client.

### 4.1 MOGP Header (68 bytes)

```c
struct SMOGroupHeader {
  /*0x00*/ uint32_t groupName;           // Offset into MOGN
  /*0x04*/ uint32_t descriptiveGroupName;
  /*0x08*/ uint32_t flags;               // See group flags below
  /*0x0C*/ CAaBox boundingBox;           // 24 bytes (2x C3Vector)
  /*0x24*/ uint16_t portalStart;         // Index into MOPR
  /*0x26*/ uint16_t portalCount;
  /*0x28*/ uint16_t transBatchCount;     // Transparent batches
  /*0x2A*/ uint16_t intBatchCount;       // Interior batches
  /*0x2C*/ uint16_t extBatchCount;       // Exterior batches
  /*0x2E*/ uint16_t padding;
  /*0x30*/ uint8_t fogIds[4];
  /*0x34*/ uint32_t groupLiquid;
  /*0x38*/ uint32_t uniqueID;            // WMOAreaTable ID
  /*0x3C*/ uint32_t flags2;
  /*0x40*/ uint32_t unk;
}; // Total: 68 bytes (0x44)
```

**Critical**: `transBatchCount + intBatchCount + extBatchCount` MUST equal total batch count in MOBA.

### 4.2 MOBA (Render Batches) - 24 bytes per batch

```c
struct SMOBatch {
  /*0x00*/ int16_t bx, by, bz;           // Bounding box min (truncated floats)
  /*0x06*/ int16_t tx, ty, tz;           // Bounding box max
  /*0x0C*/ uint32_t startIndex;          // First index in MOVI (vertex indices)
  /*0x10*/ uint16_t count;               // Number of MOVI indices
  /*0x12*/ uint16_t minIndex;            // First vertex in MOVT
  /*0x14*/ uint16_t maxIndex;            // Last vertex in MOVT (inclusive)
  /*0x16*/ uint8_t flags;
  /*0x17*/ uint8_t material_id;          // Index into MOMT
}; // Total: 24 bytes (0x18)
```

**Bounding Box**: Used for batch-level culling. Must be calculated from actual vertices.

### 4.3 MOBN (BSP Tree Nodes) - 16 bytes per node

```c
struct CAaBspNode {
  /*0x00*/ uint16_t flags;      // 0=YZ, 1=XZ, 2=XY, 4=Leaf
  /*0x02*/ int16_t negChild;    // -1 (0xFFFF) for no child
  /*0x04*/ int16_t posChild;    // -1 (0xFFFF) for no child
  /*0x06*/ uint16_t nFaces;     // Number of faces (for leaves)
  /*0x08*/ uint32_t faceStart;  // Index into MOBR
  /*0x0C*/ float planeDist;     // Split plane distance
}; // Total: 16 bytes (0x10)
```

> [!CAUTION]
> The BSP node is **exactly 16 bytes**. Writing fewer bytes (e.g., 14) will corrupt all subsequent chunk reads.

### 4.4 Key Group Flags

| Flag | Hex | Description |
|:---|:---|:---|
| HAS_BSP | 0x0001 | MOBN/MOBR chunks present |
| HAS_LIGHTMAP | 0x0002 | MOLM/MOLD (v14 only, unused in v17) |
| HAS_VERTEX_COLORS | 0x0004 | MOCV chunk present |
| EXTERIOR | 0x0008 | Outdoor group |
| EXTERIOR_LIT | 0x0040 | Use exterior lighting |
| HAS_LIGHTS | 0x0200 | MOLR chunk present |
| HAS_DOODADS | 0x0800 | MODR chunk present |
| HAS_WATER | 0x1000 | MLIQ chunk present |
| INTERIOR | 0x2000 | Indoor group |

### 4.5 Required Chunk Order (v17)

Groups must contain chunks in this exact order:
1. MOGP (header + all subchunks)
2. MOPY (triangle flags/materials)
3. MOVI (vertex indices)
4. MOVT (vertices)
5. MONR (normals)
6. MOTV (texture coordinates)
7. MOBA (render batches)
8. [MOLR] (light refs, if flag 0x200)
9. [MODR] (doodad refs, if flag 0x800)
10. [MOBN] (BSP nodes, if flag 0x1)
11. [MOBR] (BSP face refs, if flag 0x1)
12. [MOCV] (vertex colors, if flag 0x4)
13. [MLIQ] (liquid, if flag 0x1000)
