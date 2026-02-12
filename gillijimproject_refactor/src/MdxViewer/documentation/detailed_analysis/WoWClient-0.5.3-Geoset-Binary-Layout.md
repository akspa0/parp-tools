# WoW Alpha 0.5.3 (Build 3368) MDX Geoset Binary Layout Analysis

## Overview

This document provides a detailed binary layout analysis of the MDX geoset format based on Ghidra reverse engineering of WoWClient.exe (Build 3368, Dec 11 2003). It covers the exact structure of geoset data, sub-chunk parsing, and trailing fields.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`MdxReadGeosets`](MdxReadGeosets) | 0x0044eba0 | Main geoset loader |
| [`LoadGeosetData`](LoadGeosetData) | 0x0044eec0 | Core geoset data parser |
| [`LoadGeosetPrimitiveData`](LoadGeosetPrimitiveData) | 0x0044f330 | Primitive/indices parsing |
| [`LoadGeosetTransformGroups`](LoadGeosetTransformGroups) | 0x0044f570 | Bone weights/indices parsing |
| [`LoadMaterialData`](LoadMaterialData) | 0x0044e6b0 | Material layer parser |

---

## Geoset Chunk Structure (GEOS)

### Chunk Header

```
GEOS Chunk:
├── uint32_t magic      // 'GEOS' (0x534f4547)
├── uint32_t size       // Total chunk size (excluding header)
└── uint32_t geosetCount // Number of geosets
```

### Per-Geoset Sub-Chunks

Each geoset contains the following tagged sub-chunks in this **specific order**:

```
Geoset:
├── VRTX (Vertices)
├── NRMS (Normals)
├── [PVTX] (Texture Coordinates - optional, see UVAS below)
├── UVAS/TVAS (Texture Coordinate Sets)
├── [MATS] (Material ID - optional)
└── Trailing Fields (after all sub-chunks)
```

---

## Sub-Chunk Details

### 1. VRTX - Vertex Positions (Required)

```
VRTX Chunk:
├── uint32_t magic     // 'VRTX' (0x58545256)
├── uint32_t size      // Byte count: numVertices × 12 (3 × float32)
└── C3Vector vertices[numVertices]
```

**Size Calculation**: `size = numVertices × sizeof(C3Vector) = numVertices × 12`

**C3Vector Structure**:
```c
struct C3Vector {
    float x;  // +0x00
    float y;  // +0x04
    float z;  // +0x08
};
```

---

### 2. NRMS - Vertex Normals (Required)

```
NRMS Chunk:
├── uint32_t magic     // 'NRMS' (0x534d524e)
├── uint32_t size      // Byte count: numVertices × 12 (3 × float32)
└── C3Vector normals[numVertices]
```

**Size Calculation**: `size = numVertices × sizeof(C3Vector) = numVertices × 12`

**Validation**: `numNormals MUST equal numVertices` (enforced in code)

---

### 3. UVAS/TVAS - Texture Coordinates (Required)

```
UVAS Chunk:
├── uint32_t magic     // 'UVAS' (0x53415655)
├── uint32_t size      // Byte count: numUVSets × numVertices × 8
└── C2Vector uvData[numUVSets][numVertices]
```

**Size Calculation**:
```
size = numUVSets × numVertices × sizeof(C2Vector)
size = numUVSets × numVertices × 8  // 2 × float32 per UV
```

**C2Vector Structure**:
```c
struct C2Vector {
    float u;  // +0x00
    float v;  // +0x04
};
```

### Critical: Count Interpretation

**The `size` field is a raw BYTE COUNT, NOT an element count.**

- `numUVSets = size / (numVertices × 8)`
- Each UV set contains `numVertices` texture coordinates
- Total UV entries = `numUVSets × numVertices`
- Each UV = `C2Vector` (8 bytes = 2 floats)

---

### 4. PTYP - Primitive Types (Required)

```
PTYP Chunk:
├── uint32_t magic     // 'PTYP' (0x50595454)
├── uint32_t size      // Byte count: numPrimTypes
└── uint8_t primTypes[numPrimTypes]
```

**Size Calculation**: `size = numPrimTypes` (each type is 1 byte)

**Primitive Types**:
```c
enum PrimitiveType {
    PT_POINT = 0x0,
    PT_LINE = 0x1,
    PT_TRIANGLE = 0x2,
    // ... others
};
```

---

### 5. PCNT - Primitive Counts (Required)

```
PCNT Chunk:
├── uint32_t magic     // 'PCNT' (0x544e4350)
├── uint32_t size      // Byte count: numPrimTypes × 4
└── uint32_t primCounts[numPrimTypes]
```

**Size Calculation**: `size = numPrimTypes × 4` (each count is 4 bytes)

**Validation**: `numPrimTypes MUST equal numPrimTypes from PTYP` (enforced)

---

### 6. PVTX - Primitive Vertices/Indices (Required)

```
PVTX Chunk:
├── uint32_t magic     // 'PVTX' (0x58545650)
├── uint32_t size      // Byte count: numIndices × 2 (uint16_t)
└── uint16_t indices[numIndices]
```

**Size Calculation**: `size = numIndices × 2` (each index is uint16)

---

## Transform Group Sub-Chunks

### 7. GNDX - Bone Vertex Indices (Optional)

```
GNDX Chunk:
├── uint32_t magic     // 'GNDX' (0x58444e47)
├── uint32_t size      // Byte count: numVertices
└── uint8_t boneIndices[numVertices]
```

**Size Calculation**: `size = numVertices` (1 byte per vertex)

**Purpose**: Maps each vertex to a bone index (0-255)

---

### 8. MTGC/MCST - Material Shader Assignments (Optional)

```
MTGC Chunk:
├── uint32_t magic     // 'MTGC' (0x4354534d)
├── uint32_t size      // Byte count: numLayers × 4
└── uint32_t shaderIds[numLayers]
```

**Size Calculation**: `size = numLayers × 4` (each shader ID is 4 bytes)

---

### 9. MATS - Material IDs (Optional)

```
MATS Chunk:
├── uint32_t magic     // 'MATS' (0x5354414d)
├── uint32_t size      // Byte count: numLayers × 4
└── uint32_t materialIds[numLayers]
```

**Size Calculation**: `size = numLayers × 4` (each material ID is 4 bytes)

---

### 10. BIDX - Bone Indices (Optional)

```
BIDX Chunk:
├── uint32_t magic     // 'BIDX' (0x43495442)
├── uint32_t size      // Byte count: numBoneEntries
└── uint32_t boneIndices[numBoneEntries]
```

**Size Calculation**: `size = numBoneEntries × 4` (each index is 4 bytes)

---

### 11. BWGT - Bone Weights (Optional)

```
BWGT Chunk:
├── uint32_t magic     // 'BWGT' (0x54475642)
├── uint32_t size      // Byte count: numWeightEntries × 4
└── uint32_t weights[numWeightEntries]
```

**Size Calculation**: `size = numWeightEntries × 4` (each weight is 4 bytes)

---

## Trailing Fields (After All Sub-Chunks)

**Critical**: After all tagged sub-chunks, there are **unmarked binary fields** that must be parsed based on fixed positions.

### Exact Binary Layout

```
Post-SubChunk Data:
├── uint32_t materialId           // Material layer index (0-based)
├── uint32_t selectionGroup      // Selection/render group
├── uint32_t flags               // Geoset flags
├── CMdlBounds bounds            // Bounding volume (24 bytes)
│   ├── float min[3];           // Minimum bounds (x, y, z)
│   ├── float max[3];          // Maximum bounds (x, y, z)
│   └── float radius;          // Bounding sphere radius
├── uint32_t numSeqBounds        // Number of animation bounds
└── CMdlBounds seqBounds[numSeqBounds]  // Animation-specific bounds
```

### CMdlBounds Structure

```c
struct CMdlBounds {
    float min[3];    // Offset +0x00: Minimum bounding box corner
    float max[3];    // Offset +0x0C: Maximum bounding box corner
    float radius;   // Offset +0x18: Bounding sphere radius
};  // Total size: 24 bytes (0x18)
```

### Total Trailing Fields Size

```
Fixed portion: 4 + 4 + 4 + 24 = 36 bytes
Variable portion: 24 × numSeqBounds bytes
```

---

## Layer Sub-Chunk Structure (Within MTLS Chunk)

### Material Layer Entry

```
MTLS Chunk (Per Entry):
├── uint32_t size              // 40 bytes (0x28) fixed per layer
├── uint32_t priorityPlane    // Render priority
├── uint32_t blendMode        // Blend operation
├── uint32_t texCoordId      // Texture coordinate set index
├── uint32_t materialId      // Material/texture ID
└── ... more fields
```

**Key Finding**: `layerSize = 40 bytes (0x28)` - This is a **fixed size per layer**, not derived from the size field.

---

## Code Evidence: Sub-Chunk Count Field Interpretation

### MATS Chunk Processing (LoadGeosetTransformGroups)

```c
// From LoadGeosetTransformGroups at 0x0044f570:
if (*piVar4 != 0x5354414d) {  // 'MATS' magic
    // Error check
}
uVar1 = piVar4[1];  // Count from chunk
// ...
piVar6 = piVar4 + 2;
piVar8 = *(int **)(param_3 + 0x6c);
for (; uVar1 != 0; uVar1 = uVar1 - 1) {
    *piVar8 = *piVar6;  // Copy uint32 per entry
    piVar6 = piVar6 + 1;
    piVar8 = piVar8 + 1;
}
```

**Evidence**: `uVar1` (count) is used as an **element count**, iterating one `uint32` per iteration.

---

### BIDX Chunk Processing

```c
// From LoadGeosetTransformGroups:
if (*piVar4 != 0x43495442) {  // 'BIDX' magic
    // Error check
}
uVar1 = piVar4[1];  // Count from chunk
piVar4 = piVar4 + 2;
// Copy uint32 per entry
piVar6 = piVar4;
piVar8 = *(int **)(param_3 + 0x8c);
for (; uVar1 != 0; uVar1 = uVar1 - 1) {
    *piVar8 = *piVar6;
    piVar6 = piVar6 + 1;
    piVar8 = piVar8 + 1;
}
```

**Evidence**: `BIDX` count is an **element count** where each element is 4 bytes (uint32).

---

### BWGT Chunk Processing

```c
// From LoadGeosetTransformGroups:
uVar1 = piVar4[iVar2 + 1];  // Count from chunk
// ...
piVar6 = piVar4 + iVar2 + 2;
piVar8 = *(int **)(param_3 + 0x9c);
for (; uVar1 != 0; uVar1 = uVar1 - 1) {
    *piVar8 = *piVar6;  // Copy uint32 per entry
    piVar6 = piVar6 + 1;
    piVar8 = piVar8 + 1;
}
```

**Evidence**: `BWGT` count is an **element count** where each element is 4 bytes (uint32).

---

## Summary: Size vs. Count Fields

| Chunk | Size Field | Count Interpretation |
|-------|------------|---------------------|
| VRTX | Byte count | `numVertices = size / 12` |
| NRMS | Byte count | `numNormals = size / 12` |
| UVAS | Byte count | `numUVSets = size / (vertices × 8)` |
| PTYP | Byte count | `numPrimTypes = size` (1 byte each) |
| PCNT | Byte count | `numPrimTypes = size / 4` |
| PVTX | Byte count | `numIndices = size / 2` |
| GNDX | Byte count | `numVertices = size` (1 byte each) |
| MTGC | Byte count | `numLayers = size / 4` |
| MATS | Element count | Each entry = 4 bytes (uint32) |
| BIDX | Element count | Each entry = 4 bytes (uint32) |
| BWGT | Element count | Each entry = 4 bytes (uint32) |

---

## Validation Checks in Code

From [`LoadGeosetData`](LoadGeosetData) at 0x0044eec0:

```c
// Vertex count validation
if (0xffff < uVar5) {
    // Error: Too many vertices (> 65535)
}

// Normal count must match vertex count
if (numVertices != uVar5) {
    // Error: Normal count mismatch
}

// UVAS/TVAS count validation
if (numMappingChannels <= i) {
    // Error: UV set index out of bounds
}
```

From [`LoadGeosetPrimitiveData`](LoadGeosetPrimitiveData) at 0x0044f330:

```c
// Primitive count validation
if (uVar2 != uVar1) {
    // Error: numPrimCounts != numPrimTypes
}

// Index count must match geoset vertex count
if (uVar3 != *(uint *)(param_2 + 0x58)) {
    // Error: Index count mismatch
}
```

---

## Binary Example

```
GEOS 0x4eba0: Reading geoset data
├── VRTX chunk (0x58545256)
│   ├── size: 0x1770 (6000 vertices × 12 bytes)
│   └── vertices: C3Vector[6000]
├── NRMS chunk (0x534d524e)
│   ├── size: 0x1770 (6000 normals × 12 bytes)
│   └── normals: C3Vector[6000]
├── UVAS chunk (0x53415655)
│   ├── size: 0x2ee0 (1 UV set × 6000 vertices × 8 bytes)
│   └── uvData: C2Vector[6000]
├── PTYP chunk (0x50595454)
│   ├── size: 0x02 (2 primitive types)
│   └── primTypes: [0x04, 0x05]
├── PCNT chunk (0x544e4350)
│   ├── size: 0x08 (2 counts × 4 bytes)
│   └── primCounts: [2000, 0]
├── PVTX chunk (0x58545650)
│   ├── size: 0x3A20 (6000 indices × 2 bytes)
│   └── indices: uint16[6000]
└── Trailing fields:
    ├── materialId: 0x00000001
    ├── selectionGroup: 0x00000000
    ├── flags: 0x00000000
    ├── bounds.min: [0.0, 0.0, 0.0]
    ├── bounds.max: [10.0, 10.0, 10.0]
    ├── bounds.radius: 8.66
    ├── numSeqBounds: 0x00000000
    └── (no seqBounds)
```
