# MDX Doodad Placement Analysis - Ghidra Findings

## Summary

Analysis of WoW Alpha 0.5.3 (Build 3368) binary via Ghidra reveals key details about MDX doodad placement transformation. This document is a work-in-progress as analysis continues.

## Key Findings So Far

### 1. Doodad Creation Pipeline

#### `CMap::CreateDoodadDef` (0x00680300)
**Location**: Called from `CreateMapObjDefGroupDoodads`

**Function Signature**:
```cpp
CMapDoodadDef * __fastcall CMap::CreateDoodadDef(
    char *param_1,           // Doodad name/path
    C3Vector *param_2,       // Position
    float param_3,           // Rotation angle (degrees)
    int param_4              // Flags
)
```

**Matrix Construction** (PSEUDOCODE):
```cpp
// Initialize as identity matrix
mat = IdentityMatrix();

// Set position from C3Vector (x, y, z)
mat.d = position;  // Translation component

// Rotation about Z-axis by angle (param_3)
local_1c = (0, 0, 1);  // Z-axis rotation
NTempest::C44Matrix::Rotate(this, param_3, &local_1c, true);

// Also stores: field_0x28 (scale = 1.0)
```

**CRITICAL OBSERVATION**: The rotation is a **single angle about Z-axis**, not 3-axis Euler angles!

### 2. CMapDoodadDef Structure

```cpp
class CMapDoodadDef {
    C44Matrix mat;           // 0x00 - World transform matrix
    C44Matrix lMat;          // 0x30 - Local/lighting matrix
    char *modelName;         // 0x78 - MDX file path
    HMODEL__ *model;         // 0x90 - Loaded model handle
    float scale;             // 0x28 - Scale factor
    uint flags;              // 0x74 - Rendering flags
    CAaBox collideExt;       // Collision extents
    // ... other fields
};
```

### 3. Matrix Layout (C44Matrix / C34Matrix)

The client uses **row-major** matrix layout (Direct3D convention):

```
| a0  a1  a2  a3 |   Row 0: X basis vector + perspective
| b0  b1  b2  b3 |   Row 1: Y basis vector + perspective  
| c0  c1  c2  c3 |   Row 2: Z basis vector + perspective
| d0  d1  d2  d3 |   Row 3: Translation + perspective
```

**Identity Matrix**:
```
| 1   0   0   0 |
| 0   1   0   0 |
| 0   0   1   0 |
| 0   0   0   1 |
```

### 4. Rendering Pipeline

**Doodad Rendering** (`RenderDoodads` @ 0x0066d8a0):
1. Gets doodad's world matrix (`pCVar5->mat`)
2. Adjusts for camera position
3. Calls `ModelAnimate()` with transform
4. Calls `ModelAddToScene()` to queue for rendering

**Model Rendering** (`IModelSimpleAddToScene` @ 0x0042eb40):
1. Calls `AddAllGeosetsToScene()` with transform matrix
2. Each geoset queued with world transform
3. Actual D3D rendering happens later in pipeline

## Critical Questions Remaining

1. **Where do the 3 MDDF rotation values (X, Z, Y in file) get converted to a single Z-axis rotation?**

2. **What about the scale factor from MDDF?** (uint16 at offset 0x22, 1024 = 1.0)

3. **How does this differ from WMO (MODF) placement?**

4. **Is there model-space pivot adjustment from MDX PIVT/BONE data?**

## Next Steps

- [ ] Find MDDF chunk parsing function
- [ ] Analyze how rotation values are converted
- [ ] Compare with MODF/WMO placement
- [ ] Check for PIVT/bone transform application
- [ ] Document complete transform pipeline

## Related Functions Found

| Function | Address | Purpose |
|----------|---------|---------|
| `CreateDoodadDef` | 0x00680300 | Creates doodad with matrix |
| `LoadDoodadModel` | 0x00680c80 | Loads MDX model |
| `InitializeDoodadBounds` | 0x00680b70 | Computes bounding box |
| `RenderDoodads` | 0x0066d8a0 | Renders all doodads |
| `CreateMapObjDefGroupDoodads` | 0x006817c0 | Parses MDDF, creates doodads |
| `ModelAddToScene` | 0x0042ecf0 | Queues model for rendering |
| `AddGeosetToScene` | 0x0042e1f0 | Queues geoset for rendering |

## Observations on Coordinate Systems

**File Format** (MDDF):
- Position: (X, Z, Y) - C3Vector layout
- Rotation: (rotX, rotZ, rotY) - degrees

**Internal Matrix**:
- Single rotation about Z-axis (0, 0, 1)
- This suggests either:
  1. A conversion from 3-axis to 1-axis occurs
  2. The input rotation IS already a single angle
  3. The other rotation components are unused

**World Coordinates**:
- X = North
- Y = West  
- Z = Up
- Left-handed Direct3D convention

---

*Analysis ongoing. Next: Find MDDF parsing and rotation conversion.*
