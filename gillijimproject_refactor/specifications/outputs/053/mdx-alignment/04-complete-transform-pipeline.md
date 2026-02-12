# Complete MDX Doodad Transform Pipeline

## Executive Summary

This document consolidates all findings from Ghidra analysis of WoW Alpha 0.5.3 (Build 3368) regarding MDX doodad placement transformation.

## The Complete Pipeline

### Stage 1: File Parsing (MDDF Chunk → Runtime Data)

**Unknown Code Location** - MDDF chunk parsing not found in binary.

```
MDDF Entry (36 bytes from file):
┌─────────────────────────────────────────────────────────────┐
│ Offset  Size  Field                                       │
├─────────────────────────────────────────────────────────────┤
│ 0x00    4     nameIndex (uint32)                          │
│ 0x04    4     uniqueId (uint32)                           │
│ 0x08    4     position.X (float) - WoW X (North)          │
│ 0x0C    4     position.Z (float) - WoW Z (Height)         │
│ 0x10    4     position.Y (float) - WoW Y (West)           │
│ 0x14    4     rotation.X (float) - degrees                │
│ 0x18    4     rotation.Z (float) - degrees                │
│ 0x1C    4     rotation.Y (float) - degrees                │
│ 0x20    2     scale (uint16) - 1024 = 1.0                │
│ 0x22    2     flags (uint16)                              │
└─────────────────────────────────────────────────────────────┘
```

**Mystery**: The file stores 3 rotation values, but the placement code uses only 1 angle.

### Stage 2: Runtime Data Structures

**`CMapDoodadDef` Structure** (after parsing):
```cpp
class CMapDoodadDef {
    C44Matrix mat;           // 0x00 - World transform matrix (4x4)
    C44Matrix lMat;          // 0x30 - Local/lighting matrix
    C3Vector position;        // 0x1C - Parsed position
    float scale;             // 0x28 - Scale factor (1.0 = default)
    char *modelName;         // 0x78 - MDX file path
    HMODEL__ *model;         // 0x90 - Loaded model handle
    uint flags;              // 0x74 - Rendering flags
    CAaBox collideExt;       // Collision extents
    // ... other fields
};
```

### Stage 3: Matrix Construction

**`CMap::CreateDoodadDef`** (0x00680300):

```cpp
void __fastcall CMap::CreateDoodadDef(
    CMapDoodadDef *this,
    C3Vector *position,      // Parsed from MDDF
    float rotationAngle,      // SINGLE ANGLE IN DEGREES!
    int flags
) {
    // Step 1: Identity Matrix
    this->mat = Identity();
    
    // Step 2: Translation
    // D3D row-major: translation is in d0, d1, d2
    this->mat.d0 = position->x;
    this->mat.d1 = position->y;
    this->mat.d2 = position->z;
    
    // Step 3: Z-Axis Rotation (Rodrigues formula)
    C3Vector axis = {0, 0, 1};  // Z-axis
    C44Matrix::Rotate(&this->mat, rotationAngle, &axis, true);
    
    // Scale is stored separately (field_0x28 = 1.0)
}
```

**Matrix Format (D3D Row-Major):**
```
| a0  a1  a2  a3 |   Row 0: X basis vector
| b0  b1  b2  b3 |   Row 1: Y basis vector  
| c0  c1  c2  c3 |   Row 2: Z basis vector
| d0  d1  d2  d3 |   Row 3: Translation
```

**Identity Matrix:**
```
| 1   0   0   0 |
| 0   1   0   0 |
| 0   0   1   0 |
| 0   0   0   1 |
```

### Stage 4: Rotation Application (Rodrigues Formula)

**`C34Matrix::Rotation`** (0x00493eb0) - Axis-Angle to Matrix:

```cpp
// Input: angle in RADIANS, axis vector
C34Matrix Rotation(float angle, C3Vector axis, bool normalized) {
    // Normalize axis if not already normalized
    if (!normalized) {
        axis = normalize(axis);
    }
    
    // Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    float s = sin(angle);  // <-- angle must be in RADIANS!
    float c = cos(angle);
    float t = 1 - c;
    
    // Build rotation matrix
    mat.a0 = t*axis.x*axis.x + c;
    mat.a1 = t*axis.x*axis.y + s*axis.z;
    mat.a2 = t*axis.x*axis.z - s*axis.y;
    mat.b0 = t*axis.x*axis.y - s*axis.z;
    mat.b1 = t*axis.y*axis.y + c;
    mat.b2 = t*axis.y*axis.z + s*axis.x;
    mat.c0 = t*axis.x*axis.z + s*axis.y;
    mat.c1 = t*axis.y*axis.z - s*axis.x;
    mat.c2 = t*axis.z*axis.z + c;
    
    return mat;
}
```

### Stage 5: Matrix Multiplication (Apply Rotation)

**`C34Matrix::Rotate`** (0x004941d0):

```cpp
void C34Matrix::Rotate(C34Matrix *this, float angle, C3Vector *axis, bool norm) {
    C34Matrix rot;
    C34Matrix temp;
    
    // Create rotation matrix
    Rotation(&rot, angle, axis, norm);
    
    // Multiply: result = rot × this
    // This applies rotation AFTER the current transform
    operator*(&temp, rot, this);
    
    // Copy result back
    *this = temp;
}
```

### Stage 6: Rendering

**`RenderDoodads`** (0x0066d8a0):

```cpp
void RenderDoodads() {
    for each visible doodad {
        // Get doodad's world matrix
        C44Matrix *worldMat = &doodad->mat;
        
        // Adjust for camera position (relative transform)
        C44Matrix renderMat = *worldMat;
        renderMat.d0 -= camPos.x;
        renderMat.d1 -= camPos.y;
        renderMat.d2 -= camPos.z;
        
        // Animate model (bones, particles, etc.)
        ModelAnimate(model, &renderMat, scale, &camPos, &camTarget);
        
        // Queue for rendering
        ModelAddToScene(model, renderFlags);
    }
}
```

**`ModelAddToScene`** (0x0042ecf0):
- Queues model for rendering
- Passes world matrix to GPU pipeline
- Actual D3D `SetTransform` happens later

## Critical Missing Pieces

### 1. Degree-to-Radian Conversion

The `Rotation()` function takes **radians**, but MDDF stores **degrees**.

```cpp
// IN Rotation():
float s = fsin(angle);  // expects RADIANS

// BUT in CreateDoodadDef:
float rotationAngle;    // from MDDF, in DEGREES
C44Matrix::Rotate(&mat, rotationAngle, &axis, true);
```

**Where does the conversion happen?**

### 2. 3-to-1 Rotation Conversion

MDDF stores 3 rotation values, but only 1 angle is used.

```cpp
// MDDF stores:
float rotation[3];  // X, Z, Y in file

// But CreateDoodadDef receives:
float rotationAngle;  // Single value!
```

**How are the 3 values converted to 1?**

### 3. Coordinate System Mapping

**File Coordinates (WoW):**
- X = North
- Y = West
- Z = Up

**Matrix Operations:**
- Translation uses (x, y, z) directly
- Rotation is about Z-axis (0, 0, 1)

**Is there a coordinate remap?**

## Comparison: MDX vs WMO Placement

| Aspect | MDX (Doodad) | WMO |
|--------|--------------|-----|
| Function | `CreateDoodadDef` | `CreateMapObjDef` |
| Translation | Yes | Yes |
| Rotation | Z-axis only | Z-axis only |
| Scale | Field 0x28 | Unknown |
| Inverse Matrix | No | Yes (for collision) |
| Matrix Layout | Row-major D3D | Row-major D3D |

**Both use identical matrix construction!**

## Coordinate System Summary

**WoW World Coordinates:**
```
X = North (increases going north)
Y = West  (increases going west)
Z = Up    (increases going up)
```

**D3D Rendering Coordinates:**
```
Matrix stores (x, y, z) directly
Rotation about Z-axis (0, 0, 1)
Row-major layout
```

**Transformation Order:**
```
WorldVertex = RotationMatrix × Translation × ModelVertex
```

## Implementation for Renderer

```cpp
// Pseudo-code for correct MDX placement:

// 1. Parse MDDF entry
MDDFEntry entry = read_mddf(file);

// 2. Get rotation angle (MYSTERY: how to get single angle from 3 values?)
// Hypothesis: use only entry.rotation[0] (X rotation)
float angleDeg = entry.rotation[0];

// 3. Convert to radians
float angleRad = angleDeg * PI / 180.0f;

// 4. Create rotation matrix (Rodrigues about Z-axis)
C44Matrix rotation = rotation_rodrigues(angleRad, (0, 0, 1));

// 5. Create translation
C44Matrix translation = identity();
translation.d0 = entry.position.x;
translation.d1 = entry.position.y;
translation.d2 = entry.position.z;

// 6. Combine: world = rotation × translation
C44Matrix world = multiply(rotation, translation);

// 7. Apply scale
float scale = entry.scale / 1024.0f;
world = scale_matrix(world, scale);

// 8. Apply to model vertices
// VertexWorld = world × vertexModel
```

## Related Documents

- [01-overview.md](01-overview.md) - Initial findings
- [02-rotation-analysis.md](02-rotation-analysis.md) - Rotation analysis
- [03-rotation-rodrigues.md](03-rotation-rodrigues.md) - Rodrigues formula details

## Outstanding Questions

1. **Where is the MDDF chunk parser?**
2. **How are degrees converted to radians?**
3. **How are 3 rotation values converted to 1?**
4. **What coordinate remapping is applied (if any)?**
5. **How does scale interact with rotation/translation?**

## Functions Analyzed

| Function | Address | Purpose |
|----------|---------|---------|
| `CMap::CreateDoodadDef` | 0x00680300 | Construct doodad matrix |
| `CMap::CreateMapObjDef` | 0x00680f50 | Construct WMO matrix |
| `C34Matrix::Rotation` | 0x00493eb0 | Axis-angle to matrix |
| `C34Matrix::Rotate` | 0x004941d0 | Apply rotation |
| `RenderDoodads` | 0x0066d8a0 | Render all doodads |
| `ModelAddToScene` | 0x0042ecf0 | Queue model for render |
| `fsin`/`fcos` | - | Math library |
