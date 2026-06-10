# MDX Doodad Placement Analysis - Complete Documentation

## Overview

Analysis of WoW Alpha 0.5.3 (Build 3368) binary via Ghidra to understand MDX/M2 doodad placement transformation.

## Documents

### Core Analysis

| Document | Description |
|----------|-------------|
| [01-overview.md](01-overview.md) | Initial findings - key functions and structures |
| [02-rotation-analysis.md](02-rotation-analysis.md) | Critical finding: single vs 3-axis rotation |
| [03-rotation-rodrigues.md](03-rotation-rodrigues.md) | Rodrigues rotation formula implementation |
| [04-complete-transform-pipeline.md](04-complete-transform-pipeline.md) | Complete end-to-end transform pipeline |

### Related

| Document | Description |
|----------|-------------|
| [specifications/outputs/053/lighting/01-lighting-system.md](../lighting/01-lighting-system.md) | Lighting system (.lit files, fog) |

## Key Findings Summary

### 1. Rotation Format

**CRITICAL**: Both MDX and WMO use **single-angle Z-axis rotation**, not 3-axis Euler angles!

```cpp
// Both use:
C44Matrix::Rotate(matrix, angle, (0, 0, 1), true);
// Rotation about Z-axis by a single angle
```

### 2. Matrix Construction

```
1. Identity Matrix
   ↓
2. Translation (position.x, position.y, position.z)
   ↓
3. Z-Axis Rotation (Rodrigues formula)
   ↓
4. World Matrix
```

### 3. Rodrigues Formula

The client uses Rodrigues' rotation formula:
```
R = I + sin(θ)K + (1-cos(θ))K²
```

Where K is the cross-product matrix of the rotation axis.

### 4. Outstanding Mysteries

1. **Degree-to-Radian conversion location** - Not found in code
2. **3-to-1 rotation conversion** - MDDF has 3 values, but only 1 used
3. **MDDF chunk parser** - Not located in binary

## Coordinate Systems

### WoW World Coordinates
- X = North
- Y = West
- Z = Up

### D3D Matrix (Row-Major)
```
| a0  a1  a2  a3 |   Row 0: X basis
| b0  b1  b2  b3 |   Row 1: Y basis
| c0  c1  c2  c3 |   Row 2: Z basis
| d0  d1  d2  d3 |   Row 3: Translation
```

## Functions Analyzed

| Function | Address | Purpose |
|----------|---------|---------|
| `CMap::CreateDoodadDef` | 0x00680300 | Create doodad with matrix |
| `CMap::CreateMapObjDef` | 0x00680f50 | Create WMO with matrix |
| `CMap::UpdateDoodadDef` | 0x006857d0 | Update doodad transform |
| `C34Matrix::Rotation` | 0x00493eb0 | Axis-angle to matrix |
| `C34Matrix::Rotate` | 0x004941d0 | Apply rotation to matrix |
| `RenderDoodads` | 0x0066d8a0 | Render all doodads |
| `ModelAddToScene` | 0x0042ecf0 | Queue model for render |
| `LoadLightsAndFog` | 0x006c4110 | Load .lit lighting |

## Implementation Guide

```cpp
// Complete MDX placement transform (pseudo-code):

// 1. Parse MDDF entry
float posX = entry.position[0];  // North
float posY = entry.position[2];  // Up (height)
float posZ = entry.position[1];  // West

// 2. Get rotation (MYSTERY: how to convert 3 values to 1?)
float angleDeg = entry.rotation[0];  // Hypothesis: use X rotation
float angleRad = angleDeg * PI / 180.0f;

// 3. Create Z-axis rotation matrix (Rodrigues)
C44Matrix rot = rotation_rodrigues(angleRad, (0, 0, 1));

// 4. Create translation
C44Matrix trans = identity();
trans.d0 = posX;
trans.d1 = posY;
trans.d2 = posZ;

// 5. World = rot × trans
C44Matrix world = multiply(rot, trans);

// 6. Apply scale
float scale = entry.scale / 1024.0f;
world = scale_matrix(world, scale);

// 7. Render with world matrix
// VertexWorld = world × VertexModel
```

## Path-Based Rendering

**FINDING**: No evidence of directory-based rendering in Alpha 0.5.3.

- Models render based on material data, not file path
- No "Detail/" or "NoDxt/" directory processing found
- Transparency/alpha is material-based, not path-based

## Related Specifications

- [Alpha ADT Format](../../alpha-053-terrain.md)
- [Alpha WMO Format](../alpha-wmo-spec.md)
- [MDX Format Reference](../../reference_data/wowdev.wiki/MDX.md)

## Next Steps

1. Locate MDDF chunk parser to find rotation conversion
2. Verify degree-to-radian conversion location
3. Test implementation with sample data
4. Compare with working WMO transform
