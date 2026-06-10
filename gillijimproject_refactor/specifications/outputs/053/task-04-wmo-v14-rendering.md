# Task 4: WMO v14 Rendering Path Analysis

## Overview
Analysis of WMO v14 geometry rendering and transformation in WoW Alpha 0.5.3 (build 3368).

## Key Findings

### CMapObj Class Structure
**Address**: 0x00693190 (constructor)

```c
class CMapObj {
    uint32 field_0x4;
    uint32 field_0x8;
    uint32 field_0xc;
    uint32 field_0x10;
    
    CImVector ambColor;     // Ambient color (RGBA)
    
    CAaBox aaBox;           // Axis-aligned bounding box
                            //   .b = bottom point (x, y, z)
                            //   .t = top point (x, y, z)
    
    TSLink<CMapObj> lameAssLink;
    
    struct {
        IffChunkVersion iffChunkVersion;  // Version chunk
        IffChunkHeader iffChunkHeader;    // Header chunk
    } fileHeader;
    
    TSList<CMapObjGroup> groupList;       // List of WMO groups
                                          // field_0x0 = 0x1ac (identifier)
    
    TSFixedArray<CMapObjGroup*> groupPtrList;
                                          // m_count = 0x180 (384 groups max)
};
```

### WMO Group System

From the constructor:
- **Max Groups**: 0x180 = 384 groups per WMO
- **Group List**: Uses TSList for dynamic group management
- **Group Pointers**: Fixed array of 384 pointers

### Related Functions

WMO-related functions found:
- `CMapObj` @ 0x00693190 (constructor)
- `~CMapObj` @ 0x00693220 (destructor)
- `AllocMapObj` @ 0x0068fc40 (allocation)
- `CMapObjDef` @ 0x006ac280 (WMO placement/definition - MODF data)
- `CMapObjGroup` @ 0x0068b610 (individual WMO group - MOGP data)
- `AllocMapObjGroup` @ 0x0068fdb0 (group allocation)

### WMO Placement (CMapObjDef)

CMapObjDef likely stores MODF (Map Object Definition) data:
- Position
- Rotation
- Scale
- Bounding box
- Flags

### WMO Groups (CMapObjGroup)

CMapObjGroup likely handles MOGP (Map Object Group) data:
- Vertices
- Normals
- Texture coordinates
- Triangles
- Materials
- BSP tree
- Liquid data (MLIQ)

### Coordinate System Notes

From the prompt:
- **WoW Coordinate System**: X=North, Y=West, Z=Up
- **File Format**: Positions stored as (X, Z, Y) with Z (height) in the middle

### Transformation Pipeline (Preliminary)

Based on structure analysis, the likely pipeline is:

```
1. Load WMO file
   ↓
2. Parse MOHD (header)
   ↓
3. Load Groups (MOGP chunks)
   - Store in groupList
   - Create group pointers in groupPtrList
   ↓
4. Apply MODF placement transform
   - Position (X, Z, Y)
   - Rotation (likely quaternion or Euler)
   - Scale
   ↓
5. Transform vertices to world space
   - Local → World
   ↓
6. Render
```

### Bounding Box

The `CAaBox` structure stores:
```c
struct CAaBox {
    C3Vector b;  // Bottom corner (min x, y, z)
    C3Vector t;  // Top corner (max x, y, z)
};
```

This is standard AABB (Axis-Aligned Bounding Box) format used for culling.

### Rendering Functions

Functions that may handle WMO rendering:
- `CWorldScene::AddChunkLiquid` (adds WMO liquids via MLIQ)
- `QueryLiquidStatus` variants (collision detection for WMO liquids)
- Scene graph integration via TSLink structures

## Preliminary Structure Definitions

```c
struct IffChunkVersion {
    uint32 token;      // Chunk FourCC
    uint32 size;       // Chunk size
};

struct IffChunkHeader {
    uint32 token;      // Chunk FourCC (MOHD?)
    uint32 size;       // Chunk size
};

class CMapObj {
    // Fields as shown above
};

class CMapObjGroup {
    // WMO group data (MOGP)
    // Likely contains:
    //   - Vertex buffer
    //   - Index buffer
    //   - Material info
    //   - BSP tree
    //   - Liquid data (MLIQ reference)
};

class CMapObjDef {
    // WMO placement data (MODF)
    // Likely contains:
    //   - CMapObj* wmoReference
    //   - C3Vector position
    //   - C4Quaternion rotation (or Euler angles)
    //   - float scale
    //   - uint32 flags
    //   - CAaBox bounds
};
```

## MLIQ String Reference

Found string: `"pIffChunk->token == 'MLIQ'"` @ 0x008a2930

This confirms:
- **MLIQ chunks** are used for WMO liquids (separate from terrain MCLQ)
- FourCC stored in forward byte order: 0x4D4C4951 ("MLIQ")

## Next Steps

To complete WMO v14 rendering documentation, we need to:
1. Find WMO loading function (load .wmo file)
2. Trace MOHD header parsing
3. Trace MOGP group parsing
4. Find vertex transformation code (local → world)
5. Identify rotation application (quaternion vs. Euler, order of operations)
6. Check for coordinate swaps or handedness corrections
7. Document the complete rendering pipeline

## Cross-References

Functions related to WMO handling:
- `CMapinObj` @ 0x00693190 (constructor)
- `~CMapObj` @ 0x00693220 (destructor)
- `AllocMapObj` @ 0x0068fc40 (allocation)
- `CMapObjGroup` @ 0x0068b610 (group handling)
- `CMapObjDef` @ 0x006ac280 (placement definition)

String references:
- MLIQ detection @ 0x008a2930

## Confidence Level

**Medium** - We have determined:
- ✅ CMapObj class structure with bounding box, ambient color
- ✅ WMO supports up to 384 groups (0x180)
- ✅ Group management via TSList
- ✅ File header contains IFF chunk structure
- ✅ MLIQ used for WMO liquids
- ✅ Coordinate system: X=North, Y=West, Z=Up

Still investigating:
- ⏳ Complete WMO loading pipeline
- ⏳ Vertex transformation matrices
- ⏳ Rotation application (quaternion format, order)
- ⏳ Handedness corrections (if any)
- ⏳ Coordinate swap logic during loading
- ⏳ Complete MODF structure
- ⏳ Complete MOGP structure

## Differences from Later WoW Versions

- **Alpha 0.5.3**: WMO is v14, monolithic single file (.wmo)
- **Later (v17+)**: WMO split into root file (.wmo) and group files (_###.wmo)
- **Alpha**: Likely simpler group structure (fewer features)
- **Chunk Order**: Forward FourCC (MLIQ = 0x4D4C4951), not reversed like LK
