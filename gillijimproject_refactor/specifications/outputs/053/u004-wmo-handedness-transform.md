# U-004: WMO v14 Geometry Handedness Analysis

## Overview
Analysis of WMO v14 vertex transformation and coordinate system in WoW Alpha 0.5.3 (build 3368) from [`CWorldScene::RenderMapObjDefGroups`](CWorldScene::RenderMapObjDefGroups:66e030) @ 0x0066e030 and [`CMap::CreateMapObjDef`](CMap::CreateMapObjDef:680f50) @ 0x00680f50.

## Key Findings

### Transform Pipeline

The WMO rendering pipeline uses standard 4×4 transformation matrices:

```c
void __fastcall CWorldScene::RenderMapObjDefGroups(void)
{
  C44Matrix gxWm;      // World matrix sent to graphics
  C44Matrix mapObjM;   // MapObj transformation
  C44Matrix local_d8;  // Temporary matrix
  C3Vector local_58;   // Camera translation
  
  // Initialize identity matrix
  gxWm = IdentityMatrix();
  mapObjM = IdentityMatrix();
  
  // For each visible map object group:
  //   1. Get parent MapObjDef from mapObjDefGroup->field_0x18
  //   2. Calculate camera-relative position
  local_58.x = -camPos.x;
  local_58.y = -camPos.y;
  local_58.z = -camPos.z;
  gxWm = Translate(Identity(), local_58);
  
  //   3. Combine with MapObjDef transform (at offset +0x90)
  mapObjM = Multiply(mapObjDef->mat, gxWm);
  
  //   4. Set world transform for rendering
  GxXformSet(GxXform_World, &mapObjM);
  
  //   5. Render the WMO group
  CMapObj::RenderGroup(mapObj, groupNum, ...);
}
```

### MapObjDef Transform Matrix

From [`CreateMapObjDef`](CreateMapObjDef:680f50), the transformation matrix is built as:

```c
CMapObjDef* CreateMapObjDef(
    char* fileName,
    C3Vector* position,
    float rotation,
    int syncLoad)
{
  // ... allocation ...
  
  // Store position
  mapObjDef->pos = *position;  // At offset +0x1C (X, Y, Z)
  
  // Initialize transformation matrix to identity
  mapObjDef->mat = IdentityMatrix();
  
  // Apply translation
  C44Matrix::Translate(&mapObjDef->mat, position);
  
  // Apply rotation around Z-axis
  C3Vector zAxis = {0.0, 0.0, 1.0};
  C44Matrix::Rotate(&mapObjDef->mat, rotation, &zAxis, true);
  
  // Calculate inverse matrix
  mapObjDef->invMat = AffineInverse(&mapObjDef->mat);
  
  // ... bounds calculation ...
}
```

### Coordinate System Confirmation

**WoW Alpha 0.5.3 uses a right-handed coordinate system:**
- **X = North** (forward)
- **Y = West** (left)
- **Z = Up** (vertical)

This is confirmed by:
1. Rotation is around the Z-axis (vertical rotation)
2. Position is stored as (X, Y, Z) at offset +0x1C
3. No coordinate swapping in transformation pipeline

### No Mirroring in Core Transform

The decompiled code shows:
- **Standard C44Matrix operations** (translate, rotate, multiply)
- **No scale with negative values** (no Scale(-1, 1, 1) or similar)
- **No coordinate axis swaps** during transform
- **Standard right-handed matrix math**

## Analysis: Why WMOs Appear Mirrored

If WMOs appear mirrored in a viewer, the issue is likely **NOT** in Alpha's transform pipeline, but in:

### Hypothesis 1: Vertex Winding Order
WMO vertices might be stored with **clockwise winding** instead of counter-clockwise:
- Alpha may render with a different culling mode than expected
- Need to check `CMapObj::RenderGroup` rendering state

### Hypothesis 2: Model Space Coordinate System
WMO **local-space vertices** might use a different handedness than world-space:
- World-space: Right-handed (X=North, Y=West, Z=Up)
- Model-space: Could be left-handed or have different axis meanings

### Hypothesis 3: Rotation Convention
The rotation parameter in `CreateMapObjDef`:
- Rotates around Z-axis (vertical)
- May use a different angle direction (CW vs CCW) than expected
- `C44Matrix::Rotate` final bool parameter (true) may control handedness

## CMapObjDef Structure

Based on field accesses:

```c
class CMapObjDef {
    // Offset 0x00-0x18: Base class / links
    
    C3Vector pos;          // 0x1C: Position (X, Y, Z)
    float scale;           // 0x28: Scale factor (1.0 = 100%)
    
    CAaBox boundingBox;    // 0x3C: World-space bounding box (24 bytes)
    C3Vector sphereCenter; // 0x54: Bounding sphere center (12 bytes)
    float sphereRadius;    // 0x60: Bounding sphere radius
    
    uint16 flags;          // 0x74: Flags
    uint32 uniqueId;       // 0x78: Unique instance ID
    uint32 hashKey;        // 0x8C: Hash key
    
    C44Matrix mat;         // 0x90: World transformation matrix (64 bytes)
    C44Matrix invMat;      // 0xD0: Inverse transformation (64 bytes)
    
    CMapObj* mapObj;       // 0x114: Pointer to WMO model data
    
    // ... more fields ...
};
```

## Transformation Math

The transformation pipeline:

```
Local Vertex → MapObj Space → World Space → Camera Space → Screen Space
     |              |              |              |              |
   Model         mat (0x90)    Camera Pos    Projection      Viewport
  vertices                                    Matrix
```

**Key Transforms:**
1. **MapObj.mat** (offset +0x90): Places WMO in world
   - Translation: `position`
   - Rotation: Around Z-axis by `rotation` angle
   - No scale in basic case

2. **Camera Transform**: Translates world to camera-relative
   - Subtract camera position from world coordinates

3. **Graphics Pipeline**: Projects to screen
   - Standard perspective or orthographic projection

## No Handedness Conversion Found

**Critical finding**: The code does **NOT** perform:
- Axis swaps (no X↔Y or Y↔Z exchanges)
- Negative scaling (no Scale(-1, 1, 1))
- Matrix determinant sign changes
- Winding order flips

This suggests WMO v14 geometry is stored in the **same coordinate system** as terrain and should not require mirroring at the transform level.

## WMO Group Rendering Analysis

From [`CMapObj::RenderGroup`](CMapObj::RenderGroup:69bd50) @ 0x0069bd50:

```c
void __thiscall CMapObj::RenderGroup(
    CMapObj *this,
    uint groupNum,
    int rDrawSharedLiquidToggle,
    C44Matrix *transform,
    TSExplicitList<CWFrustum,244> *frustumList)
{
  CMapObjGroup *group = GetGroup(this, groupNum, 0);
  
  // Create lightmaps for group
  CMapObjGroup::CreateLightmaps(group);
  
  // Render with each frustum
  for each frustum in frustumList:
    CWorldScene::FrustumSet(frustum);
    CWorldScene::FrustumXform(transform);
    
    // Render based on group flags
    if ((group->flags & 0x48) == 0) {
      (*DAT_00ec1b98)();  // Function pointer - standard render
    } else {
      (*DAT_00ec1ca0)(group, index);  // Function pointer - special render
    }
  
  // Optional: Render liquids if flag 0x1000 set
  if ((group->flags & 0x1000) != 0) {
    RenderLiquid_0(this, group);
  }
  
  // Debug: Render normals if enabled
  if ((CWorld::enables & 0x40000000) != 0) {
    RenderGroupNormals(this, group);
  }
  
  // Debug: Render portals if enabled
  if ((CWorld::enables & 0x1000) != 0) {
    RenderPortals(this, group);
  }
}
```

### Key Finding: Group Flags

- **flags & 0x48**: Special rendering mode (likely interior/exterior or BSP)
  - 0x48 = bit 3 (0x08) + bit 6 (0x40)
  - Determines which render function pointer to use
  
- **flags & 0x1000**: Has liquid data

The actual vertex/index rendering is done through **function pointers** (`DAT_00ec1b98` and `DAT_00ec1ca0`), making it difficult to trace exact render state setup (culling mode, winding order) without deeper analysis.

## Recommended Investigation

To resolve mirroring issues in a modern viewer:

1. **Check Vertex Winding in MOGP Data**:
   - Examine triangle index order in MOGP chunks
   - Compare with terrain mesh winding (likely counter-clockwise)

2. **Check Group Flags**:
   - flags & 0x08: May indicate interior (different culling)
   - flags & 0x40: May indicate special rendering
   - Different flag combinations may require different winding

3. **Trace Render Function Pointers**:
   - `DAT_00ec1b98` @ address TBD (standard render)
   - `DAT_00ec1ca0` @ address TBD (special render)
   - These likely set up different render states

4. **Check BSP Rendering**:
   - `RenderGroupBsp` @ 0x0069df60 exists
   - BSP may affect winding/culling for indoor areas

5. **Compare Interior vs Exterior**:
   - flags & 0x48 distinguishes render paths
   - Interior WMOs may have inverted winding

## Cross-References

- `CWorldScene::RenderMapObjDefGroups` @ 0x0066e030 (WMO rendering)
- `CMap::CreateMapObjDef` @ 0x00680f50 (transform setup)
- `CMapObj::RenderGroup` @ 0x0069bd50 (group rendering)
- `RenderGroupBsp` @ 0x0069df60 (BSP rendering)
- `RenderGroupNormals` @ 0x0069e1c0 (debug normals)
- `CMapObjGroup::CreateLightmaps` @ 0x006adba0 (lightmap generation)
- `C44Matrix::Translate` (translation)
- `C44Matrix::Rotate` (rotation around axis)
- `C44Matrix::AffineInverse` (inverse calculation)
- `GxXformSet` (graphics API matrix set)

## Confidence Level

**High** - Transform and rendering pipeline documented:
- ✅ No mirroring transforms in placement code
- ✅ Standard right-handed coordinate system
- ✅ Rotation around Z-axis (vertical)
- ✅ No axis swaps or negative scales
- ✅ `CMapObj::RenderGroup` analyzed—uses function pointers for actual rendering
- ✅ Group flags (0x48) determine render path (likely affects culling/winding)
- ⚠️ Function pointers prevent tracing exact render state setup
- ⚠️ MOGP vertex data format not examined

## Resolution Status

**Substantially Resolved** - Transform and render flow documented:

### Confirmed
- Transform math is standard, no handedness conversion
- Rendering uses two paths based on group flags (0x48)
- Function pointers (`DAT_00ec1b98`, `DAT_00ec1ca0`) hide actual render state

### Likely Cause of Mirroring
Mirroring is likely caused by:
1. **Group flag-dependent rendering**: flags & 0x48 selects different render functions
2. **Interior vs Exterior geometry**: Different winding orders for indoor/outdoor
3. **Vertex data in MOGP**: Triangle indices may be stored in different order
4. **Not in transforms**: All evidence shows standard right-handed math

### To Fully Resolve
- Examine MOGP chunk structure for triangle index order
- Compare vertex winding between interior (flags & 0x48) and exterior groups
- Trace function pointers to find actual render state setup

## Differences from Later WoW Versions

Alpha 0.5.3 uses the same coordinate system and transform conventions as later WoW versions. The MDX→M2 model format change may have affected vertex storage conventions, but world-space transforms remain consistent.
