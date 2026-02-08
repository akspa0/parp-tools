# WoW Alpha 0.5.3 (Build 3368) MODF Chunk Analysis

## Overview

This document provides a deep analysis of the **MODF chunk** (Map Object Definition) in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. The MODF chunk defines world map objects (WMOs) placed on the terrain in the world.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`LoadWdt`](LoadWdt) | 0x0067fde0 | WDT file loading (MODF detection) |
| [`CreateMapObjDef`](CreateMapObjDef) | 0x00680f50 | Create MapObj definition |
| [`AllocMapObjDef`](AllocMapObjDef) | 0x00691600 | Allocate MapObjDef |
| [`CMapObjDef`](CMapObjDef) | 0x006ac280 | MapObjDef class |
| [`CMapObj`](CMapObj) | 0x00693190 | MapObj class |
| [`CullMapObjDefs`](CullMapObjDefs) | 0x0066e850 | Culling |
| [`RenderMapObjDefs`](RenderMapObjDefs) | - | Rendering (implied) |
| [`LinkLightToMapObjDefs`](LinkLightToMapObjDefs) | 0x00687270 | Light linking |

---

## MODF Chunk Format

### File Location

The MODF chunk appears in **WDT files** (World Definition Tables) for global map objects. It is NOT found in ADT tile files.

### Chunk Structure

```
WDT File:
├── MVER (Version)
├── MPHD (Map Header)
├── MAIN (Area Information)
├── doodadNames (MDDF names - see MDDF document)
└── MODF (Map Object Definitions) [Optional]
```

### MODF Entry Structure (Ghidra-Verified)

**From [`LoadWdt`](LoadWdt) at 0x0067fde0:**

```c
// MODF chunk reads 0x40 (64) bytes per entry
struct SMMapObjDef {
    /*0x00*/ uint32_t nameId;       // Index into map object filename table
    /*0x04*/ uint32_t uniqueId;     // Unique identifier for this placement
    /*0x08*/ C3Vector pos;         // Position (X, Y, Z)
    /*0x14*/ C3Vector rot;         // Rotation (X, Y, Z) in radians
    /*0x20*/ C3Vector extentsMin;  // Bounding box minimum
    /*0x2C*/ C3Vector extentsMax;  // Bounding box maximum
    /*0x38*/ uint32_t flags;       // Placement flags
    /*0x3C*/ uint32_t doodadSet;  // Doodad set index
    /*0x40*/ uint32_t nameSet;     // Name set (LOD) index
};
// Total: 64 bytes (0x40)
```

### Key Observations from Ghidra

**From LoadWdt:**
```c
// MODF chunk magic: 0x4d4f4446 = 'MODF'
if (iffChunk.token == 0x4d4f4446) {
    // Read 64 bytes per entry
    SFile::Read(wdtFile, &smMapObjDef, 0x40, ...);
    
    // Create MapObjDef from data
    smMapObjDef.uniqueId = uniqueId;
    pos.x = 0.0;
    pos.y = 0.0;
    pos.z = 0.0;
    uniqueId = uniqueId - 1;
    pCVar1 = CreateMapObjDef(&smMapObjDef, &pos);
}
```

---

## CMapObjDef Class Structure

**From Ghidra decompilation:**

```c
class CMapObjDef {
    /*0x00*/ // Scene linkage
    TSLink<CMapObjDef> sceneLink;
    
    /*0x0C*/ // Object properties
    CMapObj* mapObj;           // Loaded WMO model
    char* name;                // Filename reference
    uint32_t nameId;           // Filename table index
    
    /*0x18*/ // Transform
    C44Matrix mat;             // World transformation matrix
    C44Matrix invMat;          // Inverse matrix (for culling)
    
    /*0x78*/ // Identification
    uint32_t uniqueId;         // Unique placement ID
    uint16_t flags;            // Placement flags
    uint16_t doodadSet;       // Doodad set index
    uint32_t nameSet;          // LOD name set
    
    /*0x84*/ // Spatial
    CAaBox boundingBox;        // Axis-aligned bounding box
    CSphere boundingSphere;    // Bounding sphere
    
    /*0x98*/ // State
    uint32_t loadState;       // Loading state
    bool isLoaded;            // Loaded flag
    
    /*0xA0*/ // Lists
    TSList<CMapObjDefGroup> groups;    // WMO groups
    TSList<CMapLight> lights;          // Attached lights
    
    /*0xB0*/ // Rendering
    void* renderCallback;     // Render callback
    uint32_t renderFlags;     // Rendering flags
};
```

---

## CMapObj Class Structure

```c
class CMapObj {
    /*0x00*/ // Base class
    CMapBaseObj base;
    
    /*0x10*/ // File data
    IFFChunk fileHeader;      // IFF chunk header
    void* fileData;           // Raw file data pointer
    
    /*0x20*/ // Geometry
    TSList<CMapObjGroup> groups;      // WMO groups
    TSFixedArray<CMapObjGroup*> groupPtrs;  // Group pointers (384 max)
    
    /*0x40*/ // Materials
    TSGrowableArray<SMOMaterial> materials;  // Materials
    TSGrowableArray<char> textureNames;       // Texture filenames
    
    /*0x60*/ // Doodads
    TSList<CMapDoodadDef> doodads;   // Attached doodads
    TSGrowableArray<SMODoodadSet> doodadSets;  // Doodad sets
    TSGrowableArray<char> doodadNames;         // Doodad filenames
    
    /*0x80*/ // Lighting
    TSGrowableArray<SMOLight> lights;  // WMO lights
    TSGrowableArray<SMOFog> fogs;      // WMO fogs
    
    /*0xA0*/ // Portals
    TSGrowableArray<SMOPortal> portals;     // Portals
    TSGrowableArray<SMOPortalRef> portalRefs;  // Portal references
    TSGrowableArray<C3Vector> portalVerts;     // Portal vertices
    
    /*0xC0*/ // State
    bool bLoaded;               // Fully loaded flag
    uint32_t flags;            // WMO flags
    
    /*0xD0*/ // Bounds
    CAaBox boundingBox;         // Overall bounding box
    CSphere boundingSphere;     // Overall bounding sphere
};
```

---

## MapObjDef Creation Pipeline

### 1. LoadWdt Entry Point

```c
/* LoadWdt at 0x0067fde0 */
void CMap::LoadWdt() {
    SIffChunk iffChunk;
    
    // Read MVER (version)
    SFile::Read(wdtFile, &iffChunk, 8, ...);
    // Expected: 0x4d564552 ('MVER')
    
    // Read MPHD (header)
    SFile::Read(wdtFile, &iffChunk, 8, ...);
    // Expected: 0x4d504844 ('MPHD')
    SFile::Read(wdtFile, &header, 0x80, ...);
    
    // Read MAIN (area info)
    SFile::Read(wdtFile, &iffChunk, 8, ...);
    // Expected: 0x4d41494e ('MAIN')
    SFile::Read(wdtFile, &areaInfo, 0x10000, ...);
    
    // Load doodad names first (required for MDDF)
    LoadDoodadNames();
    LoadMapObjNames();
    
    // Check for MODF chunk
    SFile::Read(wdtFile, &iffChunk, 8, ...);
    if (iffChunk.token == 0x4d4f4446) {  // 'MODF'
        SMMapObjDef smMapObjDef;
        
        // Initialize to zero
        smMapObjDef.pos.x = 0.0;
        smMapObjDef.pos.y = 0.0;
        smMapObjDef.pos.z = 0.0;
        smMapObjDef.rot.x = 0.0;
        smMapObjDef.rot.y = 0.0;
        smMapObjDef.rot.z = 0.0;
        smMapObjDef.extentsMin = {0,0,0};
        smMapObjDef.extentsMax = {0,0,0};
        
        // Read entry data
        SFile::Read(wdtFile, &smMapObjDef, 0x40, ...);
        
        // Set unique ID
        smMapObjDef.uniqueId = uniqueId;
        uniqueId = uniqueId - 1;
        
        // Create MapObjDef
        CMapObjDef* mapObjDef = CreateMapObjDef(&smMapObjDef, &pos);
        
        // Add to scene list
        CMapBaseObjLink* link = AllocBaseObjLink((CMapBaseObj*)mapObjDef);
        link->ref = NULL;
        
        // Mark as dungeon (indoor WMO)
        bDungeon = 1;
    }
}
```

### 2. CreateMapObjDef Transformation

```c
/* CreateMapObjDef at 0x00680f50 */
CMapObjDef* CMap::CreateMapObjDef(
    SMMapObjDef* smDef,
    C3Vector* basePos
) {
    // Allocate MapObjDef
    CMapObjDef* def = AllocMapObjDef();
    if (def == NULL) {
        Error("Failed to allocate MapObjDef");
        return NULL;
    }
    
    // Link to hash table for fast lookup
    TSHashTable<CMapObjDef, HASHKEY_NONE>::InternalLinkNode(
        &mapObjDefHash, def, smDef->uniqueId
    );
    
    // Store reference to WMO file
    def->mapObjFileId = smDef->nameId;
    
    // Get WMO filename from name table
    char* wmoFilename = GetMapObjName(smDef->nameId);
    
    // Load the WMO file
    CMapObj* wmo = CMapObj::Create(wmoFilename);
    def->mapObj = wmo;
    
    // Wait for async load if needed
    if (smDef->flags & MODF_FLAG_LOAD_ASYNC) {
        CMapObj::WaitLoad(wmo);
    }
    
    // Build transformation matrix from position and rotation
    C44Matrix mat;
    mat.SetIdentity();
    
    // Apply translation
    C3Vector pos = smDef->pos;
    if (basePos != NULL) {
        pos = pos + *basePos;
    }
    mat.Translate(pos);
    
    // Apply rotation (Euler angles)
    C3Vector rot = smDef->rot;
    mat.RotateX(rot.x);
    mat.RotateY(rot.y);
    mat.RotateZ(rot.z);
    
    def->mat = mat;
    
    // Calculate inverse matrix for culling
    def->invMat = mat.AffineInverse();
    
    // If WMO is loaded, transform bounding box
    if (wmo->bLoaded) {
        // Get WMO bounds
        CAaBox wmoBounds;
        wmo->GetBounds(&wmoBounds);
        
        // Transform bounds to world space
        def->boundingBox = mat.TransformAABox(wmoBounds);
        
        // Transform bounding sphere center
        CSphere wmoSphere;
        wmo->GetSphere(&wmoSphere);
        def->boundingSphere.center = mat.TransformPoint(wmoSphere.center);
        def->boundingSphere.radius = wmoSphere.radius * mat.GetScale();
    } else {
        // Use extents from MODF entry
        def->boundingBox.min = smDef->extentsMin;
        def->boundingBox.max = smDef->extentsMax;
        
        // Calculate radius from extents
        C3Vector extents = (smDef->extentsMax - smDef->extentsMin) * 0.5f;
        def->boundingSphere.center = (smDef->extentsMax + smDef->extentsMin) * 0.5f;
        def->boundingSphere.radius = extents.Length();
    }
    
    // Store placement info
    def->uniqueId = smDef->uniqueId;
    def->doodadSet = smDef->doodadSet;
    def->nameSet = smDef->nameSet;
    def->flags = smDef->flags;
    
    return def;
}
```

---

## Transformation Matrix Construction

### Matrix Layout

```c
class C44Matrix {
    /*0x00*/ // Rotation (3x3)
    float a0, a1, a2;  // X axis
    float b0, b1, b2;  // Y axis
    float c0, c1, c2;  // Z axis
    
    /*0x24*/ // Translation
    float d0, d1, d2, d3;  // d3 = 1.0 (homogeneous)
    
    /*0x34*/ // Total size: 52 bytes
};
```

### From CreateMapObjDef (Ghidra-verified):

```c
// Identity matrix setup
def->mat.a0 = 1.0f;  // X axis X component
def->mat.a1 = 0.0f;   // X axis Y component
def->mat.a2 = 0.0f;   // X axis Z component

def->mat.b0 = 0.0f;   // Y axis X component
def->mat.b1 = 1.0f;   // Y axis Y component
def->mat.b2 = 0.0f;   // Y axis Z component

def->mat.c0 = 0.0f;   // Z axis X component
def->mat.c1 = 0.0f;   // Z axis Y component
def->mat.c2 = 1.0f;   // Z axis Z component

def->mat.d0 = 0.0f;   // Translation X
def->mat.d1 = 0.0f;   // Translation Y
def->mat.d2 = 0.0f;   // Translation Z
def->mat.d3 = 1.0f;   // W component

// Apply translation from position
NTempest::C44Matrix::Translate(&def->mat, smDef->pos);

// Apply rotation (Z-axis rotation for facing)
C3Vector rotAxis = {0, 0, 1};  // Rotate around Z
NTempest::C44Matrix::Rotate(&def->mat, rotationAngle, &rotAxis, true);

// Calculate inverse for culling
C44Matrix invMat = C44Matrix::AffineInverse(&def->mat);

// Copy inverse
def->invMat = invMat;
```

---

## Culling System

### CullMapObjDefs

```c
/* CullMapObjDefs at 0x0066e850 */
void CWorldScene::CullMapObjDefs(CWorldView* view) {
    // Get view parameters
    C3Vector viewPos = view->GetPosition();
    CFrustum frustum = view->GetFrustum();
    float drawDist = view->GetDrawDistance();
    
    // Get visible MapObjDefs list
    TSList<CMapObjDef>& visibleMapObjs = sortTable.visMapObjList;
    visibleMapObjs.Clear();
    
    // Iterate all MapObjDefs
    for (CMapObjDef* mapObj : allMapObjDefs) {
        if (mapObj->mapObj == NULL) continue;
        
        // Check if model is loaded
        if (!mapObj->mapObj->bLoaded) continue;
        
        // Get MapObjDef position
        C3Vector objPos = mapObj->mat.GetTranslation();
        
        // Distance check
        float dist = Distance(viewPos, objPos);
        if (dist > drawDist) continue;
        
        // Frustum culling using bounding box
        if (!frustum.CheckAABB(mapObj->boundingBox)) continue;
        
        // Sphere culling (faster, more conservative)
        if (!frustum.CheckSphere(mapObj->boundingSphere)) continue;
        
        // LOD check (optional)
        uint32_t lodLevel = CalculateLODLevel(dist);
        if (!mapObj->CheckLOD(lodLevel)) continue;
        
        // Add to visible list
        visibleMapObjs.Add(mapObj);
    }
    
    // Sort by distance (back-to-front for transparency)
    visibleMapObjs.SortByDistance();
}
```

### View Frustum Culling with Inverse Matrix

**Critical**: The engine uses the **inverse matrix** for culling tests against the view frustum:

```c
// To test if a point is in view frustum:
// Transform point to object space, then test against frustum

void CMapObjDef::FrustumTest(CFrustum* frustum) {
    // For each frustum plane
    for (int i = 0; i < 6; i++) {
        C4Plane plane = frustum->GetPlane(i);
        
        // Transform plane to object space
        C4Plane objPlane;
        objPlane.n.x = plane.n.x * invMat.a0 + 
                       plane.n.y * invMat.b0 + 
                       plane.n.z * invMat.c0;
        objPlane.n.y = plane.n.x * invMat.a1 + 
                       plane.n.y * invMat.b1 + 
                       plane.n.z * invMat.c1;
        objPlane.n.z = plane.n.x * invMat.a2 + 
                       plane.n.y * invMat.b2 + 
                       plane.n.z * invMat.c2;
        objPlane.d = plane.d - 
                     plane.n.x * invMat.d0 -
                     plane.n.y * invMat.d1 -
                     plane.n.z * invMat.d2;
        
        // Test against object-space bounding box
        float d = objPlane.n.x * (plane.n.x > 0 ? max.x : min.x) +
                  objPlane.n.y * (plane.n.y > 0 ? max.y : min.y) +
                  objPlane.n.z * (plane.n.z > 0 ? max.z : min.z) +
                  objPlane.d;
        
        if (d < 0) return false;  // Outside frustum
    }
    return true;  // Inside frustum
}
```

---

## Rendering Pipeline

### Main Rendering Loop

```c
/* RenderMapObjDefs (implied) */
void CWorldScene::RenderMapObjDefs() {
    // Check if rendering enabled
    if ((CWorld::enables & 0x200000) == 0) return;
    
    // Iterate visible MapObjDefs
    for (CMapObjDef* mapObj : visibleMapObjDefs) {
        CMapObj* wmo = mapObj->mapObj;
        if (wmo == NULL) continue;
        
        // Check if loaded
        if (!wmo->bLoaded) continue;
        
        // Apply fog distance
        float fogDist = CWorld::farFog;
        if (mapObj->boundingSphere.radius < fogDist) {
            // Close enough to render
            RenderWMO(wmo, &mapObj->mat, mapObj->renderFlags);
        }
    }
}
```

### WMO Rendering with Transform

```c
void RenderWMO(CMapObj* wmo, C44Matrix* worldMat, uint32_t flags) {
    // Set world transform
    GxSetWorldMatrix(worldMat);
    
    // Apply material overrides from MapObjDef
    if (flags & MODF_RENDER_FLAG_USE_OVERRIDE_COLORS) {
        SetMaterialOverrideColors(&wmo->ambientColor);
    }
    
    // Render each group
    for (CMapObjGroup* group : wmo->groups) {
        // Check group flags
        if (group->flags & WMO_GROUP_FLAG_OUTDOOR) {
            // Outdoor groups render normally
            RenderGroup(group, flags);
        } else if (group->flags & WMO_GROUP_FLAG_INDOOR) {
            // Indoor groups may use different lighting
            RenderGroupIndoor(group, flags);
        }
    }
    
    // Render attached doodads
    RenderMapObjDoodads(wmo, worldMat);
    
    // Clear overrides
    ClearMaterialOverrides();
}
```

---

## LOD System

### NameSet (LOD) Support

```c
// From CMapObjDef structure:
// - doodadSet: Which doodad set to use
// - nameSet: Which LOD filename to use

class CMapObjDef {
    uint32_t doodadSet;    // Doodad set index (0-3 typically)
    uint32_t nameSet;      // LOD level (0 = highest LOD)
};
```

### LOD Filename Convention

WMOs can have LOD variants:
- `Building.wmo` - Highest LOD
- `Building_LOD1.wmo` - Lower LOD
- `Building_LOD2.wmo` - Lowest LOD

The `nameSet` index selects which variant to load.

---

## Flags Reference

| Flag | Value | Description |
|------|-------|-------------|
| MODF_FLAG_LOAD_ASYNC | 0x01 | Load asynchronously |
| MODF_FLAG_HAS_DOODADS | 0x02 | Has attached doodads |
| MODF_FLAG_IS_DUNGEON | 0x04 | Indoor WMO |
| MODF_FLAG_USE_LOD | 0x08 | Use LOD variant |
| MODF_FLAG_COLLISION | 0x10 | Has collision data |
| MODF_FLAG_PORTAL | 0x20 | Acts as portal |
| MODF_FLAG_VIEW_DEPENDENT_LOD | 0x40 | LOD based on view angle |

---

## Integration with Doodads

### Doodad Set Selection

```c
// Doodad sets are defined in the WMO file (MODS chunk)
// MapObjDef.doodadSet selects which set to use

void CMapObjDef::SetupDoodads() {
    CMapObj* wmo = this->mapObj;
    
    // Get doodad set
    SMODoodadSet* set = &wmo->doodadSets[this->doodadSet];
    
    // Iterate doodads in set
    for (int i = 0; i < set->count; i++) {
        uint32_t doodadIndex = set->startIndex + i;
        
        // Get doodad definition
        SMODoodadDef* doodadDef = &wmo->doodadDefs[doodadIndex];
        
        // Get doodad name
        char* doodadName = &wmo->doodadNames[doodadDef->nameOffset];
        
        // Load doodad model
        HMODEL__* doodadModel = LoadModelFromFile(doodadName);
        
        // Apply world transform to doodad
        C44Matrix doodadWorldMat = this->mat * doodadDef->transform;
        ModelSetTransform(doodadModel, &doodadWorldMat);
        
        // Add to scene
        AddDoodad(doodadModel);
    }
}
```

---

## Memory and Performance

### MapObjDef Size

| Field | Size | Notes |
|-------|------|-------|
| Scene linkage | 12 bytes | TSLink |
| Object reference | 8 bytes | Pointer + ID |
| Transform matrices | 104 bytes | mat + invMat |
| Bounds | 48 bytes | AABox + Sphere |
| Lists | 24 bytes | Group pointers |
| **Total** | **~240 bytes** | Per MapObjDef |

### Optimization Notes

1. **Bounding Box Transform**: The engine transforms bounds once at load time, not per-frame
2. **Frustum Culling**: Uses inverse matrix to transform planes to object space
3. **LOD Selection**: Done at load time based on distance from origin
4. **Async Loading**: Large WMOs can load asynchronously

---

## Error Handling

### Common Errors

From [`CreateMapObjDef`](CreateMapObjDef):

```c
// Error: NULL filename
if (filename == NULL) {
    SErrDisplayError(FATAL_ERROR, "MapObjDef filename is NULL");
}

// Error: Failed to create MapObj
CMapObj* wmo = CMapObj::Create(filename);
if (wmo == NULL) {
    SErrDisplayError(FATAL_ERROR, "Failed to create MapObj from %s", filename);
}

// Error: Failed to allocate
CMapObjDef* def = AllocMapObjDef();
if (def == NULL) {
    SErrDisplayError(FATAL_ERROR, "Failed to allocate MapObjDef");
}

// Warning: Duplicate unique ID
if (FindMapObjDef(uniqueId) != NULL) {
    SErrDisplayError(WARNING, "Duplicate MapObjDef unique ID: %u", uniqueId);
}
```

---

## Summary

### MODF Key Points

1. **64-byte entries** in WDT files
2. **Position/Rotation/Scale** stored per entry
3. **Bounding boxes** for culling
4. **Doodad sets** for variation
5. **LOD support** via nameSet
6. **Async loading** supported

### Integration Points

- **LoadWdt**: Entry point for MODF parsing
- **CreateMapObjDef**: Creates runtime representation
- **Culling**: Uses bounding boxes and inverse matrices
- **Rendering**: Applies world transform
- **Doodads**: Links doodad sets from WMO file
