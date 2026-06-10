# WoW Alpha 0.5.3 (Build 3368) WMO Format Analysis

## Overview

This document provides a deep analysis of the WMO (World Map Object) format implementation in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003). The analysis is based on Ghidra reverse engineering of the WoWClient.exe binary.

## Key Differences from Later Versions

The alpha version (0.5.3) uses a **single-file WMO format** rather than the split root/group format used in later versions. This is noted in the reference documentation as using a `MOMO` chunk wrapper.

---

## WMO Chunk Structure (Alpha Version)

### MOMO Chunk (Alpha-Specific)

In version 14 (alpha), all chunks are wrapped in a `MOMO` container:

```
MOMO (container)
├── MOHD (WMO header)
├── MOTX (texture names)
├── MOMT (materials)
├── MOGN (group names)
├── MOGI (group info)
├── MOPV (portal vertices)
├── MOPT (portal info)
├── MOPR (portal references)
├── MODS (doodad sets)
├── MODN (doodad names)
├── MODD (doodad definitions)
└── MFOG (fog)
```

---

## Core Data Structures

### CMapObj Class (0x00693190)

```c
class CMapObj {
    // Basic fields
    undefined4 field_0x4;
    undefined4 field_0x8;
    undefined4 field_0xc;
    undefined4 field_0x10;
    
    // Ambient color
    CArgb ambColor;
    
    // Bounding box
    CAaBox aaBox;
    
    // Scene link
    TSLink<CMapObj> lameAssLink;
    
    // File header (IFF chunk format)
    IFFChunk fileHeader;
    
    // Group list
    TSList groupList;
    
    // Group pointers
    TSFixedArray groupPtrList;  // Count: 0x180 (384 groups max)
};
```

### CMapObjGroup Class (0x0068b610)

```c
class CMapObjGroup {
    // Header info
    uint16_t flags;           // Group flags
    uint16_t nameOffset;      // Offset into MOGN
    
    // Bounding box
    CAaBox boundingBox;
    
    // Portal info
    uint16_t portalStart;     // Start vertex index in MOPV
    uint16_t portalCount;     // Number of vertices
    
    // Group data
    uint32_t groupDataOffset;  // Offset to group data
    uint32_t groupDataSize;   // Size of group data
};
```

---

## Doodad System

### CMapDoodadDef Class (0x006a4450)

```c
class CMapDoodadDef {
    // Scene linkage
    TSLink<CMapDoodadDef> sceneLink;
    
    // Model reference
    HMODEL__* model;         // Loaded MDX/M2 model
    
    // Transform
    C44Matrix mat;           // World transformation matrix
    
    // Properties
    float field_0x28;        // Scale or time-related
    float field_0x2c;
    float field_0x30;
    
    // Animation/time
    float field_0x70;        // Likely fade or scale factor
    
    // Flags
    uint32_t flags;
    
    // Callback
    void* RenderCB;          // Render callback function
    
    // Doodad set
    uint32_t doodadSet;      // Which set this belongs to
    
    // Name reference
    uint32_t nameOffset;    // Offset into doodad name table
};
```

### Doodad Rendering (0x0066d8a0)

The `CWorldScene::RenderDoodads` function:

```c
void CWorldScene::RenderDoodads() {
    // Check if doodads are enabled
    if ((CWorld::enables & 0x400081) != 0) {
        
        // Iterate through visible doodads
        CMapDoodadDef* doodad = sortTable.visDoodadList.first;
        while (doodad != NULL) {
            
            // Remove from scene list if needed
            if (doodad->sceneLink.m_prevlink != NULL) {
                // Unlink from scene
                TSLink::NextLink(&doodad->sceneLink, -1);
                doodad->sceneLink.m_prevlink = NULL;
                doodad->sceneLink.m_next = NULL;
            }
            
            // Check if model is loaded
            if (doodad->model != NULL) {
                
                // Render collision if enabled
                if (CWorld::enables & 0x80) {
                    ModelShowCollision(doodad->model, CWorld::enables & 0x80);
                }
                
                // Render bounding box if enabled
                if (CWorld::enables & 0x400000) {
                    ModelShowCollisionAaBox(doodad->model, CWorld::enables & 0x400000);
                }
                
                // Show model
                ModelShowModel(doodad->model, CWorld::enables & 1);
                
                // Advance animation time
                int animated = ModelAdvanceTime(doodad->model);
                
                if (animated != 0) {
                    // Get doodad transform
                    C44Matrix* doodadMat = &doodad->mat;
                    C44Matrix transform;
                    
                    // Copy transform
                    memcpy(&transform, doodadMat, sizeof(C44Matrix));
                    
                    // Translate relative to camera
                    transform.d0 -= camPos.x;
                    transform.d1 -= camPos.y;
                    transform.d2 -= camPos.z;
                    
                    // Build orientation matrix
                    C34Matrix local_b4;
                    NTempest::C34Matrix::C34Matrix(
                        &local_b4,
                        transform.a0, transform.a1, transform.a2,
                        transform.b0, transform.b1, transform.b2,
                        transform.c0, transform.c1, transform.c2,
                        transform.d0, transform.d1, transform.d2
                    );
                    
                    // Calculate camera direction
                    C3Vector camDir;
                    camDir.x = camTarg.x - camPos.x;
                    camDir.y = camTarg.y - camPos.y;
                    camDir.z = camTarg.z - camPos.z;
                    
                    // Animate model
                    ModelAnimate(doodad->model, &local_b4, 
                                doodad->field_0x28,
                                &camPos, &camDir);
                    
                    // Build final transform
                    C34Matrix local_84;
                    local_84.d0 = local_b4.d0 + camPos.x;
                    local_84.d1 = local_b4.d1 + camPos.y;
                    local_84.d2 = local_b4.d2 + camPos.z;
                    local_84.a0 = local_b4.a0; local_84.a1 = local_b4.a1; local_84.a2 = local_b4.a2;
                    local_84.b0 = local_b4.b0; local_84.b1 = local_b4.b1; local_84.b2 = local_b4.b2;
                    local_84.c0 = local_b4.c0; local_84.c1 = local_b4.c1; local_84.c2 = local_b4.c2;
                    
                    // Process events
                    ModelProcessEvents(doodad->model, &local_84);
                    
                    // Check fog distance
                    uint renderFlags;
                    if (doodad->field_0x70 <= CWorld::farFog) {
                        renderFlags = 0;
                    } else {
                        renderFlags = 7;  // All LOD levels
                    }
                    
                    // Add to render queue
                    ModelAddToScene(doodad->model, renderFlags);
                }
            }
            
            // Call render callback if set
            if (doodad->RenderCB != NULL) {
                ((void(*)())doodad->RenderCB)();
            }
            
            doodad = doodad->next;
        }
    }
}
```

---

## Doodad Loading

### LoadDoodadNames (0x00680040)

Loads doodad filenames from the name table:

```c
void LoadDoodadNames(uchar* data, uint size) {
    // Parse doodad name table
    char* nameTable = (char*)data;
    
    // Each name is null-terminated
    while (*nameTable != '\0') {
        // Store name reference
        doodadNames.push_back(nameTable);
        
        // Move to next name
        nameTable += strlen(nameTable) + 1;
    }
}
```

### CreateDoodadDef (0x00680300)

Creates a doodad definition:

```c
CMapDoodadDef* CreateDoodadDef(uchar* data, uint size,
                               uint nameOffset,
                               uint doodadSet) {
    CMapDoodadDef* doodad = AllocDoodadDef();
    
    // Set doodad set
    doodad->doodadSet = doodadSet;
    
    // Get name from offset
    const char* name = GetDoodadName(nameOffset);
    
    // Load model
    doodad->model = LoadModelFromFile(name);
    
    if (doodad->model != NULL) {
        // Parse transformation data
        ParseDoodadTransform(data, size, &doodad->mat);
        
        // Apply initial transform
        ModelSetTransform(doodad->model, &doodad->mat);
    }
    
    return doodad;
}
```

---

## Doodad Management

### CreateDetailDoodads (0x006a6cf0)

Creates doodads from detail data:

```c
void CreateDetailDoodads(CDetailDoodadData* detailData,
                         uint doodadSet) {
    // Allocate doodad array
    CMapDoodadDef** doodads = new CMapDoodadDef*[detailData->count];
    
    for (uint i = 0; i < detailData->count; i++) {
        // Create each doodad
        doodads[i] = CreateDoodadDef(
            (uchar*)&detailData->entries[i],
            sizeof(DoodadEntry),
            detailData->entries[i].nameOffset,
            doodadSet
        );
        
        if (doodads[i]->model != NULL) {
            // Initialize animation
            ModelInitialize(doodads[i]->model);
        }
    }
    
    // Store in global list
    AddDoodadDefs(doodads, detail->count);
}
```

---

## WMO Group System

### CMapObjGroup (0x0068b610)

```c
class CMapObjGroup {
    // Header from MOGI
    uint32_t flags;
    uint32_t nameOffset;
    
    // Bounding volume
    CAaBox boundingBox;
    
    // Portal information
    uint16_t portalStartVertex;
    uint16_t portalVertexCount;
    
    // Group data (following chunks)
    uint8_t groupData[];
    
    // Material references
    uint16_t materialIds[];
    
    // Vertices
    C3Vector vertices[];
    
    // Faces
    uint16_t indices[];
    
    // Render flags
    uint32_t renderFlags;
};
```

### Group Flags

| Flag | Value | Description |
|------|-------|-------------|
| 0x1 | | Has collision |
| 0x2 | | Indoor |
| 0x4 | | Outdoor |
| 0x8 | | Portal affected |
| 0x10 | | Has water |
| 0x20 | | Has SLOD |
| 0x40 | | Has detail geometry |

---

## Visibility and Culling

### CullDoodads (0x0066cf50)

Culls doodads based on distance and frustum:

```c
void CullDoodads(CWorldView* view) {
    // Get view parameters
    C3Vector viewPos = view->GetPosition();
    float viewDist = view->GetDrawDistance();
    
    // Get visible doodads list
    TSList<CMapDoodadDef>& visibleDoodads = sortTable.visDoodadList;
    
    // Clear visible list
    visibleDoodads.Clear();
    
    // Iterate all doodads
    for (CMapDoodadDef* doodad : allDoodads) {
        if (doodad->model == NULL) continue;
        
        // Get doodad position from transform
        C3Vector doodadPos = ExtractPosition(doodad->mat);
        
        // Check distance
        float dist = Distance(viewPos, doodadPos);
        if (dist > viewDist) continue;
        
        // Check frustum culling
        if (!view->FrustumCheck(doodad->model->GetBoundingBox())) continue;
        
        // Add to visible list
        visibleDoodads.Add(doodad);
    }
    
    // Sort by distance (back-to-front)
    visibleDoodads.SortByDistance();
}
```

---

## WMO Material System

### SMOMaterial Structure

```c
struct SMOMaterial {
    uint32_t flags;
    
    // F_UNLIT (0x1): Disable lighting
    // F_UNFOGGED (0x2): Disable fog
    // F_UNCULLED (0x4): Two-sided
    // F_EXTLIGHT (0x8): Darkened
    // F_SIDN (0x10): Self-illuminated
    // F_WINDOW (0x20): Lighting related
    // F_CLAMP_S (0x40): Clamp S coordinate
    // F_CLAMP_T (0x80): Clamp T coordinate
    
    uint32_t blendMode;
    
    // Texture indices
    uint32_t textureId1;
    uint32_t textureId2;
    uint32_t textureId3;
    
    // Colors
    CArgb emissiveColor;
    CArgb diffuseColor;
    CArgb ambientColor;
};
```

### Blend Modes

| Value | Name | Description |
|-------|------|-------------|
| 0 | Opaque | No blending |
| 1 | Transparent | Alpha blending |
| 2 | Blend | Standard blend |
| 3 | Add | Additive |
| 4 | AddAlpha | Additive alpha |
| 5 | Modulate | Multiply |

---

## Portal System

### Portal Vertices (MOPV)

```c
struct MOPVChunk {
    C3Vector vertices[];  // Portal polygon vertices
};
```

### Portal Info (MOPT)

```c
struct SMOPortal {
    uint16_t startVertex;   // Index into MOPV
    uint16_t vertexCount;  // Number of vertices
    
    // Portal plane equation
    C4Plane plane;          // a*x + b*y + c*z + d = 0
};
```

### Portal References (MOPR)

```c
struct SMOPortalRef {
    uint16_t portalIndex;   // Index into MOPT
    uint16_t groupIndex;    // Adjacent group
    int16_t side;           // Positive or negative side
    uint16_t filler;
};
```

---

## WMO Lighting

### Light Structure

```c
struct WMOLight {
    uint32_t type;          // 0=point, 1=direct, 2=ambient
    
    C3Vector position;
    C3Vector direction;
    
    float intensity;
    float radius;
    
    CArgb color;
    
    uint32_t flags;
};
```

---

## Fog System

### Fog Chunk (MFOG)

```c
struct FogData {
    float fogStart;
    float fogEnd;
    float fogDensity;
    
    CArgb fogColor;
    
    uint32_t flags;
};
```

---

## Binary Functions Reference

| Address | Function | Purpose |
|---------|----------|---------|
| 0x00693190 | CMapObj::CMapObj | Constructor |
| 0x0068b610 | CMapObjGroup | Group class |
| 0x006a4450 | CMapDoodadDef | Doodad definition |
| 0x0066d8a0 | CWorldScene::RenderDoodads | Doodad rendering |
| 0x00680040 | LoadDoodadNames | Load doodad names |
| 0x00680300 | CreateDoodadDef | Create doodad |
| 0x006a6cf0 | CreateDetailDoodads | Create detail doodads |
| 0x0066cf50 | CullDoodads | Culling |
| 0x006817c0 | CreateMapObjDefGroupDoodads | Group doodads |
| 0x00680c80 | LoadDoodadModel | Load doodad model |
| 0x0066af60 | AddMapObjDef | Add WMO to scene |
| 0x0068fc40 | AllocMapObj | Allocate WMO |
| 0x00691600 | AllocMapObjDef | Allocate WMO def |
| 0x00691700 | FreeMapObjDef | Free WMO |
| 0x0068b610 | CMapObjGroup | Group class |

---

## WMO Rendering Pipeline

```
CWorldScene::Render()
├── RenderMapObjDefs()
│   ├── UpdatePerFrame()  // Animation, transforms
│   └── Render()         // Draw groups
│
├── RenderDoodads()
│   ├── CullDoodads()    // Frustum/distance culling
│   ├── ModelAnimate()   // Apply animations
│   └── ModelAddToScene() // Queue for rendering
│
└── RenderGroups()
    ├── Check portal visibility
    └── Render visible geometry
```

---

## Integration with Doodads

### Doodad-to-WMO Linking

```c
void LinkDoodadToWMO(CMapDoodadDef* doodad, CMapObj* wmo) {
    // Add to WMO's doodad list
    TSList<CMapDoodadDef>& doodadList = wmo->GetDoodadList();
    doodadList.Add(doodad);
    
    // Set WMO reference in doodad
    doodad->parentWmo = wmo;
    
    // Transform doodad to WMO space
    C44Matrix wmoToWorld = wmo->GetWorldTransform();
    C44Matrix doodadToWmo = doodad->GetLocalTransform();
    
    // Combine transforms
    C44Matrix doodadToWorld;
    MultiplyMatrix(&doodadToWorld, &wmoToWorld, &doodadToWmo);
    
    doodad->SetWorldTransform(&doodadToWorld);
}
```

---

## Alpha-Specific Notes

### Missing Chunks (Compared to Later Versions)

The alpha version does NOT have:
- MOUV (UV animation)
- MOM3 (Material 3.0)
- MOPV (extended portal data)
- MOLP (LOD chunks)
- MOCA (batch data)

### MOMO Container

All alpha WMOs wrap everything in `MOMO`:

```c
void ParseAlphaWMO(uchar* data, uint size) {
    // Find MOMO chunk
    uint32_t* momo = FindChunk(data, 'MOMO');
    
    // Parse children
    uchar* childData = (uchar*)(momo + 2);
    uint childSize = momo[1];
    
    while (childSize > 0) {
        uint32_t* chunk = (uint32_t*)childData;
        uint32_t magic = chunk[0];
        uint32_t chunkSize = chunk[1];
        
        switch (magic) {
            case 'MOHD':
                ParseHeader(chunk);
                break;
            case 'MOTX':
                ParseTextures(chunk);
                break;
            case 'MOMT':
                ParseMaterials(chunk);
                break;
            // ... other chunks
        }
        
        childData += 8 + chunkSize;
        childSize -= 8 + chunkSize;
    }
}
```

---

## References

- **Reference**: [wowdev.wiki WMO Specification](reference_data/wowdev.wiki/WMO.md)
- **Binary**: WoWClient.exe (Build 3368, Dec 11 2003)
- **Analysis Tool**: Ghidra 11.3.2 PUBLIC
