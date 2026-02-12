# WoW Alpha 0.5.3 (Build 3368) MDX Format Analysis

## Overview

This document provides a deep analysis of the MDX model format implementation in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003). The analysis is based on Ghidra reverse engineering of the WoWClient.exe binary.

## Build Information

- **Version**: WoW Alpha 0.5.3 (Build 3368)
- **Build Date**: Dec 11 2003 18:01:27
- **Lua Version**: 5.0
- **MDX Version**: 1300

---

## MDX Chunk Magic Numbers

The following chunk identifiers are used in the 0.5.3 client:

| Magic | Value | Purpose |
|-------|-------|---------|
| `MDLX` | 0x584c444d | File header |
| `VERS` | 0x53524556 | Version chunk |
| `MODL` | 0x4c444f4d | Model header |
| `SEQS` | 0x53514553 | Sequences |
| `GLBS` | 0x53424c47 | Global sequences |
| `MTLS` | 0x534c544d | Materials |
| `TEXS` | 0x53455854 | Textures |
| `TXAN` | 0x4e415854 | Texture animations |
| `GEOS` | 0x534f4547 | Geosets |
| `GEOA` | 0x414f4547 | Geoset animations |
| `ATCH` | 0x48435441 | Attachments |
| `PIVT` | 0x54495649 | Pivot points |
| `BONE` | 0x454e4f42 | Bones |
| `LITE` | 0x4554494c | Lights |
| `HELP` | 0x504c4548 | Helpers |
| `EVTS` | 0x53545645 | Events |
| `PREM` | 0x4d455250 | Particle emitters |
| `PRE2` | 0x32455250 | Particle emitters 2 |
| `CAMS` | 0x534d4143 | Cameras |
| `RIBB` | 0x42424952 | Ribbon emitters |
| `CLID` | 0x44494c43 | Collision |

---

## Core Loading Functions

### BuildModelFromMdxData (0x00421fb0)

The main entry point for MDX model loading:

```c
void BuildModelFromMdxData(uchar* data, uint size, 
                           CModelComplex* modelComplex,
                           CModelShared* modelShared,
                           uint* loadFlags,
                           CStatus* status) {
    // Load global properties
    MdxLoadGlobalProperties(data, size, loadFlags, modelShared);
    
    // Check if simple model path
    if ((modelComplex->flags & 0x20) == 0) {
        BuildSimpleModelFromMdxData(data, size, (CModelSimple*)modelComplex,
                                   modelShared, loadFlags, status);
        return;
    }
    
    // Full model loading path
    MdxReadTextures(data, size, loadFlags, modelComplex, status);
    MdxReadMaterials(data, size, loadFlags, modelComplex, modelShared);
    MdxReadGeosets(data, size, loadFlags, modelComplex, modelShared);
    MdxReadAttachments(data, size, loadFlags, modelComplex, modelShared, status);
    
    if ((loadFlags & 0x100) == 0) {
        MdxReadAnimation(data, size, modelComplex, loadFlags);
        MdxReadRibbonEmitters(data, size, modelComplex, modelShared);
    }
    
    MdxReadEmitters2(data, size, loadFlags, modelComplex, modelShared, status);
    MdxReadNumMatrices(data, size, loadFlags, modelShared);
    
    if ((loadFlags & 0x20) != 0) {
        MdxReadHitTestData(data, size, modelComplex, modelShared);
    }
    
    if (loadFlags < 0) {  // Negative check
        IModelEnableFullAlpha((CModelBase*)modelComplex, 0);
    }
    
    if ((loadFlags & 0x200) == 0) {
        MdxReadLights(data, size, modelComplex);
    }
    
    // Create collision data
    HCOLLISIONDATA__* collision = CollisionDataCreate(data, size);
    modelShared->collision = collision;
    
    MdxReadExtents(data, size, modelComplex, modelShared);
    MdxReadPositions(data, size, loadFlags, modelShared);
    MdxReadCameras(data, size, &modelComplex->m_cameras);
}
```

### Load Flags

The `loadFlags` parameter controls various aspects of model loading:

| Flag | Value | Meaning |
|------|-------|---------|
| 0x20 | | Complex model (vs simple) |
| 0x100 | | Skip animation and ribbons |
| 0x200 | | Skip lights |
| Negative | | Enable full alpha |

---

## Chunk Loading Details

### MdxLoadGlobalProperties (0x0044e260)

Loads global model properties from the MODL chunk:

```c
void MdxLoadGlobalProperties(uchar* data, uint size,
                            uint* loadFlags,
                            CModelShared* modelShared) {
    // Seek to MODL chunk
    uchar* modlData = MDLFileBinarySeek(data, size, 0x4c444f4d);
    
    if (modlData == NULL) {
        // Error: MODL chunk not found
    }
    
    // Extract ground track flags
    modelShared->groundTrack = modlData[0x174] & GROUND_TRACK_MASK;
    
    // Check "always animate" flag
    if ((modlData[0x174] & 4) != 0) {
        *loadFlags = *loadFlags & 0xfffffeff;  // Clear flag
    }
}
```

**GROUND_TRACK_MASK**: 0x3
- 0x0: TRACK_YAW_ONLY
- 0x1: TRACK_PITCH_YAW
- 0x2: TRACK_PITCH_YAW_ROLL

### MdxReadMaterials (0x0044e550)

Loads material definitions from MTLS chunk:

```c
void MdxReadMaterials(uchar* data, uint size, uint loadFlags,
                     CModelComplex* modelComplex,
                     CModelShared* modelShared) {
    int* mtlData = (int*)MDLFileBinarySeek(data, size, 0x534c544d);
    
    if (mtlData != NULL) {
        int* chunkEnd = (int*)(*mtlData + (mtlData + 1));
        uint numMaterials = mtlData[1];
        
        // Ensure capacity
        TSGrowableArray<HMATERIAL__*>& materials = modelComplex->m_materials;
        if (materials.count < numMaterials) {
            TSGrowableArray::Reserve(materials, numMaterials - materials.count, 1);
        }
        
        // Load each material
        int* mtlEntry = mtlData + 3;
        for (uint i = 0; i < numMaterials; i++) {
            uint entrySize = *mtlEntry;
            uint materialId = LoadMaterialData(mtlEntry + 1, loadFlags,
                                              (char*)&modelShared->numLayers);
            materials[materialId] = materialId;
            mtlEntry = (int*)((int)mtlEntry + entrySize);
        }
    }
}
```

### MdxReadGeosets (0x0044eba0)

Loads geoset data from GEOS chunk:

```c
void MdxReadGeosets(uchar* data, uint size, uint loadFlags,
                   CModelComplex* modelComplex,
                   CModelShared* modelShared) {
    int* geosData = (int*)MDLFileBinarySeek(data, size, 0x534f4547);
    
    if (geosData != NULL) {
        int* chunkEnd = (int*)(*geosData + (geosData + 1));
        uint numGeosets = geosData[1];
        
        // Maximum 255 geosets enforced
        if (numGeosets > 0xff) {
            // Error: Too many geosets
        }
        
        modelShared->numGeosets = (uchar)numGeosets;
        
        // Reserve space
        TSGrowableArray<CGeoset>& geosets = modelComplex->m_geosets;
        TSGrowableArray::Reserve(geosets, numGeosets - geosets.count, 1);
        
        // Load each geoset
        int* geosEntry = geosData + 2;
        for (uint i = 0; i < numGeosets; i++) {
            uint entrySize = *geosEntry;
            LoadGeosetData(geosEntry + 1, entrySize - 4,
                          loadFlags, i,
                          &modelShared->geosets[i]);
            geosEntry = (int*)((int)geosEntry + entrySize);
        }
        
        // Check for GEOA (geoset animation) chunk
        int* geoaData = (int*)MDLFileBinarySeek(geosData, 
                                                data + size - (int)geosData,
                                                0x414f4547);
        if (geoaData != NULL) {
            // Load geoset animations
            uint animSize = *geoaData - 4;
            uint animCount = geoaData[1];
            
            for (uint i = 0; i < animCount; i++) {
                uint time = *(uint*)((int)geoaData + 8 + i * 12);
                // Load animation data...
            }
        }
    }
}
```

### MdxReadAttachments (0x0044fc40)

Loads attachment points from ATCH chunk:

```c
void MdxReadAttachments(uchar* data, uint size, uint loadFlags,
                       CModelComplex* modelComplex,
                       CModelShared* modelShared,
                       CStatus* status) {
    int* atchData = (int*)MDLFileBinarySeek(data, size, 0x48435441);
    
    if (atchData != NULL) {
        uchar* chunkEnd = (uchar*)atchData[1];
        uint numAttachments = atchData[2];
        uint* entryPtr = (uint*)(atchData + 3);
        
        // Allocate attachment arrays
        if (chunkEnd != modelComplex->m_attached.data) {
            TSFixedArray::ReallocData(&modelComplex->m_attached, chunkEnd);
        }
        
        // Load each attachment
        for (uint i = 0; i < numAttachments; i++) {
            uint entrySize = *entryPtr;
            LoadAttachment((int*)(entryPtr + 1), loadFlags,
                         &modelComplex->m_attached[i],
                         (int*)status);
            entryPtr = (uint*)((int)entryPtr + entrySize);
        }
    }
}
```

---

## Material System

### Material Structure

The material system uses a layered approach:

```c
struct MDLMATERIALSECTION {
    uint32_t size;
    int32_t priorityPlane;      // Sorted lowest to highest
    uint32_t numLayers;
    MDLTEXLAYER texLayers[numLayers];
};

struct MDLTEXLAYER {
    uint32_t size;
    MDLTEXOP blendMode;        // See below
    MDLGEO flags;              // See below
    uint32_t textureId;        // Index into TEXS chunk, or 0xFFFFFFFF
    uint32_t transformId;      // Index into TXAN chunk, or 0xFFFFFFFF
    int32_t coordId;           // -1 for none, or UV set index
    float staticAlpha;         // 0 = transparent, 1 = opaque
};
```

### Blend Modes (MDLTEXOP)

| Value | Name | Description |
|-------|------|-------------|
| 0x0 | TEXOP_LOAD | Opaque |
| 0x1 | TEXOP_TRANSPARENT | Transparent |
| 0x2 | TEXOP_BLEND | Alpha blend |
| 0x3 | TEXOP_ADD | Additive |
| 0x4 | TEXOP_ADD_ALPHA | Additive alpha |
| 0x5 | TEXOP_MODULATE | Modulate |
| 0x6 | TEXOP_MODULATE2X | Modulate 2x |

### Geometry Flags (MDLGEO)

| Flag | Value | Description |
|------|-------|-------------|
| MODEL_GEO_UNSHADED | 0x1 | No lighting |
| MODEL_GEO_SPHERE_ENV_MAP | 0x2 | Sphere environment map |
| MODEL_GEO_WRAPWIDTH | 0x4 | Wrap width |
| MODEL_GEO_WRAPHEIGHT | 0x8 | Wrap height |
| MODEL_GEO_TWOSIDED | 0x10 | Two-sided |
| MODEL_GEO_UNFOGGED | 0x20 | No fog |
| MODEL_GEO_NO_DEPTH_TEST | 0x40 | Disable depth test |
| MODEL_GEO_NO_DEPTH_SET | 0x80 | Disable depth write |
| MODEL_GEO_NO_FALLBACK | 0x100 | No fallback shader |

---

## Animation System

### Sequence Structure

```c
struct MDLSEQUENCESSECTION {
    char name[0x50];           // 80 bytes: sequence name
    CiRange time;              // start time, end time
    float movespeed;           // Movement speed
    uint32_t flags;            // &1 = non-looping
    CMdlBounds bounds;          // Bounding box and radius
};
```

### Animation Data Loading

The animation system uses keyframe tracks:

```c
struct MDLKEYTRACK<T> {
    uint32_t count;
    MDLTRACKTYPE type;         // Interpolation type
    uint32_t globalSeqId;      // 0xFFFFFFFF if none
    MDLKEYFRAME<T> keys[count];
};

enum MDLTRACKTYPE {
    TRACK_NO_INTERP = 0x0,
    TRACK_LINEAR = 0x1,
    TRACK_HERMITE = 0x2,
    TRACK_BEZIER = 0x3,
    NUM_TRACK_TYPES = 0x4,
};
```

**Special handling**: If `MDLMODELSECTION.flags & 4` (always animate) is set, `TRACK_LINEAR` is used regardless of stored type.

---

## Texture System

### Texture Section

```c
struct MDLTEXTURESECTION {
    REPLACEABLE_MATERIAL_IDS replaceableId;  // 0 = none
    char image[0x104];                       // Path or empty
    uint32_t flags;                          // &1 = wrap width, &2 = wrap height
};

enum REPLACEABLE_MATERIAL_IDS {
    TEX_COMPONENT_SKIN = 0x1,
    TEX_COMPONENT_OBJECT_SKIN = 0x2,
    TEX_COMPONENT_WEAPON_BLADE = 0x3,
    TEX_COMPONENT_WEAPON_HANDLE = 0x4,
    TEX_COMPONENT_ENVIRONMENT = 0x5,
    TEX_COMPONENT_CHAR_HAIR = 0x6,
    TEX_COMPONENT_CHAR_FACIAL_HAIR = 0x7,
    TEX_COMPONENT_SKIN_EXTRA = 0x8,
    TEX_COMPONENT_UI_SKIN = 0x9,
    TEX_COMPONENT_TAUREN_MANE = 0xA,
    TEX_COMPONENT_MONSTER_1 = 0xB,
    TEX_COMPONENT_MONSTER_2 = 0xC,
    TEX_COMPONENT_MONSTER_3 = 0xD,
    TEX_COMPONENT_ITEM_ICON = 0xE,
    NUM_REPLACEABLE_MATERIAL_IDS = 0xF,
};
```

---

## Doodad Integration

### Model Loading for Doodads

```c
void LoadDoodadModel(CMapDoodadDef* doodadDef) {
    // Load the MDX/M2 model file
    HMODEL__* model = LoadModelFromFile(doodadDef->filename);
    
    if (model != NULL) {
        // Apply doodad transform
        C44Matrix transform;
        BuildDoodadTransform(&transform, doodadDef->position,
                           doodadDef->rotation, doodadDef->scale);
        
        // Set model transform
        ModelSetTransform(model, &transform);
        
        // Store in doodad definition
        doodadDef->model = model;
    }
}
```

---

## Rendering Pipeline

### Model Display

```c
void ModelShowModel(HMODEL__* model, int param) {
    CModelBase* modelBase;
    
    if (model == NULL) {
        // Error
        return;
    }
    
    int result = IModelDerefHandle((CModel*)model, &modelBase);
    
    if (result == 0) {
        // Not loaded yet, queue command
        EnqueueModelCommand(model, MODEL_SHOW_MODEL, param);
        return;
    }
    
    if (param != 0) {
        // Show model (clear hidden flag)
        modelBase->m_flags = modelBase->m_flags & 0xffffffef;
    } else {
        // Hide model (set hidden flag)
        modelBase->m_flags = modelBase->m_flags | 0x10;
    }
}
```

### Animation Application

```c
void ModelAnimate(HMODEL__* model,
                  C3Vector* localPos,
                  float param3,
                  C3Vector* worldPos,
                  float param5,
                  C3Vector* camPos,
                  C3Vector* camTarg) {
    C34Matrix orientation;
    C3Vector direction;
    
    // Calculate direction from camera
    direction.x = localPos->x - camPos->x;
    direction.y = localPos->y - camPos->y;
    direction.z = localPos->z - camPos->z;
    
    // Build orientation matrix
    orientation.a0 = 1.0; orientation.a1 = 0.0; orientation.a2 = 0.0;
    orientation.b0 = 0.0; orientation.b1 = 1.0; orientation.b2 = 0.0;
    orientation.c0 = 0.0; orientation.c1 = 0.0; orientation.c2 = 1.0;
    orientation.d0 = 0.0; orientation.d1 = 0.0; orientation.d2 = 0.0;
    
    // Apply world transforms
    ApplyWorldTransforms(&direction, worldPos, param3, param5, &orientation);
    
    // Continue with full animation
    ModelAnimate(model, &orientation, param5, camPos, camTarg);
}
```

---

## Collision System

```c
HCOLLISIONDATA__* CollisionDataCreate(uchar* data, uint size) {
    // Look for CLID chunk
    int* clidData = (int*)MDLFileBinarySeek(data, size, 0x44494c43);
    
    if (clidData != NULL) {
        // Parse collision data
        return ParseCollisionData(clidData);
    }
    
    return NULL;
}
```

---

## Alternative Code Paths

### Simple Model Path

When `modelComplex->flags & 0x20 == 0`, a simplified loading path is used:

```c
void BuildSimpleModelFromMdxData(uchar* data, uint size,
                                CModelSimple* modelSimple,
                                CModelShared* modelShared,
                                uint* loadFlags,
                                CStatus* status) {
    // Simplified loading - no geosets, attachments, etc.
    MdxLoadGlobalProperties(data, size, loadFlags, modelShared);
    MdxReadTextures(data, size, loadFlags, (CModelComplex*)modelSimple, status);
    MdxReadMaterials(data, size, loadFlags, (CModelComplex*)modelSimple, modelShared);
    MdxReadPositions(data, size, loadFlags, modelShared);
}
```

### Dead Code Paths

Several code paths in the binary suggest experimental or removed features:

1. **Alternative Material Loading**: Multiple `MdxReadMaterials` functions suggest A/B testing of material systems
2. **Multiple Animation Paths**: Several `ModelAnimate` variants indicate refactoring
3. **Hidden Flags**: Many unused flag combinations in structures

---

## Binary Functions Reference

| Address | Function | Purpose |
|---------|----------|---------|
| 0x00421fb0 | BuildModelFromMdxData | Main model loader |
| 0x00422d60 | BuildSimpleModelFromMdxData | Simple model loader |
| 0x0044e260 | MdxLoadGlobalProperties | Load MODL chunk |
| 0x0044e550 | MdxReadMaterials | Load MTLS chunk |
| 0x0044eaa0 | MdxReadMaterials | Alt material loader |
| 0x0044eba0 | MdxReadGeosets | Load GEOS chunk |
| 0x0044f930 | MdxReadGeosets | Alt geoset loader |
| 0x0044e310 | MdxReadTextures | Load TEXS chunk |
| 0x0044e470 | MdxReadTextures | Alt texture loader |
| 0x0044fc40 | MdxReadAttachments | Load ATCH chunk |
| 0x004221b0 | MdxReadAnimation | Load sequence data |
| 0x0044a6a0 | MdxReadLights | Load LITE chunk |
| 0x00449e90 | MdxReadCameras | Load CAMS chunk |
| 0x0044b510 | MdxReadRibbonEmitters | Load RIBB chunk |
| 0x00448f60 | MdxReadEmitters2 | Load PRE2 chunk |
| 0x004227f0 | MdxReadExtents | Load bounds |
| 0x00422a50 | MdxReadPositions | Load vertex positions |
| 0x00422230 | MdxReadHitTestData | Load hit test data |
| 0x00422100 | MdxReadNumMatrices | Load bone matrix count |

---

## References

- **Reference**: [wowdev.wiki MDX Specification](reference_data/wowdev.wiki/MDX.md)
- **Binary**: WoWClient.exe (Build 3368, Dec 11 2003)
- **Analysis Tool**: Ghidra 11.3.2 PUBLIC
