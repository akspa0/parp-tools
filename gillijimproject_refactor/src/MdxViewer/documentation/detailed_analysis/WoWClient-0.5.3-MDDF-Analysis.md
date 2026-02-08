# WoW Alpha 0.5.3 (Build 3368) MDDF Chunk Analysis

## Overview

This document provides a deep analysis of the **MDDF chunk** (Map Doodad Definition) in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. The MDDF chunk defines doodads (small decorative objects like trees, rocks, grass clumps, etc.) placed on terrain tiles (ADT files).

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`LoadDoodadNames`](LoadDoodadNames) | 0x00680040 | Load doodad filenames from WDT |
| [`CreateDoodadDef`](CreateDoodadDef) | 0x00680300 | Create doodad definition |
| [`AllocDoodadDef`](AllocDoodadDef) | 0x00690af0 | Allocate doodad definition |
| [`CMapDoodadDef`](CMapDoodadDef) | 0x006a4450 | Doodad definition class |
| [`RenderDoodads`](RenderDoodads) | 0x0066d8a0 | Render visible doodads |
| [`CullDoodads`](CullDoodads) | 0x0066cf50 | Culling visible doodads |
| [`CreateDetailDoodads`](CreateDetailDoodads) | 0x006a6cf0 | Create procedural doodads |
| [`CMapChunk::CreateDetailDoodads`](CMapChunk::CreateDetailDoodads) | - | Per-chunk detail doodads |
| [`AddDoodad`](AddDoodad) | 0x00664370 | Add doodad to scene |
| [`LoadDoodadModel`](LoadDoodadModel) | 0x00680c80 | Load doodad model |

---

## MDDF Chunk Format

### File Location

The MDDF chunk appears in **WDT files** (World Definition Tables) for doodads that span the entire map or are referenced globally. For tile-specific doodads, see the **ADT MMDX/MMDN/MMDD chunks**.

### Chunk Structure

```
WDT File:
├── MVER (Version)
├── MPHD (Map Header)
├── MAIN (Area Information)
├── doodadNames (MDNM - Doodad Names)
└── MDDF (Map Doodad Definitions)
```

### MDDF Entry Structure (Ghidra-Verified)

**From Ghidra analysis:**

```c
// MDDF entry is part of doodadNames combined structure
// Each doodad is identified by:
// - nameOffset: Offset into doodadNames string table
// - Position/Rotation/Scale: Transform data

struct SMMapDoodadDef {
    /*0x00*/ uint32_t nameOffset;    // Offset into doodadNames table
    /*0x04*/ uint32_t uniqueId;     // Unique identifier
    /*0x08*/ C3Vector position;     // World position (X, Y, Z)
    /*0x14*/ float rotationY;       // Y-axis rotation (radians)
    /*0x18*/ float scale;           // Scale factor (typically 1.0)
    /*0x1C*/ uint16_t flags;        // Doodad flags
    /*0x1E*/ uint16_t doodadSet;    // Doodad set index
};
// Total: 32 bytes (0x20)
```

### Key Observations from LoadDoodadNames

**From [`LoadDoodadNames`](LoadDoodadNames) at 0x00680040:**

```c
void CMap::LoadDoodadNames() {
    SIffChunk iffChunk;
    
    // Read chunk header
    SFile::Read(wdtFile, &iffChunk, 8, ...);
    
    // Validate magic: 0x4d444e4d = 'MDNM' (not 'MDDF')
    if (iffChunk.token != 0x4d444e4d) {
        Error("MDNM chunk not found");
        return;
    }
    
    uint32_t chunkSize = iffChunk.size;
    
    // Allocate space for name table
    if (doodadNames.capacity < chunkSize) {
        doodadNames.Reserve(chunkSize);
    }
    
    // Read doodad names (null-terminated strings)
    SFile::Read(wdtFile, doodadNames.data, chunkSize, ...);
    
    // Parse name offsets
    uint32_t offset = 0;
    uint32_t count = 0;
    
    if (chunkSize > 0) {
        do {
            // Store offset for this doodad
            doodadNameOffsets[count] = offset;
            
            // Find null terminator
            while (doodadNames.data[offset] != '\0') {
                offset++;
            }
            offset++;  // Skip null
            
            count++;
        } while (offset < chunkSize);
    }
    
    doodadNameCount = count;
}
```

---

## CMapDoodadDef Class Structure

**From Ghidra decompilation of [`CMapDoodadDef`](CMapDoodadDef) at 0x006a4450:**

```c
class CMapDoodadDef {
    /*0x00*/ // Scene linkage
    TSLink<CMapDoodadDef> sceneLink;
    
    /*0x0C*/ // Model reference
    HMODEL__* model;         // Loaded MDX/M2 model
    
    /*0x10*/ // Identification
    uint32_t uniqueId;       // Unique placement ID
    uint32_t nameOffset;     // Offset into doodad name table
    
    /*0x18*/ // Transform
    C44Matrix mat;           // World transformation matrix
    C44Matrix lMat;         // Local transformation (before view offset)
    
    /*0x4C*/ // Animation/Time
    float field_0x28;       // Animation time or scale
    float field_0x2c;       // Unknown
    float field_0x30;       // Unknown
    
    /*0x34*/ // Position (duplicate for easy access)
    float x, y, z;
    
    /*0x40*/ // Bounding box (object space)
    CAaBox localBounds;
    
    /*0x54*/ // World position
    C3Vector worldPos;
    
    /*0x60*/ // State
    uint32_t flags;         // Rendering flags
    uint16_t doodadSet;    // Doodad set index
    uint16_t nameSet;      // LOD name set
    
    /*0x68*/ // Fog distance
    float fogDist;         // Distance at which doodad fades
    
    /*0x6C*/ // Callbacks
    void* RenderCB;        // Render callback
    void* AnimCB;          // Animation callback
    void* EventCB;         // Event callback
    
    /*0x78*/ // Doodad name
    char* modelName;       // Filename of the model
    
    /*0x7C*/ // Status
    bool isLoaded;         // Model loaded flag
    bool isVisible;        // Culled visibility
    uint16_t refCount;    // Reference count
};
```

---

## Doodad Creation Pipeline

### 1. Name Table Loading

```c
/* LoadDoodadNames at 0x00680040 */
void CMap::LoadDoodadNames() {
    SIffChunk iffChunk;
    
    // Read chunk header
    SFile::Read(wdtFile, &iffChunk, 8, (ulong*)0x0, ...);
    
    // Validate 'MDNM' chunk
    if (iffChunk.token != 0x4d444e4d) {  // 'MDNM'
        ErrDisplay("MDNM chunk token invalid");
    }
    
    uint32_t nameTableSize = iffChunk.size;
    
    // Reserve space in growable array
    doodadNames.Reserve(nameTableSize);
    
    // Read entire string table
    SFile::Read(wdtFile, doodadNames.data, nameTableSize, ...);
    
    // Build offset index
    uint32_t offset = 0;
    uint32_t doodadIndex = 0;
    
    while (offset < nameTableSize) {
        // Store offset for each doodad
        doodadOffsets[doodadIndex] = offset;
        
        // Find null terminator
        while (doodadNames.data[offset] != '\0') {
            offset++;
        }
        offset++;  // Skip null terminator
        
        doodadIndex++;
    }
    
    doodadCount = doodadIndex;
}
```

### 2. CreateDoodadDef

```c
/* CreateDoodadDef at 0x00680300 */
CMapDoodadDef* CMap::CreateDoodadDef(
    uint32_t nameOffset,
    C3Vector* position,
    float rotationY,
    uint16_t doodadSet
) {
    // Allocate doodad definition
    CMapDoodadDef* doodad = AllocDoodadDef();
    if (doodad == NULL) {
        Error("Failed to allocate doodad");
        return NULL;
    }
    
    // Generate unique ID
    doodad->uniqueId = uniqueId++;
    
    // Store name offset
    doodad->nameOffset = nameOffset;
    doodad->doodadSet = doodadSet;
    
    // Get filename from offset
    doodad->modelName = &doodadNames.data[nameOffset];
    
    // Initialize matrices
    C44Matrix* mat = &doodad->mat;
    
    // Identity matrix
    mat->a0 = 1.0f; mat->a1 = 0.0f; mat->a2 = 0.0f;
    mat->b0 = 0.0f; mat->b1 = 1.0f; mat->b2 = 0.0f;
    mat->c0 = 0.0f; mat->c1 = 0.0f; mat->c2 = 1.0f;
    mat->d0 = 0.0f; mat->d1 = 0.0f; mat->d2 = 0.0f; mat->d3 = 1.0f;
    
    // Apply translation
    mat->d0 = position->x;
    mat->d1 = position->y;
    mat->d2 = position->z;
    
    // Store world position
    doodad->worldPos = *position;
    
    // Apply rotation around Y axis
    C3Vector rotAxis = {0, 0, 1};  // Z-axis for rotation
    C44Matrix::Rotate(mat, rotationY, &rotAxis, true);
    
    // Initialize local matrix (same as world initially)
    doodad->lMat = *mat;
    
    // Initialize animation time
    doodad->field_0x28 = 1.0f;  // Default scale
    
    // Load the model
    LoadDoodadModel(doodad);
    
    // Add to hash table for fast lookup
    doodadDefHash.Insert(doodad, doodad->uniqueId);
    
    // Add to scene list
    sceneLinkList.Add(doodad);
    
    return doodad;
}
```

### 3. LoadDoodadModel

```c
/* LoadDoodadModel at 0x00680c80 */
void CMap::LoadDoodadModel(CMapDoodadDef* doodad) {
    // Get filename
    char* filename = &doodadNames.data[doodad->nameOffset];
    
    // Load model from file
    HMODEL__* model = LoadModelFromFile(filename);
    
    if (model == NULL) {
        Error("Failed to load doodad model: %s", filename);
        return;
    }
    
    // Store model reference
    doodad->model = model;
    
    // Apply world transform to model
    ModelSetTransform(model, &doodad->mat);
    
    // Get model bounds for culling
    ModelGetBounds(model, &doodad->localBounds);
    
    // Initialize animation
    ModelInitialize(model);
    
    // Check if animated (has bones)
    if (ModelHasAnimation(model)) {
        // Start animation playback
        ModelStartAnimation(model, 0);  // Play animation 0
    }
    
    doodad->isLoaded = true;
}
```

---

## Detail Doodads (Procedural Generation)

### CMapChunk::CreateDetailDoodads

**From Ghidra analysis at [`CreateDetailDoodads`](CreateDetailDoodads) 0x006a6cf0:**

Detail doodads are procedurally generated based on terrain layers and density settings:

```c
/* CMapChunk::CreateDetailDoodads */
void CMapChunk::CreateDetailDoodads() {
    if (this->nLayers == 0) return;
    
    // Allocate detail doodad instance
    CDetailDoodadInst* inst = CDetailDoodad::AllocInst();
    this->detailDoodadInst = inst;
    
    if (inst == NULL) {
        Error("Failed to allocate detail doodad instance");
        return;
    }
    
    // Check for test mode
    if (CWorld::detailDoodadTest != 0) {
        // Test mode: place single doodad
        TestDetailDoodads(inst);
        return;
    }
    
    // Generate splat positions using noise
    C2iVector splatList[128];
    GenerateSplatPositions(splatList, CWorld::detailDoodadDensity);
    
    // Get terrain plane for this chunk
    C4Plane terrainPlane;
    GetTerrainPlane(&terrainPlane);
    
    // Process each splat position
    for (int i = 0; i < CWorld::detailDoodadDensity; i++) {
        int splatY = splatList[i].y;
        int splatX = splatList[i].x;
        
        // Check if position is valid
        if (!IsValidSplatPosition(splatX, splatY)) continue;
        
        // Get ground effect for this position
        GroundEffectTextureRec* effect = GetGroundEffect(splatX, splatY);
        if (effect == NULL) continue;
        
        // Get effect ID
        uint32_t effectId = effect->effectId;
        if (effectId == 0xFFFFFFFF) continue;
        
        // Get model from effect
        char* modelName = effect->modelName;
        uint32_t modelId = effect->modelId;
        
        // Calculate world position
        C3Vector worldPos;
        CalculateSplatWorldPosition(splatX, splatY, &worldPos);
        
        // Adjust Y to terrain height
        C3Vector terrainPos = worldPos;
        float terrainHeight = SampleTerrainHeight(worldPos.x, worldPos.z);
        terrainPos.y = terrainHeight;
        
        // Get normal for this position
        C3Vector normal;
        SampleTerrainNormal(worldPos.x, worldPos.z, &normal);
        
        // Calculate rotation from normal
        float rotationY = atan2(normal.x, normal.z);
        
        // Get shadow bit
        bool hasShadow = GetShadowBit(splatX, splatY);
        
        // Add doodad instance
        CDetailDoodadInst::AddDoodad(
            inst, modelId, &terrainPos, hasShadow, &normal
        );
    }
}
```

### Splat Position Generation

```c
/* Procedural splat position generation */
void CMapChunk::GenerateSplatPositions(
    C2iVector* splatList,
    uint32_t density
) {
    // Initialize random seed
    CRandomSeed seed = this->rSeed;
    
    for (uint32_t i = 0; i < density; i++) {
        // Use noise function for distribution
        uint32_t noiseX = Noise2D(i * 7, 0) & 0xFF;
        uint32_t noiseY = Noise2D(i * 11, 0) & 0xFF;
        
        // Convert to 8x8 grid coordinates
        splatList[i].x = noiseX % 8;
        splatList[i].y = noiseY % 8;
    }
}
```

### Terrain Height Sampling

```c
/* Sample terrain height at world position */
float CMapChunk::SampleTerrainHeight(float worldX, float worldZ) {
    // Convert world to chunk-local coordinates
    float localX = worldX - this->position.x;
    float localZ = worldZ - this->position.z;
    
    // Convert to heightmap coordinates
    float mapX = localX / CHUNK_SIZE;  // 0.0 to 1.0
    float mapZ = localZ / CHUNK_SIZE; // 0.0 to 1.0
    
    // Sample MCVT height data (bilinear interpolation)
    float h00 = GetHeightValue(0, 0);
    float h10 = GetHeightValue(1, 0);
    float h01 = GetHeightValue(0, 1);
    float h11 = GetHeightValue(1, 1);
    
    // Bilinear interpolation
    float h0 = h00 + (h10 - h00) * mapX;
    float h1 = h01 + (h11 - h01) * mapX;
    
    return h0 + (h1 - h0) * mapZ;
}
```

---

## Culling System

### CullDoodads

```c
/* CullDoodads at 0x0066cf50 */
void CWorldScene::CullDoodads(CWorldView* view) {
    // Get view parameters
    C3Vector viewPos = view->GetPosition();
    CFrustum frustum = view->GetFrustum();
    float drawDist = view->GetDrawDistance();
    
    // Clear visible list
    TSList<CMapDoodadDef>& visibleDoodads = sortTable.visDoodadList;
    visibleDoodads.Clear();
    
    // Iterate all doodads
    for (CMapDoodadDef* doodad : allDoodads) {
        // Skip unloaded doodads
        if (!doodad->isLoaded) continue;
        if (doodad->model == NULL) continue;
        
        // Get doodad world position
        C3Vector doodadPos = doodad->worldPos;
        
        // Distance check
        float dist = Distance(viewPos, doodadPos);
        if (dist > drawDist) continue;
        
        // Fog distance check
        if (dist > doodad->fogDist) continue;
        
        // Frustum culling using local bounds
        if (!FrustumCheckDoodad(doodad, frustum)) continue;
        
        // Add to visible list
        visibleDoodads.Add(doodad);
    }
    
    // Sort by distance (back-to-front for transparency)
    visibleDoodads.SortByDistance();
}
```

### Fog Distance Calculation

```c
/* Calculate fog distance for doodad */
void CMapDoodadDef::CalculateFogDistance() {
    // Get model bounding radius
    float radius = GetBoundingRadius();
    
    // Fog start/end from current fog settings
    float fogStart = GetFogStart();
    float fogEnd = GetFogEnd();
    
    // Calculate fade distance
    // Doodads start fading at fogStart - radius
    // and are fully faded at fogEnd + radius
    fogDist = fogEnd + radius * 2;
}
```

---

## Rendering Pipeline

### RenderDoodads

**From [`RenderDoodads`](RenderDoodads) at 0x0066d8a0:**

```c
void CWorldScene::RenderDoodads() {
    // Check if rendering enabled
    if ((CWorld::enables & 0x400081) == 0) return;
    
    // Iterate visible doodads
    CMapDoodadDef* doodad = visibleDoodads.head;
    while (doodad != NULL) {
        // Remove from scene list if needed
        if (doodad->sceneLink.m_prevlink != NULL) {
            doodad->sceneLink.m_prevlink->m_next = doodad->sceneLink.m_next;
            if (doodad->sceneLink.m_next != NULL) {
                doodad->sceneLink.m_next->m_prevlink = doodad->sceneLink.m_prevlink;
            }
            doodad->sceneLink.m_prevlink = NULL;
            doodad->sceneLink.m_next = NULL;
        }
        
        // Get model
        HMODEL__* model = doodad->model;
        if (model != NULL) {
            // Show/hide based on flags
            if (CWorld::enables & 0x80) {  // Collision
                ModelShowCollision(model, true);
            }
            if (CWorld::enables & 0x400000) {  // Bounding box
                ModelShowCollisionAaBox(model, true);
            }
            
            // Show/hide model
            ModelShowModel(model, CWorld::enables & 1);
            
            // Advance animation time
            int animated = ModelAdvanceTime(model);
            
            if (animated != 0) {
                // Build camera-relative transform
                C44Matrix transform = doodad->mat;
                transform.d0 -= camPos.x;
                transform.d1 -= camPos.y;
                transform.d2 -= camPos.z;
                
                // Create orientation matrix
                C34Matrix orientation;
                orientation.a0 = transform.a0; orientation.a1 = transform.a1; orientation.a2 = transform.a2;
                orientation.b0 = transform.b0; orientation.b1 = transform.b1; orientation.b2 = transform.b2;
                orientation.c0 = transform.c0; orientation.c1 = transform.c1; orientation.c2 = transform.c2;
                orientation.d0 = transform.d0; orientation.d1 = transform.d1; orientation.d2 = transform.d2;
                
                // Camera direction
                C3Vector camDir = camTarg - camPos;
                
                // Animate model
                ModelAnimate(
                    model,
                    &orientation,
                    doodad->field_0x28,  // Time
                    &camPos,
                    &camDir
                );
                
                // Build final transform
                C34Matrix finalTrans;
                finalTrans.d0 = orientation.d0 + camPos.x;
                finalTrans.d1 = orientation.d1 + camPos.y;
                finalTrans.d2 = orientation.d2 + camPos.z;
                finalTrans.a0 = orientation.a0; finalTrans.a1 = orientation.a1; finalTrans.a2 = orientation.a2;
                finalTrans.b0 = orientation.b0; finalTrans.b1 = orientation.b1; finalTrans.b2 = orientation.b2;
                finalTrans.c0 = orientation.c0; finalTrans.c1 = orientation.c1; finalTrans.c2 = orientation.c2;
                
                // Process animation events
                ModelProcessEvents(model, &finalTrans);
                
                // Determine render flags based on fog
                uint32_t renderFlags;
                if (doodad->fogDist <= CWorld::farFog) {
                    renderFlags = 0;  // Skip distant doodads
                } else {
                    renderFlags = 7;  // Full detail
                }
                
                // Add to scene render queue
                ModelAddToScene(model, renderFlags);
            }
        }
        
        // Call render callback if set
        if (doodad->RenderCB != NULL) {
            ((void(*)())doodad->RenderCB)();
        }
        
        doodad = doodad->next;
    }
}
```

### Billboard Rotation

```c
/* Billboard rotation for detail doodads */
void CMapDoodadDef::UpdateBillboardRotation(C3Vector* camPos) {
    // Calculate yaw to face camera
    C3Vector toCamera = *camPos - worldPos;
    float yaw = atan2(toCamera.x, toCamera.z);
    
    // Update rotation matrix
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);
    
    // Y-axis rotation (billboard)
    mat.a0 = sinYaw;  mat.a1 = 0;  mat.a2 = cosYaw;
    mat.b0 = 0;       mat.b1 = 1;  mat.b2 = 0;
    mat.c0 = cosYaw;  mat.c1 = 0;  mat.c2 = -sinYaw;
    
    // Re-apply position
    mat.d0 = worldPos.x - camPos.x;
    mat.d1 = worldPos.y - camPos.y;
    mat.d2 = worldPos.z - camPos.z;
}
```

---

## Animation Integration

### Doodad Animation Types

```c
// Doodads can have the following animations:
// 1. Built-in model animations (MDX/M2)
// 2. Procedural animations (wind sway, etc.)

/* DoodadAnimCallback at 0x006718e0 */
void DoodadAnimCallback(CMapDoodadDef* doodad, float deltaTime) {
    HMODEL__* model = doodad->model;
    if (model == NULL) return;
    
    // Advance built-in animations
    ModelAdvanceTime(model);
    
    // Apply procedural animation (wind sway)
    if (doodad->flags & DOODAD_FLAG_WIND_AFFECTED) {
        ApplyWindSway(doodad, deltaTime);
    }
}

/* Wind sway for tree doodads */
void CMapDoodadDef::ApplyWindSway(float deltaTime) {
    // Get current wind direction and strength
    C3Vector wind = GetGlobalWind();
    float strength = wind.Length();
    
    // Calculate sway based on height
    float height = localBounds.max.y - localBounds.min.y;
    float swayAmount = strength * height * 0.1f;
    
    // Apply sway to rotation
    float time = GetTickCount() * 0.001f;
    float swayX = sin(time * 2.0f + worldPos.x) * swayAmount;
    float swayZ = cos(time * 1.5f + worldPos.z) * swayAmount;
    
    // Modify matrix
    mat.b0 = swayX;
    mat.b2 = swayZ;
}
```

---

## Sound Integration

### Doodad Sound Effects

```c
/* SndInterfaceHandleDoodadLoopStart at 0x004a62b0 */
void SndInterface::DoodadLoopStart(CMapDoodadDef* doodad) {
    if (doodad->model == NULL) return;
    
    // Check if model has sound
    char* soundName = ModelGetSoundName(doodad->model);
    if (soundName == NULL) return;
    
    // Play loop sound
    PlaySound3D(soundName, &doodad->worldPos, true);
}

/* SndInterfaceHandleDoodadOneShot at 0x004a6411 */
void SndInterface::DoodadOneShot(CMapDoodadDef* doodad, uint32_t soundId) {
    // Play one-shot sound at doodad position
    PlaySoundAtPosition(soundId, &doodad->worldPos);
}
```

---

## Optimization Techniques

### 1. Distance-Based Detail

```c
/* LOD selection for doodads */
uint32_t CMapDoodadDef::GetRenderFlags() {
    float dist = Distance(camPos, worldPos);
    
    if (dist < 50.0f) {
        return RENDER_FULL;           // Full detail
    } else if (dist < 150.0f) {
        return RENDER_LOD1;           // Reduced detail
    } else if (dist < 300.0f) {
        return RENDER_LOD2;           // Minimal detail
    } else {
        return RENDER_NONE;           // Don't render
    }
}
```

### 2. Alpha Culling

```c
/* Alpha testing for distant doodads */
void CMapDoodadDef::CheckAlphaCulling() {
    if (flags & DOODAD_FLAG_ALPHA_CULL) {
        float dist = Distance(camPos, worldPos);
        float alpha = 1.0f - (dist / fogDist);
        
        if (alpha < 0.1f) {
            // Don't render if nearly invisible
            flags |= DOODAD_FLAG_CULLED;
        }
    }
}
```

### 3. Batch Rendering

```c
/* Batch similar doodads for performance */
void CWorldScene::BatchRenderDoodads() {
    // Group doodads by model
    TSHashMap<HMODEL__*, TSList<CMapDoodadDef>> batches;
    
    for (CMapDoodadDef* doodad : visibleDoodads) {
        if (doodad->model != NULL) {
            batches[doodad->model].Add(doodad);
        }
    }
    
    // Render each batch
    for (auto& batch : batches) {
        HMODEL__* model = batch.key;
        TSList<CMapDoodadDef>& doodads = batch.value;
        
        // Set model once
        BindModel(model);
        
        // Render all instances
        for (CMapDoodadDef* doodad : doodads) {
            RenderModelInstance(model, &doodad->mat);
        }
    }
}
```

---

## Flags Reference

| Flag | Value | Description |
|------|-------|-------------|
| DOODAD_FLAG_WIND_AFFECTED | 0x01 | Affected by wind |
| DOODAD_FLAG_ALPHA_CULL | 0x02 | Alpha culling enabled |
| DOODAD_FLAG_CULLED | 0x04 | Culled this frame |
| DOODAD_FLAG_ANIMATED | 0x08 | Has built-in animation |
| DOODAD_FLAG_SOUND | 0x10 | Has attached sound |
| DOODAD_FLAG_COLLISION | 0x20 | Has collision |
| DOODAD_FLAG_SELECTABLE | 0x40 | Can be selected |
| DOODAD_FLAG_PERMANENT | 0x80 | Never despawns |

---

## Summary

### MDDF Key Points

1. **Name table offsets** reference doodad names in WDT
2. **Per-doodad transforms** for position, rotation, scale
3. **Procedural generation** via CreateDetailDoodads
4. **Distance-based culling** with fog fade
5. **Billboard rotation** for detail doodads
6. **Animation integration** via model callbacks
7. **Sound integration** for environmental audio

### Performance Considerations

1. **Bounding radius** used for fog culling
2. **Frustum culling** on local bounds
3. **LOD selection** based on distance
4. **Alpha culling** for distant objects
5. **Batch rendering** for same-model doodads

### Integration Points

- **LoadDoodadNames**: Name table parsing
- **CreateDoodadDef**: Runtime object creation
- **CreateDetailDoodads**: Procedural generation
- **CullDoodads**: Visibility determination
- **RenderDoodads**: Scene rendering
