# Texture Loading Workflow

## Overview

This document describes the complete workflow for loading textures from DBC references and applying them to MDX models. The process involves multiple subsystems working together to resolve texture paths, load texture data, and bind textures to the rendering pipeline.

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Model Loading Request                                        │
│    (Load creature, item, character, etc.)                       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. DBC Lookup                                                   │
│    - Query CreatureDisplayInfo/ItemDisplayInfo                  │
│    - Retrieve model path and texture overrides                  │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MDX Model Loading                                            │
│    - Load MDX file from disk/MPQ                                │
│    - Parse model chunks (TXTS, TXID, GEOS, etc.)                │
│    - Extract embedded texture references                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Texture Reference Resolution                                 │
│    - Combine DBC texture overrides with model textures          │
│    - Resolve texture file paths                                 │
│    - Handle texture variations and fallbacks                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Texture Cache Lookup                                         │
│    - Check if texture already loaded (hash table lookup)        │
│    - If found, increment reference count and return             │
└────────────┬────────────────────────────────────────────────────┘
             │ Cache Miss
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Asynchronous Texture Loading                                 │
│    - Load BLP file from MPQ archives                            │
│    - Decompress and decode BLP format                           │
│    - Generate mipmaps if needed                                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. GPU Upload                                                   │
│    - Create GPU texture object                                  │
│    - Upload texture data to GPU memory                          │
│    - Set texture filtering parameters                           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Texture Binding                                              │
│    - Store texture handle in model structure                    │
│    - Add to texture cache for future reuse                      │
│    - Register with resource manager                             │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Rendering                                                    │
│    - Bind textures to appropriate texture units                 │
│    - Render model geometry with applied textures                │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Process

### Step 1: Model Loading Request

The process begins when the game needs to display a 3D model:

```c
// Example: Load a creature model
void LoadCreature(uint32_t creatureDisplayInfoID) {
    // Lookup display info in DBC
    CreatureDisplayInfoRec* displayInfo = 
        LookupCreatureDisplayInfo(creatureDisplayInfoID);
    
    if (!displayInfo) {
        HandleError("Invalid CreatureDisplayInfo ID");
        return;
    }
    
    // Continue with model loading
    LoadCreatureModel(displayInfo);
}
```

### Step 2: DBC Lookup

Query the appropriate DBC file to get model and texture information:

```c
CreatureDisplayInfoRec* LookupCreatureDisplayInfo(uint32_t id) {
    // Use hash table for O(1) lookup
    uint32_t hash = ComputeHash(id);
    DBCACHEHASH* entry = g_CreatureDisplayInfoCache[hash];
    
    while (entry) {
        if (entry->key == id) {
            return entry->record;
        }
        entry = entry->next;
    }
    
    return NULL;
}
```

**Retrieved Information**:
- Model file path (from CreatureModelData)
- Texture override paths (Skin1, Skin2, Skin3)
- Model scale and other properties
- Particle effects and sounds

### Step 3: MDX Model Loading

Load and parse the MDX/M2 model file:

```c
// From Ghidra analysis: MdxReadTextures function (0x0044e310)
void MdxReadTextures(uint8_t* data, uint32_t offset, 
                     uint32_t flags, CModelComplex* model, 
                     CStatus* status) {
    // Seek to TXTS (textures) chunk
    uint32_t* texSection = MDLFileBinarySeek(data, offset, 'TXTS');
    
    if (!texSection) return;
    
    // Calculate number of textures
    uint32_t numTextures = texSection[0] / sizeof(CModelTexture);
    
    // Validate section size
    if (texSection[0] != numTextures * sizeof(CModelTexture)) {
        DisplayError("Invalid texture section size");
        return;
    }
    
    // Allocate texture array if needed
    if (model->m_textures.count < numTextures) {
        ResizeTextureArray(&model->m_textures, numTextures);
    }
    
    // Process each texture entry
    ProcessTextures(&texSection[1], numTextures, flags, 
                    status, model->m_textures.data);
}
```

**MDX Texture Entry Structure**:
```c
struct CModelTexture {
    uint32_t type;              // Texture type (0 = hardcoded, 1 = player skin)
    uint32_t flags;             // Texture flags
    char     filename[256];     // Texture file path (or empty)
    uint32_t replaceSkin;       // Use external skin replacement
};
```

### Step 4: Texture Reference Resolution

The [`ProcessTextures()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/03_texture_loading_workflow.md:130) function handles texture path resolution:

```c
// From Ghidra analysis: ProcessTextures function (0x0044c2e0)
void ProcessTextures(int* textureData, uint32_t numTextures,
                    CGxTexFlags flags, CStatus* status,
                    int textureArray) {
    
    for (uint32_t i = 0; i < numTextures; i++) {
        CModelTexture* texInfo = &textureData[i];
        
        // Store texture type/flags
        textureArray[i].type = texInfo->type;
        
        // Check if we should use default texture or load from file
        bool useDefault = (flags & 0x800) && (texInfo->type == 0);
        
        if (useDefault || !texInfo->filename[0]) {
            // Create solid color placeholder
            textureArray[i].handle = 
                TextureCreateSolid(0xFFFF00FF); // Magenta
        } else {
            // Determine texture filtering
            EGxTexFilter filter = GxTex_LinearMipNearest;
            uint32_t anisotropy = 1;
            
            if (flags & 0x10000) {
                // Anisotropic filtering
                filter = GxTex_Anisotropic;
                anisotropy = 1 << ((flags >> 17) & 7);
            } else if (flags & 0x1000) {
                // Trilinear filtering
                filter = GxTex_LinearMipLinear;
            }
            
            // Get additional texture flags
            CGxTexFlags texFlags;
            GetTextureFlags(texInfo, &texFlags);
            
            // Load the texture
            textureArray[i].handle = 
                LoadModelTexture(texInfo->filename, flags, 
                               texFlags, status);
        }
    }
}
```

### Step 5: Texture Cache Lookup

Before loading, check if texture is already in cache:

```c
HTEXTURE__ * TextureCreate(const char* path, CGxTexFlags flags,
                           CStatus* status, int unknown) {
    // Normalize path for consistent hashing
    char normalizedPath[260];
    NormalizePath(path, normalizedPath);
    
    // Compute hash
    HASHKEY_TEXTUREFILE hashKey(normalizedPath);
    uint32_t hash = hashKey.GetHash();
    
    // Lookup in cache
    UCTextureHash* cached = g_TextureCache.Lookup(hash, hashKey);
    
    if (cached) {
        // Cache hit - increment reference count
        cached->refCount++;
        return cached->handle;
    }
    
    // Cache miss - continue with loading
    return LoadTextureFromFile(path, flags, status);
}
```

### Step 6: Asynchronous Texture Loading

```c
// From Ghidra analysis: LoadModelTexture function (0x00447f50)
HTEXTURE__* LoadModelTexture(const char* filename, uint32_t flags,
                             CGxTexFlags texFlags, CStatus* status) {
    char path[260];
    
    // Handle development path flag
    if ((flags & 0x4000) && filename[1] != ':' && filename[0] != '\\') {
        // Prepend development path
        strcpy(path, "\\\\Guldan\\Drive2\\Projects\\WoW\\Data\\");
        strcat(path, filename);
        return TextureCreate(path, texFlags, status, 0);
    }
    
    // Normal path
    return TextureCreate(filename, texFlags, status, 0);
}
```

**Async Loading Process**:
```c
void AsyncTextureLoad(const char* path, TextureCallback callback) {
    // Create async load request
    AsyncLoadRequest* request = CreateAsyncLoad();
    request->path = path;
    request->callback = callback;
    request->priority = PRIORITY_NORMAL;
    
    // Add to load queue
    g_AsyncLoadQueue.Enqueue(request);
    
    // Worker thread will process
}

// Worker thread
void AsyncTextureWorker() {
    while (running) {
        AsyncLoadRequest* request = g_AsyncLoadQueue.Dequeue();
        
        if (request) {
            // Load BLP file
            void* blpData = LoadFileFromMPQ(request->path);
            
            // Decode BLP
            TextureData* texData = DecodeBLP(blpData);
            
            // Callback on main thread
            PostCallback(request->callback, texData);
            
            FreeAsyncLoad(request);
        }
    }
}
```

### Step 7: GPU Upload

Once texture data is loaded, upload to GPU:

```c
HTEXTURE__* CreateGPUTexture(TextureData* data, CGxTexFlags flags) {
    // Create OpenGL/DirectX texture
    HTEXTURE__* handle = AllocateTextureHandle();
    
    // Bind texture
    glBindTexture(GL_TEXTURE_2D, handle->glTexture);
    
    // Upload base mip level
    glTexImage2D(GL_TEXTURE_2D, 0, 
                 data->format, data->width, data->height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data->pixels);
    
    // Upload mipmaps
    if (data->mipLevels > 1) {
        for (uint32_t mip = 1; mip < data->mipLevels; mip++) {
            glTexImage2D(GL_TEXTURE_2D, mip,
                        data->format, 
                        data->width >> mip, 
                        data->height >> mip, 0,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        data->mipData[mip]);
        }
    }
    
    // Set filtering
    SetTextureFiltering(handle, flags);
    
    return handle;
}
```

### Step 8: Texture Binding

Store texture handle in model and add to cache:

```c
void BindTextureToModel(CModelComplex* model, uint32_t slot,
                        HTEXTURE__* texture) {
    // Store in model's texture array
    model->m_textures.data[slot] = texture;
    
    // Add to global cache
    AddToTextureCache(texture->path, texture);
    
    // Register with resource manager for cleanup
    RegisterResource(RESOURCE_TEXTURE, texture);
}
```

### Step 9: Rendering

During rendering, bind textures to appropriate texture units:

```c
void RenderModel(CModelComplex* model) {
    // For each material/geoset
    for (uint32_t i = 0; i < model->numMaterials; i++) {
        Material* mat = &model->materials[i];
        
        // Bind textures for this material
        for (uint32_t t = 0; t < mat->numTextures; t++) {
            uint32_t texSlot = mat->textureIndices[t];
            HTEXTURE__* tex = model->m_textures.data[texSlot];
            
            // Bind to texture unit
            BindTexture(t, tex);
        }
        
        // Render geometry
        RenderGeoset(model->geosets[i]);
    }
}
```

## Texture Override Priority

When multiple texture sources exist, the priority is:

1. **DBC Skin Overrides** (highest)
   - Skin1, Skin2, Skin3 from CreatureDisplayInfo
   - ItemDisplayInfo texture fields

2. **Model Embedded Textures**
   - Texture paths in MDX TXTS chunk
   - Used when no DBC override specified

3. **Default/Fallback**
   - Solid color texture (magenta for debugging)
   - Used when texture file missing or type is 0

```c
HTEXTURE__* ResolveTexture(CModelTexture* modelTex, 
                           const char* dbcOverride) {
    // Priority 1: DBC override
    if (dbcOverride && dbcOverride[0]) {
        return LoadTexture(dbcOverride);
    }
    
    // Priority 2: Model embedded path
    if (modelTex->filename[0]) {
        return LoadTexture(modelTex->filename);
    }
    
    // Priority 3: Fallback
    return CreateDefaultTexture();
}
```

## Error Handling

Each step includes error checking:

```c
// Missing DBC record
if (!displayInfo) {
    LogError("CreatureDisplayInfo %d not found", id);
    return LoadFallbackModel();
}

// Invalid model file
if (!ValidateMDX(modelData)) {
    LogError("Invalid MDX file: %s", path);
    return NULL;
}

// Missing texture file
if (!FileExists(texturePath)) {
    LogWarning("Texture not found: %s, using fallback", texturePath);
    return CreateFallbackTexture();
}

// GPU upload failure
if (!UploadToGPU(textureData)) {
    LogError("Failed to upload texture to GPU");
    FreeTextureData(textureData);
    return NULL;
}
```

## Performance Optimizations

### Batch Loading
Load multiple textures together to reduce overhead:

```c
void BatchLoadTextures(const char** paths, uint32_t count,
                      HTEXTURE__** outHandles) {
    // Sort by MPQ location for sequential reads
    SortPathsByMPQOrder(paths, count);
    
    // Load in batch
    for (uint32_t i = 0; i < count; i++) {
        outHandles[i] = LoadTexture(paths[i]);
    }
}
```

### Background Streaming
Stream textures while player is loading:

```c
void StreamTextures(CModel* model) {
    for (uint32_t i = 0; i < model->numTextures; i++) {
        if (!model->textures[i].loaded) {
            QueueAsyncLoad(model->textures[i].path, 
                          PRIORITY_BACKGROUND);
        }
    }
}
```

### LOD (Level of Detail)
Load appropriate texture resolution based on distance:

```c
HTEXTURE__* LoadLODTexture(const char* path, float distance) {
    uint32_t lodLevel = CalculateLOD(distance);
    
    if (lodLevel > 0) {
        // Load lower resolution version
        char lodPath[260];
        sprintf(lodPath, "%s_lod%d.blp", path, lodLevel);
        return LoadTexture(lodPath);
    }
    
    return LoadTexture(path);
}
```

---

**Next**: [Database Schema](04_database_schema.md)
