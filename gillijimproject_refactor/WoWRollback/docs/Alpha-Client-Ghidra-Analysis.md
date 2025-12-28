# Alpha 0.5.3 Client Reverse Engineering Analysis

**Source**: WoWClient.exe (0.5.3.3368) with Wowae.pdb symbols  
**Tool**: Ghidra 11.3.2 + GhidraMCP  
**Date**: 2025-12-28

---

## 1. WDT Loading (`CMap::LoadWdt`)

**Address**: `0x0067fde0`

### Key Finding: Sequential Chunk Reading

The client reads WDT chunks **SEQUENTIALLY** - it does NOT seek to MPHD offsets!

```c
// Decompiled flow:
SFile::Read(wdtFile, &iffChunk, 8, ...);  // MVER header
assert(iffChunk.token == 0x4D564552);      // 'MVER'
SFile::Read(wdtFile, &version, 4, ...);    // MVER data

SFile::Read(wdtFile, &iffChunk, 8, ...);  // MPHD header  
assert(iffChunk.token == 0x4D504844);      // 'MPHD'
SFile::Read(wdtFile, &header, 0x80, ...); // MPHD data (128 bytes)

SFile::Read(wdtFile, &iffChunk, 8, ...);  // MAIN header
assert(iffChunk.token == 0x4D41494E);      // 'MAIN'
SFile::Read(wdtFile, &areaInfo, 0x10000, ...); // MAIN data (65536 bytes)

LoadDoodadNames();  // Reads MDNM sequentially
LoadMapObjNames();  // Reads MONM sequentially

SFile::Read(wdtFile, &iffChunk, 8, ...);  // Check for MODF
if (iffChunk.token == 0x4D4F4446) {        // 'MODF'
    // Process top-level WMO...
}
```

### Implications for Writers

- **NO padding bytes** between chunks - client reads sequentially
- MPHD offsets (`offsDoodadNames`, `offsMapObjNames`) are **metadata only**, not used for seeking
- FourCC tokens stored reversed on disk (e.g., 'MVER' → 'REVM')
- Chunk order MUST be: MVER → MPHD → MAIN → MDNM → MONM → [MODF optional] → [Tile Data]

---

## 2. MDNM/MONM Loading

**Addresses**: 
- `LoadDoodadNames`: `0x00680040`
- `LoadMapObjNames`: `0x006801a0`

### Key Finding: Name Counting

Both functions:
1. Read 8-byte chunk header
2. Assert token matches expected value
3. Read chunk data based on header size
4. Parse null-terminated strings, building index array

```c
// From LoadMapObjNames decompilation:
SFile::Read(wdtFile, &iffChunk, 8, ...);
assert(iffChunk.token == 0x4D4F4E4D);  // 'MONM'

if (iffChunk.size != 0) {
    SFile::Read(wdtFile, mapObjNames.data, iffChunk.size, ...);
    // Parse strings, count includes trailing empty after final null
}
```

### Implications

- `nMapObjNames` in MPHD = actual_names + 1 (for trailing null split)
- Empty chunks still need at least 1 byte (trailing null)

---

## 3. MDX Model Loading

**Address**: `BuildModelFromMdxData` @ `0x00421fb0`

### Key Finding: MDX Chunk Structure

```c
void BuildModelFromMdxData(uchar *data, uint size, CModelComplex *model, ...) {
    MdxLoadGlobalProperties(data, size, &flags, shared);
    
    if ((model->field_0x8 & 0x20) == 0) {
        BuildSimpleModelFromMdxData(...);  // Simple model path
        return;
    }
    
    // Complex model path:
    MdxReadTextures(data, size, flags, model, status);
    MdxReadMaterials(data, size, flags, model, shared);
    MdxReadGeosets(data, size, flags, model, shared);
    MdxReadAttachments(data, size, flags, model, shared, status);
    
    if ((flags & 0x100) == 0) {
        MdxReadAnimation(data, size, model, flags);
        MdxReadRibbonEmitters(data, size, model, shared);
    }
    
    MdxReadEmitters2(data, size, flags, model, shared, status);
    MdxReadNumMatrices(data, size, flags, shared);
    MdxReadHitTestData(data, size, model, shared);  // if flags & 0x20
    MdxReadLights(data, size, model);  // if flags & 0x200 == 0
    CollisionDataCreate(data, size);
    MdxReadExtents(data, size, model, shared);
    MdxReadPositions(data, size, flags, shared);
    MdxReadCameras(data, size, &model->m_cameras);
}
```

### MDX Chunk Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `MdxLoadGlobalProperties` | `0x0044e260` | Read MODL chunk (seeks 0x4C444F4D) |
| `MdxReadTextures` | `0x0044e310` | TEXS chunk |
| `MdxReadMaterials` | `0x0044e550` | MTLS chunk |
| `MdxReadGeosets` | `0x0044eba0` | GEOS chunk |
| `MdxReadAttachments` | `0x0044fc40` | ATCH chunk |
| `MdxReadAnimation` | `0x004221b0` | SEQS/GLBS chunks |
| `MdxReadRibbonEmitters` | `0x0044b510` | RIBB chunk |
| `MdxReadEmitters2` | `0x00448f60` | PRE2 chunk |
| `MdxReadLights` | `0x0044a6a0` | LITE chunk |
| `MdxReadCameras` | `0x00449e90` | CAMS chunk |

---

## 4. BLP Texture Loading

**Address**: `CreateBlpTexture` @ `0x004717f0`

### Key Finding: Async Loading

```c
void CreateBlpTexture(char *filename, int flags) {
    SFile::Open(filename, &file);
    
    CTexture *texture = SMemAlloc(0x14c, "HTEXTURE", ...);
    CTexture::CTexture(texture);
    
    SStrCopy(texture->filename, filename, 0x104);
    
    CAsyncObject *async = AsyncFileReadCreateObject();
    texture->asyncObject = async;
    async->callback = AsyncCreateBlpTextureCallback;
    async->file = file;
    async->size = SFile::GetFileSize(file, NULL);
    
    // Check if fits in texture buffer (1MB limit)
    if (async->size <= 0x100000 - currentBufferUsage) {
        // Load into shared buffer
        AsyncFileReadObject(async);
    } else {
        // Queue for later loading
    }
    
    HandleCreate(texture, "HTEXTURE");
}
```

### Key Points

- BLP textures loaded asynchronously
- 1MB buffer limit for simultaneous texture loads
- Uses `AsyncCreateBlpTextureCallback` for completion

---

## 5. Minimap Loading

**Addresses**:
- `MinimapInitialize` @ `0x006c01d0`
- `MinimapTextureCallback` @ `0x0052acc0`
- `LoadMD5Names` @ `0x006c0490`

### Key Finding: MD5 Translation File

The client uses `Textures/Minimap/md5translate.txt` to map MD5 hashes to texture files!

```c
void LoadMD5Names(void) {
    SStrPrintf(md5file, 0x104, "%s\\md5translate.txt", "Textures\\Minimap");
    
    if (SFile::LoadFile(md5file, &buffer, NULL, 1, NULL)) {
        // Parse lines: "md5hash\tfilename"
        while (SStrTokenize(&cursor, line, 0x104, "\n", NULL)) {
            if (line starts with "dir\\") continue;
            
            char *tab = SStrChr(line, '\t');
            if (tab) {
                *tab = '\0';
                // line = MD5 hash, tab+1 = filename
                
                ulong hash = SStrHashHT(line);
                // Store mapping: hash -> filename
            }
        }
        SFile::Unload(buffer);
    }
}
```

### Minimap Mystery Solved!

The **two copies** of minimaps exist because:
1. `Textures/Minimap/` - Contains the actual textures referenced by md5translate.txt
2. `World/Minimaps/` - May be a legacy/backup location or used differently

The client ONLY uses `Textures/Minimap/md5translate.txt` to find minimap textures!

---

## 6. Liquid/Water Handling

**Addresses**:
- `AllocChunkLiquid` @ `0x00691860`
- `AddChunkLiquid` @ `0x0066b120`
- `RenderLiquid_0` @ `0x0069e4b0`

### Key Finding: CChunkLiquid Structure

```c
struct CChunkLiquid {
    // From CMapChunk::CMapChunk decompilation:
    // this->liquids[0..3] = 4 liquid slots per chunk
    
    SWFlowv flowvs[2];     // Flow vectors
    float height_low;
    float height_high;
    TSLink sceneLink;
    TSLink lameAssLink;    // Yes, that's the actual name from PDB!
    // ... vertex data
};
```

### Liquid Types

From `AddChunkLiquid`:
```c
assert(type < 4);  // LQ_LAST = 4 liquid types
```

Liquid types (0-3):
- 0: Water
- 1: Ocean
- 2: Magma/Lava
- 3: Slime

---

## 7. WMO/CMapObj Loading

**Addresses**:
- `CMapObj` @ `0x00693190`
- `CreateMapObjDef` @ `0x00680f50`
- `AllocMapObj` @ `0x0068fc40`

### Key Finding: WMO Definition Structure

```c
CMapObjDef *CreateMapObjDef(char *fileName, C3Vector *pos, float rotation, int waitLoad) {
    assert(fileName != NULL);
    
    CMapObjDef *def = AllocMapObjDef();
    TSHashTable::InternalLinkNode(&mapObjDefHash, def, uniqueId);
    
    def->uniqueId = uniqueId--;
    def->mapObj = CMapObj::Create(fileName);
    
    if (waitLoad && !def->mapObj->bLoaded) {
        CMapObj::WaitLoad(def->mapObj);
    }
    
    // Set up transformation matrix
    def->mat = identity;
    C44Matrix::Translate(&def->mat, pos);
    C44Matrix::Rotate(&def->mat, rotation, &Z_AXIS, true);
    C44Matrix::AffineInverse(&def->mat, &def->invMat);
    
    // Calculate bounds if loaded
    if (def->mapObj->bLoaded) {
        CMapObj::GetBounds(def->mapObj, &def->sphere);
        CWorldMath::TransformAABox(&def->mat, &bounds, &def->aaBox);
    }
    
    return def;
}
```

---

## Summary for Implementers

| Format | Key Insight |
|--------|-------------|
| **WDT** | Sequential chunk reading, NO padding, MPHD offsets unused for seeking |
| **MDX** | Uses chunk seeking with FourCC (e.g., 0x4C444F4D = 'MODL'), WC3-like format |
| **BLP** | Async loading, 1MB buffer limit, uses SFile API |
| **Minimap** | Uses md5translate.txt for hash→filename mapping |
| **Liquid** | 4 types (water/ocean/lava/slime), 4 slots per chunk |
| **WMO** | Single monolithic file (v14), hash-based lookup, async load support |

---

---

## 8. WMO v14 Loading

**Addresses**:
- `CMapObj::CMapObj` @ `0x00693190`
- `CMapObj::WaitLoad` @ `0x00694970`
- `CMapObjGroup::CMapObjGroup` @ `0x0068b610`

### Key Finding: WMO Version 14 Confirmed

```c
void CMapObj::WaitLoad(CMapObj *this) {
    if (this->fileHeader.version == 0xE) {  // Version 14!
        if (this->asyncObject != NULL) goto wait;
        error_line = 0x2d6;
    } else {
        OsOutputDebugString("CMapObj::WaitLoad - %s wrong version", this->name);
        // ...
    }
    // Wait for async load to complete
    while (this->asyncObject != NULL) {
        AsyncFileReadWait(this->asyncObject);
    }
}
```

### CMapObjGroup Structure (WMO Group)

```c
struct CMapObjGroup {
    CAaBox aaBox;           // Bounding box
    CAaBsp aaBsp;           // BSP tree for collision
    C2iVector liquidVerts;  // Liquid vertex dimensions
    C2iVector liquidTiles;  // Liquid tile dimensions  
    C3Vector liquidCorner;  // Liquid origin corner
    TSLink lameAssLink;     // Yes, from PDB!
};
```

---

## 9. Render Distance / Far Clip System

**Addresses**:
- `CWorld::SetFarClip` @ `0x00665150`
- `CWorldParam::FarClipCallback` @ `0x00671c20`

### Key Finding: Far Clip Range

```c
bool FarClipCallback(CVar *cvar, char *oldVal, char *newVal, void *) {
    float val = SStrToFloat(newVal);
    
    // Range check: 177.0 to 777.0 yards!
    if (val >= 177.0f && val <= 777.0f) {
        CWorld::SetFarClip(val);
        return true;
    }
    
    ConsoleWrite("FarClip must be in range 177.0 to 777.0", DEFAULT_COLOR);
    return false;
}
```

**Render Distance**: **177 to 777 yards** (game units)

### Chunk AOI Calculation

```c
void CWorld::SetFarClip(float distance) {
    if (farClip != distance) {
        farClip = distance;
        
        // Calculate chunk area of interest size
        int aoiSize = 1 - (int)distance;  // Negative = more chunks
        chunkAoiSize.x = aoiSize;
        chunkAoiSize.y = aoiSize;
        
        // Reserve vertex/index buffers
        int vertCount = (aoiSize * aoiSize * 4 / 2) * 0x91;  // 145 verts per chunk
        int indexCount = (aoiSize * aoiSize * 4 / 2) * 0x300; // 768 indices per chunk
        GxBufReserve(GxBWF_Low, GxVBF_PN, vertCount, indexCount);
    }
}
```

---

## 10. WDL Heightmap Loading (Distant Terrain)

**Address**: `CMap::LoadWdl` @ `0x0067fa20`

### Key Finding: WDL Format

```c
void CMap::LoadWdl(void) {
    SStrPrintf(filename, 0x100, "%s\\%s.wdl", mapPath, mapName);
    SFile::Open(filename, &wdlFile);
    
    if (wdlFile != NULL) {
        // Read MVER
        SFile::Read(wdlFile, &iffChunk, 8, ...);
        assert(iffChunk.token == 0x4D564552);  // 'MVER'
        SFile::Read(wdlFile, &version, 4, ...);
        assert(version == 0x12);  // Version 18
        
        // Read MAOF (area low offsets)
        SFile::Read(wdlFile, &iffChunk, 8, ...);
        assert(iffChunk.token == 0x4D414F46);  // 'MAOF'
        SFile::Read(wdlFile, &areaLowOffsets, 0x4000, ...);  // 64x64 grid
        
        // For each tile with data
        for (y = 0; y < 64; y++) {
            for (x = 0; x < 64; x++) {
                if (areaLowOffsets[index] != 0) {
                    SFile::SetFilePointer(wdlFile, areaLowOffsets[index], ...);
                    
                    // Read MARE (area low entry)
                    SFile::Read(wdlFile, &iffChunk, 8, ...);
                    assert(iffChunk.token == 0x4D415245);  // 'MARE'
                    
                    // Allocate CMapAreaLow
                    CMapAreaLow *areaLow = SMemAlloc(0x8c8, ...);
                    
                    // Read 545 height values (17x17 outer + 16x16 inner)
                    SFile::Read(wdlFile, heights, 0x442, ...);  // 1090 bytes = 545 shorts
                    
                    // Calculate bounding box and sphere
                    // ...
                }
            }
        }
        SFile::Close(wdlFile);
    }
}
```

### WDL Chunk Summary

| Chunk | Size | Purpose |
|-------|------|---------|
| MVER | 4 | Version = 18 (0x12) |
| MAOF | 0x4000 (16384) | 64x64 offset table (4096 uint32s) |
| MARE | 0x442 (1090) | 545 int16 height values per tile |

### Height Grid Layout

- **545 heights** = 17×17 outer grid + 16×16 inner grid
- Stored as **int16** (signed short)
- Used for distant terrain rendering when outside FarClip range

---

## 11. Terrain Culling System

**Addresses**:
- `CWorldScene::CullChunks` @ `0x0066d3f0`
- `CWorldScene::FrustumCull` @ `0x0066c980`
- `CMap::RenderAreaLow` @ `0x0069f360`

### Culling Flow

```c
void CWorldScene::CullChunks(CSortEntry *sortEntry) {
    float cullDistance = CWorld::farClip - 533.33f;  // One ADT width
    
    for each chunk in sortEntry->chunkList {
        if (chunk->areaIndex != -1) {
            // 1. Frustum cull (is it in view?)
            if (!FrustumCull(&chunk->aaBox)) {
                // 2. Clip buffer cull (is it within render distance?)
                if (!ClipBufferCull(&chunk->aaBox, 0)) {
                    nChunksRendered++;
                    
                    // Add to visible or update list
                    if (chunk->needsUpdate && chunk->distance < cullDistance) {
                        CMapChunk::UpdateClipBuffer(chunk);
                    }
                }
            }
        }
    }
}

int FrustumCull(C3Vector *center, float radius) {
    CAaSphere sphere = { center->x, center->y, center->z, radius };
    WorldCullStatus status = CWFrustum::Cull(&frustumStack[frustumIndex], &sphere);
    return (status == WorldCull_outside);  // 1 = culled, 0 = visible
}
```

### Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| ADT Width | 533.33 yards | One terrain tile width |
| Min FarClip | 177.0 yards | Minimum render distance (~1/3 ADT) |
| Max FarClip | 777.0 yards | Maximum render distance (~1.5 ADTs) |
| Heights per WDL tile | 545 | 17×17 + 16×16 height samples |

---

## 12. Fog System

**Addresses**:
- `ComputeFogBlend` @ `0x00689b40`
- `LoadLightsAndFog` @ `0x006c4110`
- `SetFogColors` @ `0x006bc780`

### Fog Blend Calculation

```c
float ComputeFogBlend(FogData *fog, float distance) {
    assert(distance < fog->fogEnd);
    
    if (distance < fog->fogStart) {
        return 1.0f;  // No fog
    }
    
    // Linear interpolation between fogStart and fogEnd
    return 1.0f - (distance - fog->fogStart) / (fog->fogEnd - fog->fogStart);
}
```

### FogData Structure

```c
struct FogData {
    // offset 0x10: fogStart
    // offset 0x14: fogEnd
    float fogStart;  // Distance where fog begins
    float fogEnd;    // Distance where fog is 100%
};
```

---

## Summary: Render Distance System

1. **FarClip CVar**: Controls max render distance (177-777 yards)
2. **Chunk Loading**: Terrain chunks within FarClip are loaded with full detail
3. **WDL Fallback**: Terrain beyond FarClip uses WDL low-detail heightmap
4. **Fog**: Linear blend from fogStart to fogEnd, hides transition
5. **Frustum Culling**: Only render chunks in camera view frustum
6. **Clip Buffer**: Additional distance-based culling

### Terrain LOD Flow

```
Player Position
      ↓
[Within FarClip?]──Yes──→ Load Full ADT Chunks (MCNK)
      │                          ↓
      No                   Render with textures
      ↓
[Has WDL Data?]──Yes──→ Use CMapAreaLow (545 heights)
      │                          ↓
      No                   Render low-detail mesh
      ↓
   (Nothing rendered)
```

---

## Files Analyzed

- `WoWClient.exe` - Main client executable
- `Wowae.pdb` - Debug symbols (18.6 MB)

## Tool Versions

- Ghidra 11.3.2
- GhidraMCP 1.4
