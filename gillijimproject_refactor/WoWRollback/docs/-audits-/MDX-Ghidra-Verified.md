# MDX Format Specification (Ghidra-Verified)

**Source**: Fresh Ghidra decompilation of WoWClient.exe (0.5.3.3368) with Wowae.pdb  
**Date**: 2025-12-28  
**Trust Level**: HIGH - All data from actual binary analysis

---

## 1. Loading Pipeline (BuildModelFromMdxData @ 0x00421fb0)

### Load Order (Ghidra-verified)

```c
void BuildModelFromMdxData(uchar *data, uint size, CModelComplex *model, 
                           CModelShared *shared, uint *flags, CStatus *status) {
    // 1. Load global properties (MODL)
    MdxLoadGlobalProperties(data, size, &flags, shared);
    
    // 2. Check model type - simple vs complex
    if ((model->field_0x8 & 0x20) == 0) {
        BuildSimpleModelFromMdxData(...);  // Simple path
        return;
    }
    
    // 3. Load textures (TEXS)
    MdxReadTextures(data, size, flags, model, status);
    
    // 4. Load materials (MTLS)
    MdxReadMaterials(data, size, flags, model, shared);
    
    // 5. Load geosets (GEOS)
    MdxReadGeosets(data, size, flags, model, shared);
    
    // 6. Load attachments (ATCH)
    MdxReadAttachments(data, size, flags, model, shared, status);
    
    // 7. Load animations (unless flag 0x100)
    if ((flags & 0x100) == 0) {
        MdxReadAnimation(data, size, model, flags);    // SEQS/GLBS
        MdxReadRibbonEmitters(data, size, model, shared); // RIBB
    }
    
    // 8. Load particle emitters (PRE2)
    MdxReadEmitters2(data, size, flags, model, shared, status);
    
    // 9. Load matrices
    MdxReadNumMatrices(data, size, flags, shared);
    
    // 10. Load hit test (if flag 0x20)
    if (flags & 0x20) {
        MdxReadHitTestData(data, size, model, shared);  // HTST
    }
    
    // 11. Enable full alpha (if flag 0x80)
    if (flags & 0x80) {
        IModelEnableFullAlpha(model, 0);
    }
    
    // 12. Load lights (unless flag 0x200)
    if ((flags & 0x200) == 0) {
        MdxReadLights(data, size, model);  // LITE
    }
    
    // 13. Load collision (CLID)
    shared->collision = CollisionDataCreate(data, size);
    
    // 14. Load extents/bounds
    MdxReadExtents(data, size, model, shared);
    
    // 15. Load pivot positions (PIVT)
    MdxReadPositions(data, size, flags, shared);
    
    // 16. Load cameras (CAMS)
    MdxReadCameras(data, size, &model->m_cameras);
}
```

---

## 2. Chunk Seeking (MDLFileBinarySeek @ 0x0078be40)

```c
uchar * MDLFileBinarySeek(uchar *fileData, uint fileSize, ulong chunkId) {
    ulong *end = (ulong *)(fileData + fileSize);
    
    while (fileData < end) {
        if (chunkId == *(ulong *)fileData) {
            return (uchar *)((int)fileData + 4);  // Return pointer after FourCC
        }
        fileData = (uchar *)((int)fileData + *(ulong *)((int)fileData + 4) + 8);
    }
    return NULL;
}
```

**Key insight**: Chunks are searched linearly, FourCC at +0, size at +4, data at +8.

---

## 3. TEXS Chunk (MdxReadTextures @ 0x0044e310)

### Texture Entry Size

```c
puVar3 = MDLFileBinarySeek(data, size, 0x53584554);  // 'TEXS'
if (puVar3 != NULL) {
    uVar6 = *puVar3 / 0x10c;  // ← 268 bytes per texture!
    if (*puVar3 != uVar6 * 0x10c) {
        SErrDisplayError("sectionBytes % numTextures == s...");
    }
    // ...
}
```

**TEXS Entry**: **268 bytes** (0x10C) each - CONFIRMED

### Texture Structure (268 bytes)

```c
struct MDXTexture {
    uint32 replaceableId;     // +0x00 (4 bytes)
    char   filename[260];     // +0x04 (260 bytes, null-padded)
    uint32 flags;             // +0x108 (4 bytes)
};  // Total: 268 bytes
```

---

## 4. GEOS Chunk (MdxReadGeosets @ 0x0044eba0)

### Geoset Container

```c
piVar4 = MDLFileBinarySeek(data, size, 0x534f4547);  // 'GEOS'
if (piVar4 != NULL) {
    piVar5 = (int *)(*piVar4 + (int)(piVar4 + 1));  // End pointer
    uVar11 = piVar4[1];  // Geoset count
    piVar4 = piVar4 + 2;  // Skip to first geoset
    
    if (0xff < uVar11) {
        SErrDisplayError("numGeosets < 0xff");  // Max 255 geosets
    }
    
    // Each geoset has variable size, stored at start
    for each geoset {
        iVar9 = *piVar10;  // Geoset size
        piVar4 = (int *)(iVar9 + (int)piVar10);  // Next geoset
        LoadGeosetData(piVar10 + 1, iVar9 - 4, ...);
    }
}
```

---

## 5. Geoset Sub-Chunks (LoadGeosetData @ 0x0044eec0)

### Sub-Chunk FourCCs (Ghidra-verified)

| Chunk | FourCC | Entry Size | Notes |
|-------|--------|------------|-------|
| VRTX | 0x58545256 | 12 bytes | `count * 3` floats |
| NRMS | 0x534D524E | 12 bytes | Same count as VRTX |
| UVAS | 0x53415655 | - | UV set count |
| UV data | - | 8 bytes | `numVertices * 2` floats |

### Geoset Data Parsing

```c
void LoadGeosetData(int *geosetData, int dataSize, uint flags, uint index, ...) {
    // VRTX - Vertices
    if (*ptr != 0x58545256) {  // 'VRTX'
        SErrDisplayError("...SMVRTX...");
    }
    numVertices = ptr[1];
    // Copy vertices: numVertices * 12 bytes (3 floats each)
    
    // NRMS - Normals  
    if (*ptr != 0x534d524e) {  // 'NRMS'
        SErrDisplayError("...SMNRMS...");
    }
    // numNormals must equal numVertices
    if (numVertices != numNormals) {
        SErrDisplayError("numVertices == numNormals");
    }
    // Copy normals: numNormals * 12 bytes
    
    // UVAS - UV set count
    if (*ptr == 0x53415655) {  // 'UVAS'
        numMappingChannels = ptr[1];
        // For each UV set: numVertices * 8 bytes (2 floats)
    }
    
    // Then: primitive data, transform groups, etc.
}
```

---

## 6. MTLS Chunk (MdxReadMaterials @ 0x0044e550)

### Material Container

```c
piVar4 = MDLFileBinarySeek(data, size, 0x534c544d);  // 'MTLS'
if (piVar4 != NULL) {
    piVar5 = (int *)(*piVar4 + (int)(piVar4 + 1));  // End
    uVar1 = piVar4[1];  // Material count
    shared->numLayers = 0;
    piVar4 = piVar4 + 3;  // Skip header
    
    // Each material has variable size
    for each material {
        iVar2 = *piVar4;  // Material size
        LoadMaterialData(piVar4 + 1, flags, &shared->numLayers);
        piVar4 = (int *)((int)piVar4 + iVar2);
    }
}
```

---

## 7. GEOA Chunk (Geoset Animations)

```c
// After GEOS, look for GEOA
piVar4 = MDLFileBinarySeek(piVar4, remaining, 0x414f4547);  // 'GEOA'
if (piVar4 != NULL) {
    // Each animation entry in GEOA
    for (iVar9 = count; iVar9 != 0; iVar9--) {
        uVar13 = *puVar12;      // Entry size
        uVar6 = puVar12[1];     // Geoset index
        float alpha = (float)puVar12[2];  // Alpha value
        // Apply to geoset color array
    }
}
```

---

## 8. Key Ghidra Addresses

| Function | Address | Chunk |
|----------|---------|-------|
| BuildModelFromMdxData | 0x00421fb0 | Entry point |
| MdxLoadGlobalProperties | 0x0044e260 | MODL |
| MdxReadTextures | 0x0044e310 | TEXS |
| MdxReadMaterials | 0x0044e550 | MTLS |
| MdxReadGeosets | 0x0044eba0 | GEOS |
| LoadGeosetData | 0x0044eec0 | VRTX/NRMS/UVAS |
| MdxReadAttachments | 0x0044fc40 | ATCH |
| MdxReadAnimation | 0x004221b0 | SEQS/GLBS |
| MdxReadRibbonEmitters | 0x0044b510 | RIBB |
| MdxReadEmitters2 | 0x00448f60 | PRE2 |
| MdxReadLights | 0x0044a6a0 | LITE |
| MdxReadCameras | 0x00449e90 | CAMS |
| MdxReadExtents | 0x004227f0 | Bounds |
| MdxReadPositions | 0x00422a50 | PIVT |
| MdxReadNumMatrices | 0x00422100 | Matrices |
| MdxReadHitTestData | 0x00422230 | HTST |
| MDLFileBinarySeek | 0x0078be40 | Chunk finding |
| CollisionDataCreate | - | CLID |

---

## 9. Verified Chunk FourCCs

| Chunk | FourCC (hex) | FourCC (str) |
|-------|--------------|--------------|
| GEOS | 0x534F4547 | 'GEOS' |
| GEOA | 0x414F4547 | 'GEOA' |
| TEXS | 0x53584554 | 'TEXS' |
| MTLS | 0x534C544D | 'MTLS' |
| VRTX | 0x58545256 | 'VRTX' |
| NRMS | 0x534D524E | 'NRMS' |
| UVAS | 0x53415655 | 'UVAS' |

---

## 10. Verified Structure Sizes

| Structure | Size | Source |
|-----------|------|--------|
| TEXS entry | **268 bytes** | `size / 0x10c` |
| Vertex (C3Vector) | 12 bytes | `count * 3` floats |
| Normal (C3Vector) | 12 bytes | Same as vertex |
| UV (C2Vector) | 8 bytes | `count * 2` floats |

---

## 11. Load Flags

| Flag | Meaning |
|------|---------|
| 0x20 | Complex model (vs simple) |
| 0x20 | Load hit test data |
| 0x80 | Enable full alpha |
| 0x100 | Skip animation loading |
| 0x200 | Skip lights loading |

---

## 12. Corrections to Old Documentation

Based on Ghidra analysis:

1. **TEXS size**: Confirmed **268 bytes** per entry (old docs said this, verified)
2. **Chunk seeking**: Linear search, not offset-based
3. **Max geosets**: 255 (0xFF) - enforced by client
4. **VRTX/NRMS counts**: Must be equal (enforced)
5. **Load order**: Specific order matters (see pipeline above)

---

*This document contains ONLY Ghidra-verified information from decompilation of WoWClient.exe (0.5.3.3368). No old community documentation was used.*
