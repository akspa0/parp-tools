# MDX Format Specification

**Version**: MDX (Alpha 0.5.3.3368)  
**Sources**: Ghidra reverse engineering + existing parser code (WoWFormatParser)  
**Date**: 2025-12-28

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Chunk Reference](#3-chunk-reference)
4. [Geometry Data](#4-geometry-data)
5. [Animation System](#5-animation-system)
6. [Materials & Textures](#6-materials--textures)
7. [Loading Pipeline](#7-loading-pipeline)
8. [MDX vs M2 Differences](#8-mdx-vs-m2-differences)
9. [Conversion Guide](#9-conversion-guide)

---

## 1. Overview

MDX is a **Warcraft III-derived model format** used in WoW Alpha 0.5.3. It shares significant similarities with WC3's MDX but has WoW-specific extensions.

### Key Characteristics

- **Magic**: `MDLX` (4 bytes)
- **Format**: Chunked IFF-style
- **Origin**: Warcraft III model format
- **Replaced by**: M2 format in later WoW versions

### Ghidra Loading Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `BuildModelFromMdxData` | 0x00421fb0 | Main MDX loader |
| `MdxLoadGlobalProperties` | 0x0044e260 | Load MODL chunk |
| `MdxReadTextures` | 0x0044e310 | Load TEXS chunk |
| `MdxReadMaterials` | 0x0044e550 | Load MTLS chunk |
| `MdxReadGeosets` | 0x0044eba0 | Load GEOS chunk |
| `MdxReadAttachments` | 0x0044fc40 | Load ATCH chunk |
| `MdxReadAnimation` | 0x004221b0 | Load SEQS/GLBS |
| `MdxReadLights` | 0x0044a6a0 | Load LITE chunk |
| `MdxReadCameras` | 0x00449e90 | Load CAMS chunk |
| `MdxReadRibbonEmitters` | 0x0044b510 | Load RIBB chunk |
| `MdxReadEmitters2` | 0x00448f60 | Load PRE2 chunk |

---

## 2. File Structure

### Top-Level Layout

```
MDLX (magic - 4 bytes)
VERS (4 bytes) - Version number
MODL - Model info (name, bounds, flags)
SEQS - Animation sequences
GLBS - Global sequences
TEXS - Textures
MTLS - Materials
GEOS - Geosets (geometry)
GEOA - Geoset animations
BONE - Bones/skeleton
HELP - Helper objects
PIVT - Pivot points
ATCH - Attachments
CAMS - Cameras
EVTS - Events
HTST - Hit test shapes
CLID - Collision data
PRE2 - Particle emitters v2
RIBB - Ribbon emitters
LITE - Lights
TXAN - Texture animations
```

### Chunk Header

```c
struct MDXChunk {
    char   fourCC[4];   // Chunk identifier
    uint32 size;        // Data size (excludes header)
};
```

---

## 3. Chunk Reference

### VERS - Version

```c
struct VERS {
    uint32 version;  // Model version (e.g., 800, 1000, 1500)
};
```

### MODL - Model Info

```c
struct MODL {
    char     name[80];          // Model name (null-padded)
    char     animationFile[260]; // Animation file path
    CExtent  bounds;            // Bounding box + radius
    uint32   blendTime;         // Animation blend time
    uint8    flags;             // Model_Flags
};

struct CExtent {
    float    boundsRadius;
    C3Vector min;
    C3Vector max;
};

enum Model_Flags : uint8 {
    TrackYawOnly     = 0x00,
    TrackPitchYaw    = 0x01,
    TrackPitchYawRoll = 0x02,
    AlwaysAnimate    = 0x04
};
```

### SEQS - Sequences (Animations)

```c
struct SEQS_Header {
    uint32 count;     // Number of sequences
};

struct SEQS {
    char   name[80];      // Sequence name
    uint32 intervalStart; // Start time (ms)
    uint32 intervalEnd;   // End time (ms)
    float  moveSpeed;     // Movement speed
    uint32 flags;         // Sequence flags
    float  rarity;        // Random selection weight
    uint32 syncPoint;     // Sync point
    CExtent bounds;       // Sequence bounds
};
```

### GLBS - Global Sequences

```c
struct GLBS {
    uint32 durations[];  // Duration for each global sequence
};
```

### TEXS - Textures

Each texture entry is 268 bytes:

```c
struct TEXS {
    uint32 replaceableId;     // Replaceable texture ID
    char   filename[260];     // Texture path (null-padded)
    uint32 flags;             // Texture flags
};

enum TextureFlags {
    WrapWidth  = 0x01,
    WrapHeight = 0x02
};
```

### MTLS - Materials

```c
struct MTLS_Header {
    uint32 count;     // Number of materials
    uint32 unused;
};

struct MTLS {
    uint32 priorityPlane;     // Render priority
    uint32 flags;             // Material flags
    // Followed by LAYS sub-chunks
};

struct LAYS {
    uint32 filterMode;        // Blend mode
    uint32 shadingFlags;      // Shading options
    uint32 textureId;         // Index into TEXS
    uint32 textureAnimId;     // Index into TXAN (-1 = none)
    uint32 coordId;           // UV set
    float  alpha;             // Layer alpha
    // Optional animated alpha track
};

enum FilterMode {
    None = 0,
    Transparent = 1,
    Blend = 2,
    Additive = 3,
    AddAlpha = 4,
    Modulate = 5,
    Modulate2x = 6
};
```

---

## 4. Geometry Data

### GEOS - Geosets

```c
struct GEOS_Header {
    uint32 count;  // Number of geosets
};

struct GEOS {
    // Sub-chunks:
    VRTX vrtx;     // Vertices
    NRMS nrms;     // Normals
    PTYP ptyp;     // Primitive types
    PCNT pcnt;     // Primitive counts
    PVTX pvtx;     // Primitive vertices (indices)
    GNDX gndx;     // Vertex group indices
    MTGC mtgc;     // Matrix group counts
    MATS mats;     // Matrix indices
    UVAS uvas;     // UV set count
    UVBS uvbs;     // UV coordinates
    
    // Header data
    uint32 materialId;
    uint32 selectionGroup;
    uint32 selectionFlags;
    CExtent bounds;
    uint32 nAnims;
};
```

### VRTX - Vertices

```c
struct VRTX {
    uint32   count;
    C3Vector vertices[];  // count × 12 bytes
};
```

### NRMS - Normals

```c
struct NRMS {
    uint32   count;
    C3Vector normals[];   // count × 12 bytes
};
```

### PTYP - Primitive Types

```c
struct PTYP {
    uint32 count;
    uint32 types[];  // 4 = triangles
};
```

### PCNT - Primitive Counts

```c
struct PCNT {
    uint32 count;
    uint32 counts[];  // Number of indices per primitive group
};
```

### PVTX - Primitive Vertices (Indices)

```c
struct PVTX {
    uint32 count;
    uint16 indices[];  // Triangle indices
};
```

### UVAS - UV Animation Sets

```c
struct UVAS {
    uint32 count;  // Number of UV sets
};
```

### UVBS - UV Coordinates

```c
struct UVBS {
    uint32   count;
    C2Vector uvs[];  // count × 8 bytes (u, v)
};
```

### GNDX - Vertex Groups

```c
struct GNDX {
    uint32 count;
    uint8  groups[];  // Vertex to bone group mapping
};
```

### MTGC - Matrix Group Counts

```c
struct MTGC {
    uint32 count;
    uint32 counts[];  // Matrices per group
};
```

### MATS - Matrix Indices

```c
struct MATS {
    uint32 count;
    uint32 indices[];  // Bone matrix indices
};
```

---

## 5. Animation System

### BONE - Bones

```c
struct BONE_Header {
    uint32 count;
};

struct BONE {
    // Node base
    char   name[80];
    uint32 objectId;
    uint32 parentId;      // -1 = root
    uint32 flags;
    
    // Animation tracks
    KGTR translation;     // Translation keyframes
    KGRT rotation;        // Rotation keyframes
    KGSC scaling;         // Scale keyframes
    
    // Bone-specific
    uint32 geosetId;
    uint32 geosetAnimId;
};
```

### Animation Tracks

```c
struct MDXTrack<T> {
    uint32 count;         // Number of keyframes
    uint32 interpolationType; // 0=none, 1=linear, 2=hermite, 3=bezier
    uint32 globalSeqId;   // Global sequence (-1 = none)
    
    struct Keyframe {
        uint32 time;
        T      value;
        T      inTan;     // Hermite/Bezier only
        T      outTan;    // Hermite/Bezier only
    } keyframes[];
};

// Track types:
// KGTR - C3Vector translation
// KGRT - C4Quaternion rotation
// KGSC - C3Vector scale
// KATV - float alpha
// KGAO - float alpha (geoset animation)
```

### ATCH - Attachments

```c
struct ATCH_Header {
    uint32 count;
    uint32 unused;
};

struct ATCH {
    // Node base (same as BONE)
    char   name[80];
    uint32 objectId;
    uint32 parentId;
    uint32 flags;
    KGTR   translation;
    KGRT   rotation;
    KGSC   scaling;
    
    // Attachment-specific
    char   path[260];     // Attachment model path
    uint32 attachmentId;  // Attachment point ID
    KATV   visibility;    // Visibility track
};
```

### GEOA - Geoset Animations

```c
struct GEOA_Header {
    uint32 count;
};

struct GEOA {
    float  alpha;         // Static alpha
    uint32 flags;
    CArgb  color;         // Static color
    uint32 geosetId;      // Target geoset
    KGAO   alphaTrack;    // Animated alpha
    KGAC   colorTrack;    // Animated color
};
```

---

## 6. Materials & Textures

### Texture Loading (from Ghidra @ 0x0044e310)

```c
void MdxReadTextures(uchar *data, uint size, uint flags, CModelComplex *model) {
    // TEXS chunk contains 268-byte entries
    // Each entry: replaceableId(4) + filename(260) + flags(4)
    
    for each texture {
        if (replaceableId != 0) {
            // Use replaceable texture system
            texture = GetReplaceableTexture(replaceableId);
        } else {
            // Load from file
            texture = CreateBlpTexture(filename, flags);
        }
    }
}
```

### Replaceable Texture IDs

```c
enum ReplaceableTextureId {
    None           = 0,
    TeamColor      = 1,   // Player color
    TeamGlow       = 2,   // Player glow
    BarberShop     = 11,  // Character customization
    Skin           = 12,  // Character skin
    SkinExtra      = 13,  // Additional skin
    Hair           = 14,  // Character hair
};
```

---

## 7. Loading Pipeline

### Full Loading Flow (from Ghidra @ 0x00421fb0)

```c
void BuildModelFromMdxData(uchar *data, uint size, CModelComplex *model, ...) {
    // 1. Load model properties
    MdxLoadGlobalProperties(data, size, &flags, shared);  // MODL
    
    // 2. Check model type
    if ((model->field_0x8 & 0x20) == 0) {
        BuildSimpleModelFromMdxData(...);  // Simple model path
        return;
    }
    
    // 3. Load textures and materials
    MdxReadTextures(data, size, flags, model, status);    // TEXS
    MdxReadMaterials(data, size, flags, model, shared);   // MTLS
    
    // 4. Load geometry
    MdxReadGeosets(data, size, flags, model, shared);     // GEOS
    
    // 5. Load attachments
    MdxReadAttachments(data, size, flags, model, shared, status); // ATCH
    
    // 6. Load animations (unless flag 0x100)
    if ((flags & 0x100) == 0) {
        MdxReadAnimation(data, size, model, flags);       // SEQS/GLBS
        MdxReadRibbonEmitters(data, size, model, shared); // RIBB
    }
    
    // 7. Load effects
    MdxReadEmitters2(data, size, flags, model, shared, status); // PRE2
    
    // 8. Load additional data
    MdxReadNumMatrices(data, size, flags, shared);
    
    // 9. Load hit test (if flag 0x20)
    if (flags & 0x20) {
        MdxReadHitTestData(data, size, model, shared);    // HTST
    }
    
    // 10. Load lights (unless flag 0x200)
    if ((flags & 0x200) == 0) {
        MdxReadLights(data, size, model);                 // LITE
    }
    
    // 11. Load collision and bounds
    CollisionDataCreate(data, size);                      // CLID
    MdxReadExtents(data, size, model, shared);
    MdxReadPositions(data, size, flags, shared);          // PIVT
    MdxReadCameras(data, size, &model->m_cameras);        // CAMS
}
```

---

## 8. MDX vs M2 Differences

### Fundamental Architecture

| Aspect | MDX | M2 |
|--------|-----|-----|
| Origin | Warcraft III | WoW-specific |
| Chunk style | Named chunks (MODL, GEOS) | Anonymous offsets |
| Animation | Track-based with keyframes | Sequence-based |
| Bones | Embedded in BONE chunk | Separate bone/sequence arrays |
| Materials | MTLS with LAYS sub-chunks | M2Material array |
| Geometry | GEOS with sub-chunks | Vertex/index views |

### Key Structural Differences

```
MDX Structure:
MDLX → VERS → MODL → SEQS → TEXS → MTLS → GEOS → BONE → ...
[Named chunks, variable order]

M2 Structure:
MD20 header → arrays at fixed offsets
[Fixed header with offset/count pairs pointing to data]
```

### Animation System Comparison

**MDX Animation Tracks:**
```c
// MDX: Per-bone tracks with keyframes
struct KGTR {  // Translation track
    uint32 count;
    uint32 interpType;
    uint32 globalSeqId;
    Keyframe<C3Vector> keys[];
};
```

**M2 Animation:**
```c
// M2: Global animation sequences + timeline arrays
struct M2Sequence {
    uint16 id;
    uint16 variationIndex;
    uint32 duration;
    float  moveSpeed;
    uint32 flags;
    // ...
};
// Actual transforms stored separately in .anim files (later versions)
```

### Material System Comparison

**MDX Materials:**
```c
struct MTLS {
    uint32 priorityPlane;
    uint32 flags;
    LAYS layers[];  // Each layer has texture, blend mode
};
```

**M2 Materials:**
```c
struct M2Material {
    uint16 flags;
    uint16 blendingMode;  // 0-7 blend modes
};
// Textures referenced separately via texture units
```

---

## 9. Conversion Guide

### MDX → M2 Conversion

1. **Header**: Create M2 header with offset/count pairs
2. **Name**: Copy from MODL
3. **Bounding**: Copy CExtent from MODL
4. **Sequences**: Convert SEQS to M2Sequence array
5. **Global Sequences**: Copy GLBS to M2 global sequences
6. **Bones**: Convert BONE hierarchy:
   - Extract translation/rotation/scale tracks
   - Convert keyframe times to M2 timeline format
7. **Textures**: Convert TEXS entries to M2Texture array
8. **Materials**: Convert MTLS/LAYS to M2Material + texture units
9. **Geometry**: Convert GEOS to M2 vertices/indices:
   - Merge all geosets or keep as submeshes
   - Convert PVTX indices to M2 index buffer
   - Build M2SkinSection for each geoset
10. **Attachments**: Convert ATCH to M2Attachment array
11. **Particles**: Convert PRE2 to M2Particle array
12. **Ribbons**: Convert RIBB to M2Ribbon array

### M2 → MDX Conversion

1. **Magic**: Write "MDLX"
2. **VERS**: Write version (e.g., 800)
3. **MODL**: Extract name, bounds from M2 header
4. **SEQS**: Convert M2Sequence array to SEQS chunks
5. **GLBS**: Copy global sequences
6. **TEXS**: Convert M2Texture to TEXS (268 bytes each)
7. **MTLS**: Build material hierarchy from M2Material + texture units
8. **GEOS**: Convert M2SkinSection to GEOS:
   - Write VRTX, NRMS, UVBS from M2 vertex data
   - Write PVTX from M2 index buffer
   - Generate PTYP, PCNT, GNDX, MTGC, MATS
9. **BONE**: Convert M2Bone array:
   - Build KGTR/KGRT/KGSC tracks from M2 animation data
10. **ATCH**: Convert M2Attachment array
11. **PRE2**: Convert M2Particle to PRE2
12. **RIBB**: Convert M2Ribbon to RIBB

### Conversion Challenges

| Challenge | Solution |
|-----------|----------|
| Keyframe format | MDX: absolute time, M2: timeline indices → interpolate |
| Bone hierarchy | Same concept, different storage |
| UV sets | MDX: explicit UVAS count, M2: implicit in vertex format |
| Blend modes | Direct mapping (0-6 same) |
| Particle systems | Similar but different param order |

### Code Reference

Existing parsers in codebase:
- `WoWFormatParser/Structures/MDX/MDX.cs` - Full MDX parser
- `WoWFormatParser/Structures/MDX/GEOS.cs` - Geoset parsing
- `WoWFormatParser/Structures/MDX/BONE.cs` - Bone parsing
- `lib/wow.tools.local/WoWFormatLib/Structs/MDX.struct.cs` - Structures

---

## Quick Reference

### Chunk FourCCs

| Chunk | Description | Size/Entry |
|-------|-------------|------------|
| MDLX | Magic | 4 bytes |
| VERS | Version | 4 bytes |
| MODL | Model info | ~356 bytes |
| SEQS | Sequences | Variable |
| GLBS | Global seqs | 4 bytes each |
| TEXS | Textures | 268 bytes each |
| MTLS | Materials | Variable |
| GEOS | Geosets | Variable |
| BONE | Bones | Variable |
| ATCH | Attachments | Variable |
| PIVT | Pivots | 12 bytes each |
| CAMS | Cameras | Variable |
| EVTS | Events | Variable |
| HTST | Hit tests | Variable |
| CLID | Collision | Variable |
| GEOA | Geoset anims | Variable |
| PRE2 | Particles v2 | Variable |
| RIBB | Ribbons | Variable |
| LITE | Lights | Variable |
| TXAN | Tex animations | Variable |

### Ghidra Function Summary

| Function | Address | Chunk |
|----------|---------|-------|
| MdxLoadGlobalProperties | 0x0044e260 | MODL |
| MdxReadTextures | 0x0044e310 | TEXS |
| MdxReadMaterials | 0x0044e550 | MTLS |
| MdxReadGeosets | 0x0044eba0 | GEOS |
| MdxReadAttachments | 0x0044fc40 | ATCH |
| MdxReadAnimation | 0x004221b0 | SEQS/GLBS |
| MdxReadRibbonEmitters | 0x0044b510 | RIBB |
| MdxReadEmitters2 | 0x00448f60 | PRE2 |
| MdxReadLights | 0x0044a6a0 | LITE |
| MdxReadCameras | 0x00449e90 | CAMS |
| MdxReadExtents | 0x004227f0 | bounds |
| MdxReadPositions | 0x00422a50 | PIVT |
| MdxReadNumMatrices | 0x00422100 | matrices |
| MdxReadHitTestData | 0x00422230 | HTST |

---

*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) with Wowae.pdb symbols and existing parser code in WoWFormatParser.*
