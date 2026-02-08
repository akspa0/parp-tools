# API Reference

## Overview

This document provides a complete API reference for the DBC texturing system, based on reverse engineering analysis of the WoW client binary using Ghidra.

## Core Functions

### MDX Texture Loading

#### MdxReadTextures

**Address**: `0x0044e310`  
**Signature**: `void __fastcall MdxReadTextures(uint8_t* data, uint32_t offset, uint32_t flags, CModelComplex* model, CStatus* status)`

Reads and processes texture references from an MDX model file.

**Parameters**:
- `data`: Pointer to MDX file data in memory
- `offset`: Byte offset into the file to start reading
- `flags`: Texture loading flags (CGxTexFlags)
- `model`: Pointer to CModelComplex structure to populate
- `status`: Status object for error reporting

**Description**:
Seeks to the TXTS (textures) chunk in the MDX file and processes all texture entries. Allocates or resizes the model's texture array and calls [`ProcessTextures()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:25) to load each texture.

**Example**:
```c
CModelComplex model;
CStatus status;
MdxReadTextures(mdxData, 0, 0, &model, &status);
```

**Related Functions**:
- [`ProcessTextures()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:25)
- [`MDLFileBinarySeek()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:37)

---

#### ProcessTextures

**Address**: `0x0044c2e0`  
**Signature**: `void __fastcall ProcessTextures(int* textureData, uint32_t numTextures, CGxTexFlags flags, CStatus* status, int textureArray)`

Processes an array of texture entries from an MDX file and loads the corresponding texture files.

**Parameters**:
- `textureData`: Pointer to array of CModelTexture structures
- `numTextures`: Number of textures to process
- `flags`: Texture loading and filtering flags
- `status`: Status object for error reporting
- `textureArray`: Destination array for loaded texture handles

**Description**:
Iterates through each texture entry in the MDX file. For each texture:
1. Stores the texture type and flags
2. Determines if a default texture should be used
3. Configures texture filtering (point, bilinear, trilinear, anisotropic)
4. Calls [`LoadModelTexture()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:59) to load the texture file
5. Stores the resulting texture handle

**Texture Flags**:
- `0x800`: Force default solid texture
- `0x1000`: Enable trilinear filtering
- `0x10000`: Enable anisotropic filtering
- `0x1E0000`: Anisotropy level mask (bits 17-20)

**Example**:
```c
int textureArray[16];
ProcessTextures(mdxTextureData, 4, 0x1000, &status, textureArray);
```

---

#### LoadModelTexture

**Address**: `0x00447f50`  
**Signature**: `HTEXTURE__ * __fastcall LoadModelTexture(const char* filename, uint32_t flags, CGxTexFlags texFlags, CStatus* status)`

Loads a texture file referenced by an MDX model.

**Parameters**:
- `filename`: Path to the texture file (relative to game data directory)
- `flags`: Loading flags
- `texFlags`: Texture-specific flags (filtering, wrapping, etc.)
- `status`: Status object for error reporting

**Returns**: Handle to the loaded texture, or NULL on failure

**Description**:
Handles special path handling for development builds (prepends `\\Guldan\Drive2\Projects\WoW\Data\` if flag 0x4000 is set). Delegates to [`TextureCreate()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:97) for actual loading.

**Path Handling**:
- Checks for absolute paths (starting with drive letter or backslash)
- Applies development path prefix if appropriate
- Normalizes path separators

**Example**:
```c
HTEXTURE__* texture = LoadModelTexture(
    "Creature\\Murloc\\Murloc.blp",
    0,
    defaultTexFlags,
    &status
);
```

---

### Texture Creation and Loading

#### TextureCreate

**Signature**: `HTEXTURE__ * TextureCreate(const char* path, CGxTexFlags flags, CStatus* status, int unknown)`

Creates a new texture from a file or returns an existing cached texture.

**Parameters**:
- `path`: Full or relative path to texture file
- `flags`: Texture creation flags
- `status`: Status object for error reporting
- `unknown`: Reserved parameter

**Returns**: Handle to the texture

**Description**:
1. Normalizes the texture path
2. Computes hash of the path
3. Looks up in texture cache (CTextureHash hash table)
4. If found, increments reference count and returns cached handle
5. If not found, loads texture from file and adds to cache

**Caching**:
Uses a hash table keyed by normalized file path for O(1) lookup. Multiple references to the same texture share the same GPU resource.

**Example**:
```c
CGxTexFlags flags;
flags.filter = GxTex_LinearMipLinear;
flags.wrapU = GxTex_Clamp;
flags.wrapV = GxTex_Clamp;

HTEXTURE__* tex = TextureCreate("Interface\\Icons\\INV_Sword_01.blp", 
                                flags, &status, 0);
```

---

#### TextureCreateSolid

**Signature**: `HTEXTURE__ * TextureCreateSolid(CImVector* color, CStatus* status)`

Creates a 1x1 solid color texture.

**Parameters**:
- `color`: RGBA color value (format: 0xAABBGGRR)
- `status`: Status object for error reporting

**Returns**: Handle to the solid color texture

**Description**:
Creates a procedural 1x1 texture filled with the specified color. Used for:
- Placeholder textures when files are missing
- Debug visualization (typically magenta: 0xFFFF00FF)
- Simple colored materials

**Example**:
```c
// Create magenta placeholder
CImVector magenta = 0xFFFF00FF;
HTEXTURE__* placeholder = TextureCreateSolid(&magenta, &status);
```

---

#### AsyncTextureHandler

**Address**: `0x0046fb00`  
**Signature**: `void AsyncTextureHandler()`

Main callback handler for asynchronous texture loading.

**Description**:
Processes completed texture loads from the async loading queue. Called from the main thread to safely upload textures to GPU and update model references.

**Thread Safety**:
- Worker threads decompress and decode BLP files
- Main thread handles GPU upload and cache updates
- Uses inter-thread message queue for communication

---

#### AsyncTextureWait

**Address**: `0x00472300`  
**Signature**: `void AsyncTextureWait()`

Blocks until all pending async texture loads are complete.

**Description**:
Typically called during loading screens to ensure all required textures are loaded before gameplay resumes.

**Example**:
```c
// Load textures asynchronously
QueueAsyncTextureLoad("Texture1.blp");
QueueAsyncTextureLoad("Texture2.blp");

// Wait for completion
AsyncTextureWait();

// Now safe to render
```

---

### Texture Hash Tables

#### CTextureHash

**Description**: Primary texture cache using file paths as keys.

**Structure**:
```c
struct UCTextureHash {
    char              path[260];      // Normalized texture path
    HTEXTURE__*       handle;         // Texture handle
    uint32_t          refCount;       // Reference counter
    UCTextureHash*    next;           // Collision chain
};
```

**Operations**:
```c
// Lookup
UCTextureHash* CTextureHash::Lookup(uint32_t hash, HASHKEY_TEXTUREFILE& key);

// Insert
void CTextureHash::Insert(UCTextureHash* entry);

// Remove
void CTextureHash::Remove(uint32_t hash, HASHKEY_TEXTUREFILE& key);
```

---

#### CSolidTextureHash

**Description**: Cache for solid color textures.

**Structure**:
```c
struct UCSolidTextureHash {
    uint32_t              color;      // RGBA color value
    HTEXTURE__*           handle;     // Texture handle
    UCSolidTextureHash*   next;       // Collision chain
};
```

**Key**: Uses color value as hash key (no string comparison needed).

---

### Handle Management

#### HandleClose

**Signature**: `void HandleClose(HTEXTURE__* handle)`

Decrements reference count and destroys texture if count reaches zero.

**Parameters**:
- `handle`: Texture handle to close

**Description**:
1. Decrements reference count
2. If count reaches 0:
   - Removes from texture cache
   - Releases GPU resources
   - Frees system memory
   - Deletes handle

**Example**:
```c
HTEXTURE__* texture = LoadTexture("Texture.blp");
// ... use texture ...
HandleClose(texture);  // Release reference
```

---

### DBC Cache Functions

#### LoadDBCaches

**Address**: `0x005653c0`  
**Signature**: `void LoadDBCaches()`

Initializes all DBC cache systems and loads required DBC files.

**Description**:
Called during client initialization. Sets up hash tables and loads DBC files including:
- CreatureDisplayInfo.dbc
- CreatureModelData.dbc
- ItemDisplayInfo.dbc
- CharSections.dbc

---

#### DBCache_Initialize

**Address**: `0x005653b0`  
**Signature**: `void DBCache_Initialize()`

Initializes the DBC caching system.

**Description**:
Allocates hash tables for caching DBC records. Must be called before loading any DBC files.

---

#### DBCache_RegisterHandlers

**Address**: `0x00565440`  
**Signature**: `void DBCache_RegisterHandlers()`

Registers callback handlers for DBC cache events.

**Description**:
Sets up callbacks for:
- Cache misses (load from server)
- Cache updates (receive new data)
- Cache invalidation (remove stale entries)

---

#### DBCache_Destroy

**Address**: `0x00565430`  
**Signature**: `void DBCache_Destroy()`

Destroys all DBC cache structures and frees memory.

**Description**:
Called during client shutdown. Frees all cached DBC records and hash table structures.

---

### Character Customization

#### ChangeFaceTexture

**Address**: `0x004b5710` (one variant)  
**Address**: `0x004b6700` (another variant)  
**Signature**: `void ChangeFaceTexture(/* parameters vary */)`

Updates a character's face texture based on customization choices.

**Description**:
Looks up the appropriate texture from CharSections.dbc based on race, gender, and selected face ID. Updates the character model's texture references.

**Related**:
- [`ChangeFacialHairTexture()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:353)
- [`ChangeScalpHairTexture()`](e:/parp-2026/parp-tools/gillijimproject_refactor/src/MdxViewer/documentation/dbc_texturing/11_api_reference.md:361)

---

#### ChangeFacialHairTexture

**Address**: `0x004b5770` (one variant)  
**Address**: `0x004b6740` (another variant)  
**Signature**: `void ChangeFacialHairTexture(/* parameters vary */)`

Updates facial hair (beard, mustache) textures.

---

#### ChangeScalpHairTexture

**Address**: `0x004b5860` (one variant)  
**Address**: `0x004b6830` (another variant)  
**Signature**: `void ChangeScalpHairTexture(/* parameters vary */)`

Updates hairstyle textures for character customization.

---

### Utility Functions

#### BindTexture

**Address**: `0x005a0070`  
**Signature**: `void BindTexture(uint32_t unit, HTEXTURE__* texture)`

Binds a texture to a specific texture unit for rendering.

**Parameters**:
- `unit`: Texture unit index (0-7 typically)
- `texture`: Texture handle to bind

**Description**:
Sets the active texture in the graphics API (OpenGL/DirectX). Must be called before rendering geometry that uses the texture.

**Example**:
```c
// Bind diffuse texture to unit 0
BindTexture(0, model->diffuseTexture);

// Bind normal map to unit 1
BindTexture(1, model->normalMap);

// Render
RenderGeometry();
```

---

#### AllocBlankTexture

**Address**: `0x004c0140` (one variant)  
**Address**: `0x004c0f20` (another variant)  
**Signature**: `HTEXTURE__* AllocBlankTexture(uint32_t width, uint32_t height, uint32_t format)`

Allocates a blank texture with specified dimensions.

**Parameters**:
- `width`: Texture width in pixels
- `height`: Texture height in pixels
- `format`: Pixel format (e.g., RGBA8, DXT1, etc.)

**Returns**: Handle to the blank texture

**Description**:
Creates an empty texture that can be filled with pixel data later. Useful for:
- Render targets
- Dynamic textures
- Procedural content generation

---

## Data Structures

### CModelComplex

Model structure containing geometry and texture references.

```c
struct CModelComplex {
    // Texture array
    TSGrowableArray<CModelTexture> m_textures;
    
    // Other model data
    uint32_t              numVertices;
    uint32_t              numIndices;
    uint32_t              numMaterials;
    // ...
};
```

---

### CModelTexture

Individual texture entry in a model.

```c
struct CModelTexture {
    HTEXTURE__*     handle;         // GPU texture handle
    uint32_t        type;           // Texture type/source
    uint32_t        flags;          // Texture flags
    char            filename[256];  // Original file path
};
```

**Texture Types**:
- `0`: Hardcoded texture (use file from model)
- `1`: Character skin (use player's skin texture)
- `2`: Item texture (use equipped item texture)
- `3+`: Other special types

---

### CGxTexFlags

Texture creation and rendering flags.

```c
struct CGxTexFlags {
    EGxTexFilter    filter;         // Filtering mode
    EGxTexWrap      wrapU;          // U coordinate wrapping
    EGxTexWrap      wrapV;          // V coordinate wrapping
    uint32_t        anisotropy;     // Anisotropic filtering level
    uint32_t        flags;          // Additional flags
};
```

**EGxTexFilter values**:
```c
enum EGxTexFilter {
    GxTex_Nearest = 0,              // Point sampling
    GxTex_Linear = 1,               // Bilinear filtering
    GxTex_LinearMipNearest = 2,     // Bilinear + mip point
    GxTex_LinearMipLinear = 3,      // Trilinear filtering
    GxTex_Anisotropic = 4           // Anisotropic filtering
};
```

**EGxTexWrap values**:
```c
enum EGxTexWrap {
    GxTex_Repeat = 0,               // Tile texture
    GxTex_Clamp = 1,                // Clamp to edge
    GxTex_Mirror = 2                // Mirror repeat
};
```

---

### HTEXTURE__

Opaque texture handle (pointer to internal structure).

**Usage**:
Treat as opaque pointer - never dereference directly. Use provided API functions to manipulate.

```c
// Correct usage
HTEXTURE__* tex = LoadTexture("Texture.blp");
BindTexture(0, tex);
HandleClose(tex);

// INCORRECT - never do this
// tex->internalData = ...;  // Undefined behavior!
```

---

### CreatureDisplayInfoRec

DBC record structure for creature appearance.

```c
struct CreatureDisplayInfoRec {
    uint32_t ID;                    // Record ID
    uint32_t ModelID;               // CreatureModelData reference
    uint32_t SoundID;               // Sound set
    uint32_t ExtraDisplayInfoID;    // Extra display info
    float    Scale;                 // Model scale multiplier
    uint32_t Opacity;               // Alpha (0-255)
    char*    Skin1;                 // Texture override 1
    char*    Skin2;                 // Texture override 2
    char*    Skin3;                 // Texture override 3
    char*    PortraitTextureName;   // Portrait icon
    uint32_t BloodID;               // Blood type
    uint32_t NPCSoundID;            // NPC sound set
    uint32_t ParticleColorID;       // Particle effects
    uint32_t CreatureGeosetData;    // Geoset visibility
    uint32_t ObjectEffectPackageID; // Visual effects
};
```

---

## Error Codes

Common error codes returned by texture functions:

```c
#define TEXTURE_OK                  0x00000000
#define TEXTURE_ERROR_FILE_NOT_FOUND 0x80004005
#define TEXTURE_ERROR_INVALID_FORMAT 0x8000FFFF
#define TEXTURE_ERROR_OUT_OF_MEMORY  0x8007000E
#define TEXTURE_ERROR_GPU_FAILURE    0x88760868
```

---

## Global Variables

### Texture Cache

```c
// Global texture hash table
extern TSHashTable<UCTextureHash, HASHKEY_TEXTUREFILE> g_TextureCache;

// Global solid texture cache  
extern TSHashTable<UCSolidTextureHash, HASHKEY_NONE> g_SolidTextureCache;
```

### DBC Caches

```c
// Creature display info cache
extern DBCache<CreatureDisplayInfoRec, int, HASHKEY_INT> 
    g_CreatureDisplayInfoCache;

// Item display info cache
extern DBCache<ItemDisplayInfoRec, int, HASHKEY_INT>
    g_ItemDisplayInfoCache;
```

---

## Best Practices

### Resource Management

```c
// Good: Always pair LoadTexture with HandleClose
HTEXTURE__* tex = LoadTexture("Texture.blp");
if (tex) {
    UseTexture(tex);
    HandleClose(tex);
}

// Bad: Leaking texture handle
HTEXTURE__* tex = LoadTexture("Texture.blp");
UseTexture(tex);
// Missing HandleClose() - memory leak!
```

### Error Handling

```c
// Good: Check return values
HTEXTURE__* tex = LoadTexture("Texture.blp");
if (!tex) {
    LogError("Failed to load texture");
    tex = LoadFallbackTexture();
}

// Bad: Assuming success
HTEXTURE__* tex = LoadTexture("Texture.blp");
BindTexture(0, tex);  // Crash if tex is NULL!
```

### Performance

```c
// Good: Cache texture lookups
static HTEXTURE__* cachedTexture = NULL;
if (!cachedTexture) {
    cachedTexture = LoadTexture("Texture.blp");
}
BindTexture(0, cachedTexture);

// Bad: Loading every frame
for (int i = 0; i < 1000; i++) {
    HTEXTURE__* tex = LoadTexture("Texture.blp");  // Slow!
    BindTexture(0, tex);
    Render();
    HandleClose(tex);
}
```

---

**Next**: [Code Examples](12_code_examples.md)
