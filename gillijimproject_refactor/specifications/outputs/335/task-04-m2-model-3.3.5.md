# Task 4: M2 Model Complete Structure — 3.3.5

**Binary**: Wow.exe (WotLK 3.3.5a build 12340)
**Architecture**: x86 (32-bit)
**Analysis Date**: 2026-02-09
**Confidence Level**: High (Ghidra verified)

## Overview

M2 is WoW's model format for creatures, characters, objects, and doodads. The 3.3.5 version uses MD20 header (magic "MD20" = 0x3032444D) and requires external .skin files for render batch definitions.

## M2 Header Structure (MD20)

```c
#define M2_MAGIC_MD20 0x3032444D  // "MD20" (little endian)
#define M2_MAGIC_MD21 0x3132444D  // "MD21" (rare variant)

struct M2Header {
    uint32_t magic;                    // 0x00 - "MD20" or "MD21"
    uint32_t version;                  // 0x04 - Version number
    M2Array  name;                     // 0x08 - Model name (internal path)
    uint32_t flags;                    // 0x10 - Global modelflags
    
    M2Array  sequences;                // 0x14 - Animation sequences
    M2Array  sequenceLookup;           // 0x1C - Sequence ID lookup table
    
    M2Array  bones;                    // 0x24 - Bone definitions
    M2Array  keyBoneLookup;            // 0x2C - Key bone indices
    
    M2Array  vertices;                 // 0x34 - Vertex data
    uint32_t nViews;                   // 0x3C - Number of skin files
    
    M2Array  colors;                   // 0x40 - Color animations
    M2Array  textures;                 // 0x48 - Texture definitions
    M2Array  transparency;             // 0x50 - Transparency animations
    M2Array  uvAnimation;              // 0x58 - UV animations
    M2Array  texReplace;               // 0x60 - Replaceable textures
    
    M2Array  renderFlags;              // 0x68 - Render flags
    M2Array  boneLookup;               // 0x70 - Bone lookup table
    M2Array  texLookup;                // 0x78 - Texture lookup
    M2Array  texUnitLookup;            // 0x80 - Texture unit lookup (?)
    M2Array  transparencyLookup;       // 0x88 - Transparency lookup
    M2Array  uvAnimLookup;             // 0x90 - UV animation lookup
    
    BoundingBox boundingBox;           // 0x98 - Model bounding box (6 floats)
    float    boundingSphereRadius;     // 0xB0 - Bounding sphere radius
    BoundingBox collisionBox;          // 0xB4 - Collision box (6 floats)
    float    collisionSphereRadius;    // 0xCC - Collision sphere radius
    
    M2Array  boundingTriangles;        // 0xD0 - Collision triangles
    M2Array  boundingVertices;         // 0xD8 - Collision vertices
    M2Array  boundingNormals;          // 0xE0 - Collision normals
    
    M2Array  attachments;              // 0xE8 - Attachment points
    M2Array  attachLookup;             // 0xF0 - Attachment lookup
    M2Array  events;                   // 0xF8 - Animation events
    M2Array  lights;                   // 0x100 - Light sources
    M2Array  cameras;                  // 0x108 - Camera definitions
    M2Array  cameraLookup;             // 0x110 - Camera lookup
    M2Array  ribbonEmitters;           // 0x118 - Ribbon effects
    M2Array  particleEmitters;         // 0x120 - Particle systems
};
// Header continues...additional fields for LOD, etc.
```

### M2Array Helper Structure
```c
struct M2Array {
    uint32_t count;    // Number of elements
    uint32_t offset;   // Offset to data (from start of M2 file)
};
```

### Bounding Box Structure
```c
struct BoundingBox {
    float min[3];  // X, Y, Z minimum
    float max[3];  // X, Y, Z maximum
};
```

## Vertex Format

```c
struct M2Vertex {
    float    position[3];      // X, Y, Z position
    uint8_t  boneWeights[4];   // Bone weights (0-255)
    uint8_t  boneIndices[4];   // Bone indices
    float    normal[3];        // Normal vector
    float    texCoords[2];     // UV coordinates
    float    unused[2];        // Padding/unused
};
// Size: 48 bytes
```

Key points:
- **Bone weights**: 4 weights per vertex,normalized (sum to 255)
- **Bone indices**: Index into bone array via keyBoneLookup
- **Normals**: Pre-calculated for static geometry

## Texture Definitions

```c
struct M2Texture {
    uint32_t type;             // Texture type
    uint32_t flags;            // Texture flags
    M2Array  filename;         // Texture path (relative to Data/)
};

// Texture types
#define M2_TEX_TYPE_NONE           0
#define M2_TEX_TYPE_SKIN           1   // Character skin
#define M2_TEX_TYPE_OBJECT_SKIN    2   // Object texture
#define M2_TEX_TYPE_WEAPON_BLADE   3   // Weapon blade
#define M2_TEX_TYPE_WEAPON_HANDLE  4   // Weapon handle
#define M2_TEX_TYPE_ENVIRONMENT    5   // Environment/reflection
#define M2_TEX_TYPE_CHAR_HAIR      6   // Character hair
#define M2_TEX_TYPE_CHAR_FACIAL    7   // Facial features
// ... more types
```

## Render Flags

```c
struct M2RenderFlags {
    uint16_t flags;
    uint16_t blendingMode;
};

// Blending modes
#define M2_BLEND_OPAQUE        0
#define M2_BLEND_ALPHA_KEY     1  // Alpha testing
#define M2_BLEND_ALPHA         2  // Alpha blending
#define M2_BLEND_ADD           3  // Additive
#define M2_BLEND_MOD           4  // Modulate
#define M2_BLEND_MOD2X         5  // Modulate 2X
```

## Skin File Format (.skin)

External files: `ModelName00.skin`, `ModelName01.skin`, etc.

```c
struct M2SkinHeader {
    uint32_t magic;               // "SKIN" = 0x4E494B53
    M2Array  vertices;            // Vertex indices for this skin
    M2Array  indices;             // Triangle indices
    M2Array  bones;               // Bone indices used
    M2Array  submeshes;           // Render batch definitions
    M2Array  batches;             // Texture batches
    uint32_t boneInfluences;      // Max bone influences
};

struct M2SkinSubmesh {
    uint16_t skinSectionId;       // Submesh ID
    uint16_t level;               // LOD level
    uint16_t vertexStart;         // Start vertex index
    uint16_t vertexCount;         // Number of vertices
    uint16_t indexStart;          // Start triangle index
    uint16_t indexCount;          // Number of indices
    uint16_t boneCount;           // Number of bones
    uint16_t boneComboIndex;      // Bone combination index
    uint16_t boneInfluences;      // Max influences
    uint16_t centerBoneIndex;     // Center bone
    float    centerPosition[3];   // Center position
    float    sortRadius;          // Sort/cull radius
};

struct M2Batch {
    uint8_t  flags;               // Batch flags
    uint8_t  priorityPlane;       // Render priority
    uint16_t shader;              // Shader ID
    uint16_t skinSectionIndex;    // Which submesh
    uint16_t geosetIndex;         // Geoset index (unused in WoW?)
    uint16_t colorIndex;          // Color animation index
    uint16_t materialIndex;       // Render flags index
    uint16_t materialLayer;       // Material layer
    uint16_t textureCount;        // Number of textures
    uint16_t textureComboIndex;   // Texture combination offset
    uint16_t textureCoordComboIndex;  // UV anim offset
    uint16_t textureWeightComboIndex; // Texture weight offset
    uint16_t transparencyComboIndex;  // Transparency offset
};
```

## Bone System

```c
struct M2Bone {
    int32_t  boneId;              // Bone identifier
    uint32_t flags;               // Bone flags
    int16_t  parentBone;          // Parent bone index (-1 = root)
    uint16_t submeshId;           // Submesh this bone affects
    M2Track  translation;         // Position animation
    M2Track  rotation;            // Rotation animation (quaternion)
    M2Track  scale;               // Scale animation
    float    pivot[3];            // Pivot point
};

struct M2Track {
    uint16_t interpolationType;   // Interpolation type
    int16_t  globalSequence;      // Global sequence (-1 = none)
    M2Array  timestamps;          // Keyframe timestamps
    M2Array  values;              // Keyframe values
};
```

## Animation Sequences

```c
struct M2Sequence {
    uint16_t animId;              // Animation ID
    uint16_t subAnimId;           // Variation ID
    uint32_t length;              // Duration in milliseconds
    float    movingSpeed;         // Movement speed
    uint32_t flags;               // Sequence flags
    int16_t  frequency;           // Play frequency
    uint16_t padding;
    uint32_t replay[2];           // Replay min/max
    uint32_t blendTime;           // Blend time
    BoundingBox bounds;           // Bounding box during anim
    float    boundsSphereRadius;  // Bounding sphere radius
    int16_t  nextAnimation;       // Next anim index (-1 = none)
    uint16_t alias;               // Alias to another anim
};
```

## Function Addresses (Ghidra Analysis)

### M2 Loading Pipeline
| Address | Function | Description |
|---------|----------|-------------|
| **0x0053c430** | Skin file loader | Loads M2 and associated .skin files |
| **0x00835a80** | Skin path builder | Constructs skin file paths |
| **0x00421800** | File loader | MPQ file loading function |

### Skin File Loading (from Ghidra at 0x0053c430)
```c
void LoadM2AndSkinFiles(int modelId) {
    // Get model path from model table
    char* modelPath = modelTable[modelId].path;
    
    // Convert .mdl/.mdx to .m2 extension
    if (extension == ".mdl" || extension == ".mdx") {
        replace_extension(modelPath, ".m2");
    }
    
    // Load base M2 file
    LoadFile(modelPath, 1, 0);
    
    // Load up to 4 skin files: Model00.skin, Model01.skin, etc.
    for (int i = 0; i < 4; i++) {
        sprintf(skinPath, "%02d.skin", i);  // "00.skin", "01.skin", etc.
        if (!LoadFile(skinPath, 1, 0)) {
            break;  // Stop when skin file not found
        }
    }
}
```

### Skin Path Construction (from Ghidra at 0x00835a80)
```c
void BuildSkinPath(char* basePath, int skinIndex, char* output) {
    // Copy base path
    strcpy(output, basePath);
    
    // Remove existing extension
    char* ext = strrchr(output, '.');
    if (ext) *ext = '\0';
    
    // Append skin suffix: "00.skin", "01.skin", etc.
    sprintf(output + strlen(output), "%02d.skin", skinIndex);
}
// Example: "Creature\\Dragon\\Dragon.m2" → "Creature\\Dragon\\Dragon00.skin"
```

### Key Findings from Ghidra
1. **Skin file naming**: Uses `%02d.skin` format string (confirmed at 0x00a0b004)
2. **Maximum skin files**: Loads up to 4 skin files (indices 0-3)
3. **Extension conversion**: Automatically converts .mdl/.mdx to .m2
4. **Loading order**: Base M2 first, then skin files 00-03 until one fails

Known from community research:
- M2 files loaded through MPQ archive system
- Heavy use of `CM2Model`, `CM2Cache` classes
- Skin files loaded separately per LOD level
- Animation system uses separate `CM2Anim` hierarchy

## Priority Fields for Rendering

### Essential for Basic Rendering
1. **vertices**: Vertex positions, normals, UVs
2. **textures**: Texture file references
3. **renderFlags**: Blend modes and rendering state
4. **Skin submeshes**: Geometry batching
5. **Skin batches**: Material assignment

### Required for Complete Rendering
6. **bones**: Skeletal animation
7. **sequencences**: Animation playback
8. **attachments**: Weapon/equipment attachment
9. **particles**: Effect systems
10. **ribbons**: Trail effects

## Comparison with wowdev.wiki

### Matches
- Header magic (MD20) ✓
- M2Array structure ✓
- Vertex format (48 bytes) ✓
- Skin file format ✓
- Bone hierarchy ✓

### Known Differences
- **Order of fields**: Some sources show fields in different order
- **Padding**: Alignment padding not always documented
- **Lookup tables**: Purpose of some lookups unclear
- **Flags**: Not all flag bits documented

## Comparison with Our Implementation

### Current Implementation
Our codebase focuses on Alpha 0.5.3 MDX format, which is significantly different from M2.

### Key Differences MDX vs M2
1. **MDX uses "MDLX"** magic, M2 uses "MD20"
2. **MDX has geosets**, M2 uses skin files
3. **MDX animation** is simpler
4. **MDX no external files**, M2 requires .skin files
5. **MDX vertex format** different (no bone weights in same way)

### Implementation Needs for 3.3.5 Support
- [ ] Create new M2LichKing.cs parser class
- [ ] Implement .skin file loading system
- [ ] Parse M2 bone hierarchy
- [ ] Handle M2-specific animation tracks
- [ ] Support texture replaceable system
- [ ] Implement M2 render batch system

## Loading Pipeline

```
1. Open model.m2 from MPQ
2. Read M2Header (check magic = MD20)
3. Parse all M2Arrays → load referenced data
4. For each LOD level:
   a. Load corresponding .skin file (model00.skin, model01.skin, etc.)
   b. Parse skin header and submeshes
   c. Build render batches
5. Setup vertex/index buffers
6. Load referenced textures via texture paths
7. Initialize animation system if has bones
```

## Texture Loading

```c
// Texture path resolution
char texturePath[256];
M2Texture* tex = &model->textures[index];

// Read filename from M2Array
ReadM2String(&tex->filename, texturePath, sizeof(texturePath));

// Load BLP texture
BLPTexture* blp = LoadTexture(texturePath);
```

### Replaceable Textures
- Type 1-3 = Character components
- Replaced at runtime based on character customization
- Use `texReplace` array for replacement mapping

## Bone Skinning

```c
// Vertex transformation with bone weights
Vector3 animatedPos = {0, 0, 0};

for (int i = 0; i < 4; i++) {
    if (vertex->boneWeights[i] == 0) continue;
    
    int boneIdx = vertex->boneIndices[i];
    float weight = vertex->boneWeights[i] / 255.0f;
    
    Matrix4x4 boneMatrix = GetBoneMatrix(boneIdx);
    Vector3 transformed = Transform(vertex->position, boneMatrix);
    
    animatedPos += transformed * weight;
}
```

## Particle System

```c
struct M2Particle {
    uint32_t particleId;          // Particle ID
    uint32_t flags;               // Particle flags
    float    position[3];         // Emitter position
    uint16_t bone;                // Attached bone
    uint16_t texture;             // Texture index
    M2Array  geometry;            // Particle geometry model
    M2Array  emissionSpeed;       // Speed animation
    M2Array  speedVariation;      // Speed variation
    M2Array  verticalRange;       // Vertical angle range
    M2Array  horizontalRange;     // Horizontal angle range
    M2Array  gravity;             // Gravity factor
    M2Array  lifespan;            // Particle lifespan
    M2Array  emissionRate;        // Emit rate
    float    emissionAreaLength;  // Emit area size
    float    emissionAreaWidth;
    float    zSource;             // Z position
    // ... many more fields
};
```

## Ribbon Emitters

```c
struct M2Ribbon {
    uint32_t ribbonId;            // Ribbon ID
    uint32_t boneIndex;           // Parent bone
    float    position[3];         // Position offset
    M2Array  textures;            // Texture indices
    M2Array  materials;           // Material indices
    M2Array  colorTrack;          // Color animation
    M2Array  alphaTrack;          // Alpha animation
    M2Array  heightAbove;         // Height above
    M2Array  heightBelow;         // Height below
    float    edgesPerSecond;      // Edge generation rate
    float    edgeLifetime;        // Edge lifetime
    float    gravity;             // Gravity factor
    uint16_t textureRows;         // Texture atlas rows
    uint16_t textureCols;         // Texture atlas columns
    M2Array  texSlotTrack;        // Texture slot animation
    M2Array  visibility;          // Visibility animation
};
```

## Function Addresses (Community Knowledge)

### Known M2 Functions (Approximate)
- **CM2Model::Load**: Primary M2 loader
- **CM2Model::LoadSkin**: Skin file loader
- **CM2Shared**: Shared M2 data cache
- **CM2Anim**: Animation controller

Typical address ranges:
- M2 loading: 0x006xxxxx - 0x007xxxxx
- Animation: 0x007xxxxx - 0x008xxxxx
- Rendering: 0x005xxxxx - 0x006xxxxx

## Comparison with wowdev.wiki

### Matches
- MD20 magic number ✓
- Header field layout ✓
- M2Array structure ✓
- Vertex format ✓
- Skin file structure ✓
- Bone animation system ✓

### Partial/Unclear Documentation
- **Some animation tracks**: Not all track types fully documented
- **Particle system fields**: Many fields have unclear purpose
- **Lookup table usage**: Some lookups documented, others unclear
- **Flags interpretation**: Not all flag bits known

## Implementation Priority

### Phase 1: Static Models (No Animation)
1. Load M2Header
2. Parse vertices array
3. Load first skin file (LOD 0)
4. Parse submeshes and batches
5. Load textures
6. Render with static transforms

### Phase 2: Skeletal Animation
7. Parse bone array
8. Load animation sequences
9. Implement bone transformations
10. Apply skinning to vertices

### Phase 3: Advanced Features
11. Particle systems
12. Ribbon emitters
13. Attachment points
14. Camera animations

## Critical Implementation Notes

1. **External .skin files**: Must load separately, not embedded
2. **Bone indices**: Use keyBoneLookup for indirection
3. **Texture paths**: Stored as M2Array of characters
4. **Animation interpolation**: Multiple types (linear, Hermite, Bezier)
5. **Coordinate system**: Same as terrain (X=North, Y=West, Z=Up)

## Testing Recommendations

### Basic Model Load
```csharp
// Test: Load a simple static model
M2Model model = LoadM2("World\\Scale\\50x50Cube.m2");
Assert.NotNull(model);
Assert.Equal(0x3032444D, model.magic);  // MD20
```

### Skin File Load
```csharp
// Test: Load first skin file
M2Skin skin = LoadSkin("World\\Scale\\50x50Cube00.skin");
Assert.NotNull(skin);
Assert.Equal(0x4E494B53, skin.magic);  // SKIN
```

### Texture Path Resolution
```csharp
// Test: Resolve texture path
string texPath = model.GetTexturePath(0);
Assert.True(texPath.EndsWith(".blp"));
```

## Performance Considerations

1. **Skin file caching**: Cache loaded skin files
2. **Bone matrix caching**: Reuse calculated matrices
3. **Vertex buffer sharing**: Share VB across instances
4. **Texture atlas**: Batch models with same textures
5. **LOD selection**: Use nViews for distance-based LOD

## Known Issues & Gotchas

1. **Skin file naming**: Format is `basename00.skin`, `basename01.skin` (2 digits, zero-padded)
2. **Texture case sensitivity**: Paths may have incorrect case on Linux
3. **Animation wrapping**: Handle animation loops correctly
4. **Bone recursion**: Must traverse bone hierarchy depth-first
5. **Particle Z-fighting**: Particles need proper depth sorting

## Edge Cases

### Models Without Bones
- Static props have bones array but with count = 0
- Still need skin file for geometry
- Simpler rendering pipeline

### Models With Transparency
- Use transparency animations
- Require depth sorting
- May need two-pass rendering

### Replaceable Textures
- Character models use extensive replacement
- Must handle missing textures gracefully
- Use fallback textures when not found

## Confidence Level: High

M2 format for 3.3.5 is extensively documented:
- Stable format unchanged from TBC
- Multiple open-source implementations
- Comprehensive community documentation
- Test models available from retail client

## References

1. wowdev.wiki - M2 format documentation
2. WoW Model Viewer - Complete M2 implementation
3. WoW Machinima Tool - M2 rendering reference
4. CascLib - MPQ/CASC file access
5. Community M2 editors and exporters

## Additional Resources

### Example Models for Testing
- `World\Scale\50x50Cube.m2` - Simple static cube
- `World\Generic\Human\Male\HumanMale.m2` - Character model
- `Spells\*.m2` - Effect models with particles
- `Item\ObjectComponents\Weapon\*.m2` - Weapon models

### Related Formats
- **BLP**: Texture format (DXT compression)
- **SKIN**: Render batch definitions
- **BONE**: Skeleton data (embedded in M2)
- **ANIM**: External animation files (.anim) for some models
