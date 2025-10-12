# M2.SKIN File Format

## Overview
The .skin files were introduced in Wrath of the Lich King (WotLK) to store optimization data for rendering M2 models. They contain information about submeshes, texture units, batches, and bone mappings that was previously included directly in the M2 files. Each M2 model can have multiple .skin files, one for each level of detail (LOD), named as "Modelname0x.skin", where x is a digit from 0 to 3 representing each LOD level.

## File Structure
The .skin file begins with a header followed by several data blocks. All offsets in the .skin file are relative to the file itself, rather than the parent M2 file.

## Header
```cpp
struct M2SkinProfile {
    uint32_t magic;                         // 'SKIN'
    M2Array<unsigned short> vertices;       // Vertex lookup table
    M2Array<unsigned short> indices;        // Index lookup table
    M2Array<ubyte4> bones;                  // Bone lookup table
    M2Array<M2SkinSection> submeshes;       // Submesh definitions
    M2Array<M2Batch> batches;               // Texture unit/batch definitions
    uint32_t boneCountMax;                  // Maximum number of bones per draw call
    M2Array<M2ShadowBatch> shadow_batches;  // Shadow batch definitions (MoP+)
};
```

## Data Blocks

### Vertices
This is a lookup table to select a subset of vertices from the global vertex list in the M2 file:
```cpp
struct VertexLookup {
    uint16_t vertex;    // Index into the global vertex list
};
```

### Indices
This is a lookup table to select a subset of vertices from the local vertex list. These indices form triangles for rendering:
```cpp
struct IndexLookup {
    uint16_t index;     // Index into the local skin vertex list
};
```
Indices form triangles, with every 3 indices representing a single triangle. The total number of indices should be divisible by 3. Triangles are right-handed.

### Bones
This is a lookup table to select a subset of bones from the global bone list in the M2 file:
```cpp
struct BoneLookup {
    uint8_t bones[4];   // 4 indices into the global bone list
};
```
Blizzard uses a standard 4-bone rig for animations, so each entry represents 4 bone indices.

### Submeshes
Defines sections of the model with distinct properties:
```cpp
struct M2SkinSection {
    uint16_t skinSectionId;            // Mesh part ID, usually 0
    uint16_t level;                    // Level of detail (LOD), usually 0
    uint16_t vertexStart;              // First vertex in the lookup table
    uint16_t vertexCount;              // Number of vertices
    uint16_t indexStart;               // First index in the lookup table
    uint16_t indexCount;               // Number of indices
    uint16_t boneCount;                // Number of bones referenced
    uint16_t boneComboIndex;           // First bone lookup in the bone combo table
    uint16_t boneInfluences;           // Number of bone indices per vertex (max 4)
    uint16_t centerBoneIndex;          // Center bone index
    C3Vector centerPosition;           // Average position of all vertices
    C3Vector sortCenterPosition;       // Center of sorting
    float sortRadius;                  // Sorting sphere radius
};
```

### Texture Units/Batches
Defines the textures and materials for each batch of geometry:
```cpp
struct M2Batch {
    uint8_t flags;                       // Control flags for the batch
    int8_t priorityPlane;                // Render priority order
    uint16_t shader_id;                  // Shader identifier
    uint16_t skinSectionIndex;           // Index to the submesh
    uint16_t geosetIndex;                // Flags2 in newer versions (0x2 - projected, 0x8 - requires EDGF)
    uint16_t colorIndex;                 // Index into the colors block, or -1
    uint16_t materialIndex;              // Index into the materials/render flags block
    uint16_t materialLayer;              // Material layer (capped at 7)
    uint16_t textureCount;               // Number of textures (1 to 4)
    uint16_t textureComboIndex;          // Index into texture lookup table
    uint16_t textureCoordComboIndex;     // Index into texture mapping lookup table
    uint16_t textureWeightComboIndex;    // Index into transparency lookup table
    uint16_t textureTransformComboIndex; // Index into UV animation lookup table
};
```

#### Flags
- **0x01**: Materials invert something
- **0x02**: Transform
- **0x04**: Projected texture
- **0x10**: Batch compatible
- **0x20**: Projected texture?
- **0x40**: Don't multiply transparency by texture weight

#### Shader ID
The shader_id field determines which shaders to use for rendering. In WotLK and earlier, this is often 0 in the file but computed at runtime. In Cataclysm and later, it's a direct reference to a shader effect.

### Shadow Batches
Added in MoP, these define how the model casts shadows:
```cpp
struct M2ShadowBatch {
    uint8_t flags;              // Control flags
    uint8_t flags2;             // Additional flags
    uint16_t unknown1;
    uint16_t submesh_id;        // Reference to submesh
    uint16_t texture_id;        // Texture index (already looked up)
    uint16_t color_id;          // Color index
    uint16_t transparency_id;   // Transparency index (already looked up)
};
```

## Dependencies
- Requires the parent M2 model, as it references vertices, bones, and other data in the main M2 file
- References textures, materials, and animations defined in the M2 file
- May reference external textures through the file ID system

## Usage
The .skin files are used to:
- Optimize rendering by breaking the model into manageable submeshes
- Define multiple levels of detail (LOD) for efficient rendering at different distances
- Provide material and texture information for each part of the model
- Map bones to vertices for skeletal animation
- Define how shadows are cast by the model

## Legacy Support
- Prior to WotLK, this data was stored directly in the M2 file
- Models from Vanilla and TBC may still use the embedded skin profiles
- Modern versions (Legion+) reference skin files via the SFID chunk in the M2 file

## Implementation Notes
- Each .skin file represents one level of detail
- The "0" LOD is the highest detail version, with higher numbers being lower detail
- Batches may need to be sorted for correct transparency rendering
- Login screens often require special handling due to depth buffer settings
- Shader selection may be static (Cata+) or dynamic (WotLK and earlier)
- Shadow batches are auto-generated if certain criteria are met

## Version History
- Introduced in Wrath of the Lich King
- Enhanced in MoP with shadow batch support
- Referenced via file IDs in Legion+ rather than filenames 