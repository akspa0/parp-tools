# Pre-Legion M2 Format Structure

## Overview
Prior to Legion, M2 files used a non-chunked format with a header-based structure containing offsets pointing to various data sections. This documentation describes the structure of these pre-Legion M2 files.

## File Structure
The M2 file begins with a header containing counts and offsets to various data arrays. Each offset points to the corresponding data block relative to the beginning of the file. Record sizes are not specified in the file itself, but are known based on the data type.

## Header
```cpp
struct M2Header 
{
    /*0x000*/  uint32_t magic;                           // "MD20" - File identifier
    /*0x004*/  uint32_t version;                         // Version of the file format
    /*0x008*/  M2Array<char> name;                       // Model name, should be globally unique
    
    /*0x010*/  uint32_t global_flags;                    // Global model flags
               // 0x01: Flag Tilt X
               // 0x02: Flag Tilt Y
               // 0x08: Use texture combiner combos
               // 0x20: Load physics data
               // 0x80: With this flag unset, demon hunter tattoos stop glowing
               // 0x100: Camera related
               // 0x200: New particle record
               // 0x800: Texture transforms use bone sequences
               // 0x2000: Chunked animation files
               // 0x200000: Use upgraded model format
         
    /*0x014*/  M2Array<M2Loop> global_loops;             // Global animation sequences
    /*0x01C*/  M2Array<M2Sequence> sequences;            // Animation sequence definitions
    /*0x024*/  M2Array<uint16_t> sequenceIdxHashById;    // Mapping of sequence IDs to animations
    /*0x02C*/  M2Array<M2CompBone> bones;                // Skeletal bone definitions
    /*0x034*/  M2Array<uint16_t> boneIndicesById;        // Lookup table for key bones
    /*0x03C*/  M2Array<M2Vertex> vertices;               // Vertex definitions
    
    /*0x044*/  uint32_t num_skin_profiles;               // Number of skin files (Views/LOD)
    
    /*0x048*/  M2Array<M2Color> colors;                  // Color and alpha animation definitions
    /*0x050*/  M2Array<M2Texture> textures;              // Texture definitions
    /*0x058*/  M2Array<M2TextureWeight> texture_weights; // Transparency definitions
    /*0x060*/  M2Array<M2TextureTransform> texture_transforms; // Texture animation definitions
    /*0x068*/  M2Array<uint16_t> textureIndicesById;     // Replaceable texture lookup
    /*0x070*/  M2Array<M2Material> materials;            // Materials and blending modes
    /*0x078*/  M2Array<uint16_t> boneCombos;             // Bone lookup table
    /*0x080*/  M2Array<uint16_t> textureCombos;          // Texture lookup table
    /*0x088*/  M2Array<uint16_t> textureCoordCombos;     // Texture mapping lookup table
    /*0x090*/  M2Array<uint16_t> textureWeightCombos;    // Transparency lookup table
    /*0x098*/  M2Array<uint16_t> textureTransformCombos; // Texture transforms lookup table
    
    /*0x0A0*/  CAaBox bounding_box;                      // Model bounding box
    /*0x0B8*/  float bounding_sphere_radius;             // Model bounding sphere radius
    /*0x0BC*/  CAaBox collision_box;                     // Collision bounding box
    /*0x0D4*/  float collision_sphere_radius;            // Collision sphere radius
    
    /*0x0D8*/  M2Array<uint16_t> collisionIndices;       // Collision triangles
    /*0x0E0*/  M2Array<C3Vector> collisionPositions;     // Collision vertices
    /*0x0E8*/  M2Array<C3Vector> collisionFaceNormals;   // Collision normals
    /*0x0F0*/  M2Array<M2Attachment> attachments;        // Attachment points
    /*0x0F8*/  M2Array<uint16_t> attachmentIndicesById;  // Attachment lookup table
    /*0x100*/  M2Array<M2Event> events;                  // Animation events
    /*0x108*/  M2Array<M2Light> lights;                  // Light definitions
    /*0x110*/  M2Array<M2Camera> cameras;                // Camera definitions
    /*0x118*/  M2Array<uint16_t> cameraIndicesById;      // Camera lookup table
    /*0x120*/  M2Array<M2Ribbon> ribbon_emitters;        // Ribbon emitter definitions
    /*0x128*/  M2Array<M2Particle> particle_emitters;    // Particle emitter definitions
    
    // Optional field, only present if global_flags has flag_use_texture_combiner_combos (0x08) set
    /*0x130*/  M2Array<uint16_t> textureCombinerCombos;  // Second texture material override combos
};
```

## Common Structures

### M2Array
Array structure used throughout M2 files to reference data blocks:
```cpp
template<typename T>
struct M2Array {
    uint32_t size;       // Number of elements
    uint32_t offset;     // Offset to elements, relative to beginning of file
};
```

### M2Loop
Used for global animation sequences:
```cpp
struct M2Loop
{
    uint32_t timestamp;  // Upper limit for global sequence range
};
```

### Animation Structures

#### M2TrackBase
Base structure for animation tracks:
```cpp
struct M2TrackBase {
    uint16_t trackType;       // Interpolation type: 0=none, 1=linear, 2=bezier, 3=hermite
    uint16_t loopIndex;       // Global sequence index, or -1
    M2Array<M2SequenceTimes> sequenceTimes;  // Timestamps for each animation
};
```

#### M2Track
Animation track for various value types:
```cpp
template<typename T>
struct M2Track : M2TrackBase {
    M2Array<M2Array<T>> values;  // Animation values for each timestamp
};
```

## Key Components

### Vertices
```cpp
struct M2Vertex
{
    C3Vector pos;             // Position
    uint8_t bone_weights[4];  // Weights for bone influences (0-255)
    uint8_t bone_indices[4];  // Bone indices
    C3Vector normal;          // Normal vector
    C2Vector tex_coords[2];   // Texture coordinates for two UV layers
};
```

### Bones
```cpp
struct M2CompBone
{
    int32_t key_bone_id;      // Index to key bone lookup or -1
    uint32_t flags;           // Bone flags (billboarding, etc.)
    int16_t parent_bone;      // Parent bone ID or -1
    uint16_t submesh_id;      // Submesh ID
    
    // Union used only in later versions
    union {
        struct {
            uint16_t uDistToFurthDesc;
            uint16_t uZRatioOfChain;
        } CompressData;
        uint32_t boneNameCRC;  // CRC of bone name (for debugging)
    };
    
    M2Track<C3Vector> translation;  // Position animation
    M2Track<M2CompQuat> rotation;   // Rotation animation (compressed quaternion)
    M2Track<C3Vector> scale;        // Scale animation
    C3Vector pivot;                 // Pivot point for bone
};
```

### Textures
```cpp
struct M2Texture
{
    uint32_t type;          // Texture type (0=file, 1=skin, etc.)
    uint32_t flags;         // Texture flags (wrap, etc.)
    M2Array<char> filename; // Texture filename (empty for hardcoded textures)
};
```

### Materials
```cpp
struct M2Material
{
    uint16_t flags;          // Material flags (unlit, two-sided, etc.)
    uint16_t blending_mode;  // Blending mode
};
```

## Relationships with Other Files
Pre-Legion M2 files often worked in conjunction with:
- .skin files (WotLK and beyond): For rendering optimization
- .anim files (Cataclysm and beyond): For storing animation data
- .phys files (MoP and beyond): For physics data

## Implementation Notes
- All offsets in the M2 header are relative to the beginning of the file
- Animation data can be stored in the main file or in external .anim files
- The format evolved over time with additional fields in newer versions
- The structure was replaced by a chunked format in Legion

## Version History
- Version 256-257: Classic (1.0-1.1)
- Version 260-263: Burning Crusade (1.4-1.7)
- Version 264: Wrath of the Lich King (1.8)
- Version 265-272: Cataclysm (1.9-1.16)
- Version 272: Mists of Pandaria, Warlords of Draenor (1.16)
- Version 272-274: Legion, Battle for Azeroth, Shadowlands (1.16-1.18)

The non-chunked format was used until Legion, at which point the chunked format with MD21 chunk was introduced. 