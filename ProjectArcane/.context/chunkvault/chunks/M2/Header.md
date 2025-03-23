# M2 Header Structure

## Overview
The M2 Header is the foundation of the pre-Legion M2 file format. It appears at the beginning of the file and contains metadata about the model along with offsets to various data arrays within the file. The header defines the number and location of all model components such as vertices, bones, animations, textures, and more.

## Structure
```cpp
struct M2Header {
  /*0x000*/  uint32_t magic;                      // "MD20" signature
  /*0x004*/  uint32_t version;                    // Version number (see version table)
  /*0x008*/  M2Array<char> name;                  // Model name, null-terminated string
  /*0x010*/  uint32_t globalFlags;                // Global model flags (see flags section)
  /*0x014*/  M2Array<M2Loop> global_loops;        // Global sequences timing information
  /*0x01C*/  M2Array<M2Sequence> sequences;       // Animation sequence definitions
  /*0x024*/  M2Array<uint16_t> sequenceIdxHashById; // Animation sequence lookup table
  /*0x02C*/  M2Array<M2CompBone> bones;           // Bone definitions
  /*0x034*/  M2Array<uint16_t> boneIndicesById;   // Key bone lookup table
  /*0x03C*/  M2Array<M2Vertex> vertices;          // Vertex data
  /*0x044*/  uint32_t num_skin_profiles;          // Number of skin profiles (in separate .skin files)
  /*0x048*/  M2Array<M2Color> colors;             // Color animation definitions
  /*0x050*/  M2Array<M2Texture> textures;         // Texture definitions
  /*0x058*/  M2Array<M2TextureWeight> texture_weights; // Texture transparency animations
  /*0x060*/  M2Array<M2TextureTransform> texture_transforms; // Texture coordinate animations
  /*0x068*/  M2Array<uint16_t> textureIndicesById; // Replaceable texture lookup
  /*0x070*/  M2Array<M2Material> materials;       // Materials/render flags
  /*0x078*/  M2Array<uint16_t> boneCombos;        // Bone lookup table
  /*0x080*/  M2Array<uint16_t> textureCombos;     // Texture lookup table
  /*0x088*/  M2Array<uint16_t> textureCoordCombos; // Texture mapping lookup table
  /*0x090*/  M2Array<uint16_t> textureWeightCombos; // Transparency lookup table
  /*0x098*/  M2Array<uint16_t> textureTransformCombos; // Texture transform lookup table
  /*0x0A0*/  CAaBox bounding_box;                 // Model bounding box
  /*0x0B8*/  float bounding_sphere_radius;        // Model bounding sphere radius
  /*0x0BC*/  CAaBox collision_box;                // Collision bounding box
  /*0x0D4*/  float collision_sphere_radius;       // Collision sphere radius
  /*0x0D8*/  M2Array<uint16_t> collisionIndices;  // Collision triangle indices
  /*0x0E0*/  M2Array<C3Vector> collisionPositions; // Collision vertex positions
  /*0x0E8*/  M2Array<C3Vector> collisionFaceNormals; // Collision face normals
  /*0x0F0*/  M2Array<M2Attachment> attachments;   // Attachment points
  /*0x0F8*/  M2Array<uint16_t> attachmentIndicesById; // Attachment lookup table
  /*0x100*/  M2Array<M2Event> events;             // Animation events
  /*0x108*/  M2Array<M2Light> lights;             // Light definitions
  /*0x110*/  M2Array<M2Camera> cameras;           // Camera definitions
  /*0x118*/  M2Array<uint16_t> cameraIndicesById; // Camera lookup table
  /*0x120*/  M2Array<M2Ribbon> ribbon_emitters;   // Ribbon emitter definitions
  /*0x128*/  M2Array<M2Particle> particle_emitters; // Particle emitter definitions
  /*0x130*/  M2Array<uint16_t> textureCombinerCombos; // (Optional) Second texture material override combos
};
```

## Fields
- **magic**: "MD20" identifier that confirms this is an M2 file
- **version**: Format version number (see version table)
- **name**: Model name as null-terminated string
- **globalFlags**: Global model flags controlling various features
- **global_loops**: Timing information for global animation sequences
- **sequences**: Animation sequence definitions
- **sequenceIdxHashById**: Lookup table for animation sequences by ID
- **bones**: Skeletal bone definitions
- **boneIndicesById**: Lookup table for key bones
- **vertices**: Vertex data including positions, normals, and texture coordinates
- **num_skin_profiles**: Number of skin profiles (stored in separate .skin files)
- **colors**: Color animation definitions
- **textures**: Texture definitions referencing texture filenames
- **texture_weights**: Transparency animation definitions
- **texture_transforms**: Texture coordinate animation definitions
- **textureIndicesById**: Lookup table for replaceable textures
- **materials**: Material definitions controlling rendering properties
- **boneCombos**: Bone lookup table for skeletal animations
- **textureCombos**: Texture lookup table for various texture units
- **textureCoordCombos**: Texture coordinate mapping lookup table
- **textureWeightCombos**: Transparency animation lookup table
- **textureTransformCombos**: Texture transform lookup table
- **bounding_box**: Axis-aligned bounding box for the model
- **bounding_sphere_radius**: Radius of the model's bounding sphere
- **collision_box**: Axis-aligned bounding box for collision detection
- **collision_sphere_radius**: Radius of the collision sphere
- **collisionIndices**: Triangle indices for collision mesh
- **collisionPositions**: Vertex positions for collision mesh
- **collisionFaceNormals**: Face normals for collision mesh
- **attachments**: Attachment point definitions
- **attachmentIndicesById**: Lookup table for attachment points
- **events**: Animation event definitions
- **lights**: Light source definitions
- **cameras**: Camera definitions
- **cameraIndicesById**: Lookup table for cameras
- **ribbon_emitters**: Ribbon emitter definitions
- **particle_emitters**: Particle emitter definitions
- **textureCombinerCombos**: (Only if global_flags.flag_use_texture_combiner_combos is set) Texture combiner combos

## M2Array Structure
The M2Array structure is used throughout the header to define arrays of data:
```cpp
template<typename T>
struct M2Array {
  uint32_t size;    // Number of elements in the array
  uint32_t offset;  // Offset from the beginning of the file
};
```

## Global Flags
The globalFlags field contains bit flags that control various model features:
- **0x01**: Flag to tilt model along X axis
- **0x02**: Flag to tilt model along Y axis
- **0x08**: Use texture combiner combos (adds textureCombinerCombos array)
- **0x20**: Load physics data (in Mists of Pandaria+)
- **0x80**: With this flag unset, demon hunter tattoos stop glowing
- **0x100**: Camera-related flag
- **0x200**: New particle record format
- **0x800**: Texture transforms use bone sequences

## Version History
| Version | Expansion | Notes |
|---------|-----------|-------|
| 256-257 | Classic | Original format |
| 260-263 | TBC | The Burning Crusade updates |
| 264 | Wrath | Wrath of the Lich King format |
| 265-272 | Cataclysm | Multiple format improvements |
| 272+ | MoP/WoD | Expanded feature set |
| 274+ | Legion+ | Pre-chunked format, later replaced by chunked format |

## Implementation Notes
- All offsets in the header are relative to the start of the file
- The M2Array structure is used to locate arrays of data within the file
- The version number determines which fields are valid and how they should be interpreted
- Some fields may not be present in older versions of the format
- In Legion+, this format was replaced by the chunked format where the MD21 chunk contains this data
- When loading a Legion+ chunked M2 file, all offsets within the MD21 chunk are relative to the beginning of the chunk data, not the file
- The header must be read first to determine the size and location of all other data in the file
- Many fields are optional and may be indicated by a zero size or offset in their M2Array

## Legacy Support
- This header format was used from Classic through Legion
- In Legion, the chunked format became available, but some files still used this format
- When implementing, check for the "MD20" magic value to determine if this format is used
- If "MD20" is not found, the file likely uses the chunked format starting with an MD21 chunk 