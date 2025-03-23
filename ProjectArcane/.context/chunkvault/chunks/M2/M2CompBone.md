# M2CompBone Structure

## Overview
The M2CompBone structure defines a bone in an M2 model's skeleton. Bones form the skeletal animation system, allowing the model to be animated by transforming vertices based on bone movements. Each bone can have a parent bone, creating a hierarchical skeleton.

## Structure

```cpp
struct M2CompBone {
  int32_t bone_id;                 // Bone ID (index in the bone lookup table)
  uint32_t flags;                  // Bone flags
  int16_t parent_bone;             // Parent bone index (-1 for root bones)
  uint16_t submesh_id;             // Unknown / submesh ID
  uint16_t unknown[2];             // Unknown values
  M2Track<C3Vector> position;      // Position animation data
  M2Track<C4Quaternion> rotation;  // Rotation animation data
  M2Track<C3Vector> scale;         // Scale animation data
  C3Vector pivot;                  // Pivot point for rotation
};
```

## Fields

- **bone_id**: Unique identifier for the bone
- **flags**: Bit flags that control bone behavior and properties
- **parent_bone**: Index of the parent bone in the bone array (-1 if this is a root bone)
- **submesh_id**: Possibly relates to submesh binding or visibility
- **unknown**: Unknown values, possibly padding or reserved fields
- **position**: Track for position animation (translation)
- **rotation**: Track for rotation animation (using quaternions)
- **scale**: Track for scale animation
- **pivot**: Pivot point around which rotations occur

## Animation Tracks

The bone animation data is stored in M2Track structures, which contain keyframes for different animation sequences:

```cpp
template<typename T>
struct M2Track {
  uint16_t interpolation_type;           // Type of interpolation between keyframes
  uint16_t global_sequence;              // Global sequence ID or 0xFFFF
  M2Array<M2Array<uint32_t>> timestamps; // Timestamps for keyframes
  M2Array<M2Array<T>> values;            // Values at each keyframe
};
```

## Bone Flags

The flags field contains bit flags that control various bone properties:
- **0x008**: Spherical billboard - bone always faces the camera for sprites
- **0x010**: Cylindrical billboard X - rotates along X axis to face camera
- **0x020**: Cylindrical billboard Y - rotates along Y axis to face camera
- **0x040**: Cylindrical billboard Z - rotates along Z axis to face camera
- **0x080**: Transformed - indicates this bone can affect vertices
- **0x100**: Kinematic - controlled by physics engine in modern versions
- **0x200**: Animation-powered - transformed by animation control
- **0x400**: Animation-powered with collision - physics enabled
- **0x800**: Has skinning transition - related to animation blending

## Interpolation Types

The interpolation_type field in each M2Track specifies how to interpolate between keyframes:
- **0**: None (no interpolation, use keyframe directly)
- **1**: Linear (linear interpolation between keyframes)
- **2**: Hermite (smooth cubic hermite spline interpolation)
- **3**: Bezier (cubic bezier interpolation)

## Implementation Notes

- The bone hierarchy must be processed from roots to leaves when calculating transformations
- Each bone's transformation is relative to its parent bone
- The pivot point is used as the center of rotation for animation
- Bones with the "transformed" flag can affect vertex positions
- Billboard bones always face the camera with different constraints based on flags
- The global_sequence field in an M2Track, if not 0xFFFF, references a global animation loop
- For non-global animations, the timestamps and values arrays have entries for each animation sequence
- Quaternion rotations should be normalized before use in transformations
- Bone IDs are sometimes used as references from other structures (e.g., attachments)

## Usage in Animation

To animate a model using its skeleton:
1. Calculate the transformation matrix for each bone:
   - Start with root bones (parent_bone = -1)
   - For each bone, calculate its local transformation using position, rotation, pivot, and scale
   - Combine with the parent's transformation to get the final bone transformation
2. For each vertex in the model:
   - Get the bones affecting the vertex and their weights
   - Apply a weighted blend of bone transformations to the vertex
3. Update this process each frame using the appropriate keyframes for the current animation

## Version Differences

- The M2CompBone structure has evolved over different WoW versions
- In earlier versions, some fields might have different meanings or be absent
- Physics-related flags were added in later versions with the introduction of the physics engine
- Later versions may have additional specialized bone types with extended functionality 