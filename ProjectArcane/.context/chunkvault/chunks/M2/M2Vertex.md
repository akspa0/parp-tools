# M2Vertex Structure

## Overview
The M2Vertex structure defines a vertex in an M2 model's geometry. Each vertex contains position, normal, texture coordinates, and bone weights information. The vertex data is used for rendering the model and for skeletal animation.

## Structure

```cpp
struct M2Vertex {
    C3Vector position;         // Vertex position (x, y, z)
    uint8_t bone_weights[4];   // Weights for up to 4 bones (0-255)
    uint8_t bone_indices[4];   // Indices into the bone lookup table
    C3Vector normal;           // Normal vector (x, y, z)
    C2Vector texture_coords[2]; // Texture coordinates for 2 texture units
};
```

## Fields

- **position**: 3D coordinates (x, y, z) of the vertex position in model space
- **bone_weights**: Influence weights of up to 4 bones on this vertex (0-255, normalized)
- **bone_indices**: Indices into the bone lookup table for up to 4 bones affecting this vertex
- **normal**: Normal vector (x, y, z) for lighting calculations
- **texture_coords**: Texture coordinates (u, v) for up to 2 texture units

## Bone Weights

The bone_weights field contains the weights of how much each of up to 4 bones influences this vertex. These weights range from 0 to 255, with the sum typically being 255 (representing 100% influence). When a value is 0, the corresponding bone has no influence on the vertex. The bone_indices field identifies which bones affect this vertex.

To calculate the influence percentage of each bone:
- `influence_percentage = bone_weight / 255.0`

For proper vertex skinning, the weights should be normalized if their sum is not 255.

## Bone Indices

The bone_indices field contains indices into the model's bone lookup table, not direct indices to bones. The bone lookup table (contained elsewhere in the M2 file) maps these indices to actual bone indices in the bone array.

The lookup process works like this:
1. Get the bone lookup index from bone_indices
2. Use this index to access the bone lookup table
3. Get the actual bone index from the lookup table
4. Access the bone data using this index

This indirection allows reuse of bones across different parts of the model and optimizes the skinning process.

## Texture Coordinates

Texture coordinates are stored as (u, v) pairs for up to 2 texture units. The first set of coordinates is used for the primary texture (e.g., diffuse map), while the second set can be used for additional textures (e.g., specular maps, normal maps).

- Coordinates are typically in the range [0.0, 1.0]
- U coordinate increases from left to right
- V coordinate increases from top to bottom (DirectX convention)

Some models may use texture coordinates outside the [0.0, 1.0] range for texture tiling or special effects.

## Normals

The normal field contains a 3D vector perpendicular to the surface at the vertex position. It's used for lighting calculations:
- Normal vectors are typically unit-length (magnitude of 1.0)
- Components are in the range [-1.0, 1.0]
- Direction determines how light reflects off the surface

During animation, normals need to be transformed by the same bone matrices that transform the vertices, but only the rotational component is applied (not translation).

## Implementation Notes

### Vertex Skinning

To apply skeletal animation to a vertex:
1. Get the bones affecting this vertex (using bone_indices)
2. Get the weights for each bone (using bone_weights)
3. Transform the vertex position and normal by each bone's matrix
4. Blend the transformed positions and normals based on the weights
5. Use the resulting position and normal for rendering

Example pseudocode:
```
final_position = Vector3(0, 0, 0)
final_normal = Vector3(0, 0, 0)
weight_sum = 0

for i = 0 to 3 do
    if bone_weights[i] > 0 then
        bone_id = bone_lookup_table[bone_indices[i]]
        bone_matrix = bone_matrices[bone_id]
        
        weighted_position = bone_matrix * position * (bone_weights[i] / 255.0)
        weighted_normal = rotation_part(bone_matrix) * normal * (bone_weights[i] / 255.0)
        
        final_position += weighted_position
        final_normal += weighted_normal
        weight_sum += bone_weights[i]
    end if
end for

if weight_sum < 255 then
    // Normalize if weights don't sum to 255
    final_position *= (255.0 / weight_sum)
    final_normal *= (255.0 / weight_sum)
end if

// Normalize the normal to ensure it has unit length
final_normal = normalize(final_normal)
```

### Texture Animation

Texture coordinates can be animated using M2AnimTrack structures:
1. Look up the current animation frame for texture coordinate animation
2. Get the animated texture transformation (translation, rotation, scaling)
3. Apply the transformation to the base texture coordinates from the vertex
4. Use the transformed coordinates for rendering

### Vertex Colors

M2 models typically don't store per-vertex colors directly in the vertex structure. Instead, they use separate systems:
1. Per-geoset vertex colors (stored elsewhere in the file)
2. Color animation tracks (for animated vertex colors)
3. Material colors (applied uniformly to all vertices using a material)

## Usage in Rendering

To render an M2 model:
1. For each geoset (submesh) in the model:
   a. Get the material and texture information for this geoset
   b. Set up the appropriate rendering state (blend mode, textures, etc.)
   c. For each vertex in the geoset:
      i. Apply any active animation to the vertex (skeletal, texture, etc.)
      ii. Pass the transformed vertex data to the rendering pipeline
2. Render all geosets according to the rendering batch information

## Version Differences

- In very early versions of the M2 format (pre-TBC), vertices might have used a slightly different format
- Later versions (Cataclysm+) added support for extended vertex formats in separate data structures
- The core M2Vertex structure has remained relatively stable to maintain backward compatibility 