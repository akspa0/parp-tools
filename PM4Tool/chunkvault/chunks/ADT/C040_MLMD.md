# MLMD Chunk

## Overview
**Chunk ID**: MLMD  
**Related Expansion**: Legion and later  
**Used in Format**: ADT  
**Implementation Status**: Not implemented

## Description
The MLMD (Map Lod Model Definitions) chunk was introduced in Legion. It contains definitions for object placements in Level of Detail (LOD) terrain rendering. This chunk is similar to the MODF chunk but lacks the bounding box data. It defines WMO (World Map Object) placements for distant terrain rendering. This chunk is found in the split file system, specifically in the obj1 files.

## Chunk Structure

### C++ Structure
```cpp
struct MLMD {
    uint32_t nameId;        // Index into the WMO name list
    uint32_t uniqueId;      // Unique identifier for this object instance
    C3Vector position;      // Position of the object in the world
    C3Vector rotation;      // Rotation of the object (in radians)
    uint16_t flags;         // Various flags for the object
    uint16_t doodadSet;     // Doodad set index for the object
    uint16_t nameSet;       // Name set index
    uint16_t scale;         // Scale factor (1024 = 1.0)
};
```

### C# Structure
```csharp
public struct MLMD
{
    public uint NameId;      // Index into the WMO name list
    public uint UniqueId;    // Unique identifier for this object instance
    public Vector3 Position; // Position of the object in the world
    public Vector3 Rotation; // Rotation of the object (in radians)
    public ushort Flags;     // Various flags for the object
    public ushort DoodadSet; // Doodad set index for the object
    public ushort NameSet;   // Name set index
    public ushort Scale;     // Scale factor (1024 = 1.0)
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| nameId | uint32_t | Index into the WMO name list |
| uniqueId | uint32_t | Unique identifier for this object instance |
| position | C3Vector | Position of the object in the world |
| rotation | C3Vector | Rotation of the object (in radians) |
| flags | uint16_t | Various flags for the object |
| doodadSet | uint16_t | Doodad set index for the object |
| nameSet | uint16_t | Name set index |
| scale | uint16_t | Scale factor (1024 = 1.0) |

## Related Chunks
- MLMX - Contains bounding box and radius data for LOD objects
- MODF - Regular (non-LOD) WMO placement data in ADT files
- Other ML* chunks - Various components of the LOD terrain system

## Notes
- This chunk was introduced in Legion as part of the expanded LOD system for terrain.
- The chunk is found in the obj1 files in the split file system.
- MLMD is similar to MODF but does not include bounding box information (which is instead stored in MLMX).
- The entries in MLMD appear to be sorted based on the radius in the corresponding MLMX entries, from largest to smallest, likely for optimization.
- This sorting might be different from the order in MODF.
- In Legion and later, the scale field works the same way as in MDDF - a value of 1024 means a scale of 1.0.
- These object definitions are used for rendering WMO objects at a distance as part of the LOD system.

## Version History
- **Legion**: Introduced as part of the enhanced LOD terrain system

## References
- [ADT_v18.md documentation](../../docs/ADT_v18.md) 