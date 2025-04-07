# S011: MCBB

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCBB (Map Chunk Bounding Box) subchunk contains an axis-aligned bounding box for an MCNK chunk. It was introduced in Mists of Pandaria to provide more accurate spatial information for culling and collision detection.

## Structure
```csharp
struct MCBB
{
    /*0x00*/ CAaBox bounding_box;  // Axis-aligned bounding box (24 bytes)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| bounding_box | CAaBox | Axis-aligned bounding box for the chunk |

## CAaBox Structure
```csharp
struct CAaBox
{
    /*0x00*/ C3Vector min;  // Minimum corner coordinates
    /*0x0C*/ C3Vector max;  // Maximum corner coordinates
};
```

## C3Vector Structure
```csharp
struct C3Vector
{
    /*0x00*/ float x;  // X coordinate
    /*0x04*/ float y;  // Y coordinate
    /*0x08*/ float z;  // Z coordinate
};
```

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCNK.mcbb - Offset to this subchunk
- MCVT (S001) - Heights used to calculate the bounding box

## Presence Determination
This subchunk is only present when:
- MCNK.mcbb offset is non-zero
- The ADT is from Mists of Pandaria or later (version >= 15)

## Implementation Notes
- The bounding box encloses all terrain vertices in the MCNK chunk
- It's used for more efficient culling during rendering
- Also used for collision detection and physics calculations
- Min corner has the lowest X, Y, and Z coordinates
- Max corner has the highest X, Y, and Z coordinates
- Coordinates are in absolute world space, not relative to the chunk
- Introduced in Mists of Pandaria to optimize rendering and collision

## Implementation Example
```csharp
public class MCBB : IChunk
{
    public CAaBox BoundingBox { get; set; }
    
    public void Parse(BinaryReader reader)
    {
        // Read minimum corner
        float minX = reader.ReadSingle();
        float minY = reader.ReadSingle();
        float minZ = reader.ReadSingle();
        
        // Read maximum corner
        float maxX = reader.ReadSingle();
        float maxY = reader.ReadSingle();
        float maxZ = reader.ReadSingle();
        
        BoundingBox = new CAaBox(
            new C3Vector(minX, minY, minZ),
            new C3Vector(maxX, maxY, maxZ)
        );
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write minimum corner
        writer.Write(BoundingBox.Min.X);
        writer.Write(BoundingBox.Min.Y);
        writer.Write(BoundingBox.Min.Z);
        
        // Write maximum corner
        writer.Write(BoundingBox.Max.X);
        writer.Write(BoundingBox.Max.Y);
        writer.Write(BoundingBox.Max.Z);
    }
    
    // Helper method to check if a point is inside the bounding box
    public bool Contains(C3Vector point)
    {
        return point.X >= BoundingBox.Min.X && point.X <= BoundingBox.Max.X &&
               point.Y >= BoundingBox.Min.Y && point.Y <= BoundingBox.Max.Y &&
               point.Z >= BoundingBox.Min.Z && point.Z <= BoundingBox.Max.Z;
    }
    
    // Helper method to check if this bounding box intersects with another
    public bool Intersects(CAaBox other)
    {
        return BoundingBox.Min.X <= other.Max.X && BoundingBox.Max.X >= other.Min.X &&
               BoundingBox.Min.Y <= other.Max.Y && BoundingBox.Max.Y >= other.Min.Y &&
               BoundingBox.Min.Z <= other.Max.Z && BoundingBox.Max.Z >= other.Min.Z;
    }
    
    // Get the center point of the bounding box
    public C3Vector GetCenter()
    {
        return new C3Vector(
            (BoundingBox.Min.X + BoundingBox.Max.X) * 0.5f,
            (BoundingBox.Min.Y + BoundingBox.Max.Y) * 0.5f,
            (BoundingBox.Min.Z + BoundingBox.Max.Z) * 0.5f
        );
    }
    
    // Get the dimensions of the bounding box
    public C3Vector GetDimensions()
    {
        return new C3Vector(
            BoundingBox.Max.X - BoundingBox.Min.X,
            BoundingBox.Max.Y - BoundingBox.Min.Y,
            BoundingBox.Max.Z - BoundingBox.Min.Z
        );
    }
}
```

## Bounding Box Calculation
The bounding box is calculated from the terrain vertices:
- Min corner is the minimum X, Y, Z coordinates of any vertex in the chunk
- Max corner is the maximum X, Y, Z coordinates of any vertex in the chunk
- All terrain features within the chunk should be contained in this box
- May include some additional margin for safety

## Version Information
- MCBB was introduced in Mists of Pandaria (version 15)
- It's present in all chunks in maps from MoP and later
- The presence is determined by a non-zero MCNK.mcbb offset

## Usage Applications
The MCBB has several important applications:
- **View Frustum Culling**: Quickly determine if a chunk is visible on screen
- **Occlusion Culling**: Determine if a chunk is hidden behind others
- **Collision Detection**: Efficiently check for potential collisions
- **Spatial Queries**: Find chunks that contain or intersect with points or volumes
- **Distance Calculations**: Compute distance to chunks for level of detail decisions

## Usage Context
The MCBB subchunk improves rendering performance and physics calculations by providing a tight bounding volume for each terrain chunk. This allows the game engine to quickly determine whether a chunk needs to be rendered or included in physics calculations. Before the introduction of this subchunk, bounding boxes had to be calculated dynamically, which was less efficient. By pre-calculating and storing this data, the game can perform spatial operations more efficiently, especially on complex terrain with varying heights. 