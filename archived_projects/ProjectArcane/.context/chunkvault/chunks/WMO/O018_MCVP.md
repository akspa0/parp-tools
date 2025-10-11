# MCVP - WMO Convex Volume Planes

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MCVP chunk defines planes that make up convex volumes for visibility testing within the WMO. These planes are used to create closed volumes that help determine which parts of the model are visible from different viewpoints. This is part of the WMO's approach to occlusion culling and rendering optimization, working alongside the portal system to efficiently render complex structures.

## Structure

```csharp
public struct MCVP
{
    public SMOConvexVolumePlane[] planes; // Array of plane definitions
}

public struct SMOConvexVolumePlane
{
    public float a; // Plane equation coefficient A
    public float b; // Plane equation coefficient B
    public float c; // Plane equation coefficient C
    public float d; // Plane equation coefficient D
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | a | float | Plane equation coefficient A (X component of the normal vector) |
| 0x04 | b | float | Plane equation coefficient B (Y component of the normal vector) |
| 0x08 | c | float | Plane equation coefficient C (Z component of the normal vector) |
| 0x0C | d | float | Plane equation coefficient D (negative distance from origin to plane) |

## Dependencies
- **MOHD**: May contain information about the number of convex volumes.
- **MOVV** and **MOVB**: These chunks also participate in the visibility system and may use these planes.

## Implementation Notes
- Each plane definition is 16 bytes in size (4 floats × 4 bytes).
- The planes are defined using the standard plane equation: Ax + By + Cz + D = 0.
- The values (a, b, c) form the normal vector of the plane, which points outward from the convex volume.
- The d value represents the negative distance from the origin to the plane.
- These planes define the boundaries of convex volumes used for visibility testing.
- A convex volume is formed by the intersection of multiple planes, creating a closed space.
- The normal vectors of the planes should point outward from the volume.
- The number of planes can be calculated from the chunk size: `planeCount = chunkSize / 16`.
- These planes work together with the MOVV and MOVB chunks to implement visibility culling.

## Implementation Example

```csharp
public class MCVPChunk : IWmoChunk
{
    public string ChunkId => "MCVP";
    public List<ConvexVolumePlane> Planes { get; set; } = new List<ConvexVolumePlane>();

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate how many planes are in the chunk
        int count = (int)(size / 16); // Each plane is 16 bytes (4 floats × 4 bytes)
        
        for (int i = 0; i < count; i++)
        {
            ConvexVolumePlane plane = new ConvexVolumePlane
            {
                A = reader.ReadSingle(),
                B = reader.ReadSingle(),
                C = reader.ReadSingle(),
                D = reader.ReadSingle()
            };
            Planes.Add(plane);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate size (16 bytes per plane)
        uint dataSize = (uint)(Planes.Count * 16);
        writer.Write(dataSize);
        
        // Write all planes
        foreach (var plane in Planes)
        {
            writer.Write(plane.A);
            writer.Write(plane.B);
            writer.Write(plane.C);
            writer.Write(plane.D);
        }
    }
    
    public class ConvexVolumePlane
    {
        public float A { get; set; }
        public float B { get; set; }
        public float C { get; set; }
        public float D { get; set; }
        
        // Utility method to determine if a point is in front of the plane
        // (positive side, where the normal points)
        public bool IsPointInFront(float x, float y, float z)
        {
            return A * x + B * y + C * z + D > 0;
        }
        
        // Utility method to normalize the plane equation
        // (makes the normal vector unit length)
        public void Normalize()
        {
            float length = (float)Math.Sqrt(A * A + B * B + C * C);
            if (length > 0)
            {
                A /= length;
                B /= length;
                C /= length;
                D /= length;
            }
        }
    }
}
```

## Validation Requirements
- The number of planes must be a whole number (chunk size must be divisible by 16).
- The normal vector components (a, b, c) should form a unit vector or be normalizable.
- Planes should be oriented correctly to form valid convex volumes.
- The convex volumes defined by these planes should make sense in the context of the WMO structure.
- The planes should correspond to actual boundaries in the model.

## Usage Context
- **Visibility Testing:** The convex volume planes define regions used for determining visibility within the WMO.
- **Occlusion Culling:** By testing whether a viewpoint is within specific convex volumes, the engine can determine which parts of the model need to be rendered.
- **Spatial Organization:** The convex volumes help partition the WMO into logical sections for visibility determination.
- **Rendering Optimization:** This system allows the game engine to efficiently render only the parts of complex WMO models that are potentially visible.
- **Level Design:** These volumes may correspond to rooms, corridors, or other logical spatial divisions within the WMO. 