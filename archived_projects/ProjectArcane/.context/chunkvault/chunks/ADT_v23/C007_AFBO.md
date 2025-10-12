# AFBO - Flight Box Information

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The AFBO (Flight Box) chunk contains information about the flight boundaries for the ADT tile, defining areas where flying mounts or vehicles are allowed or restricted. It uses a series of planes to define these boundaries, with maximum and minimum values stored as shorts instead of floats (unlike the MFBO chunk in v18) for more compact storage.

## Structure

```csharp
public struct AFBO
{
    // Maximum (upper) flight boundaries
    public short[,] maximum;   // 3x3 array of shorts defining the upper boundary
    
    // Minimum (lower) flight boundaries
    public short[,] minimum;   // 3x3 array of shorts defining the lower boundary
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| maximum | short[3,3] | Upper flight boundary planes (3 planes × 3 values) |
| minimum | short[3,3] | Lower flight boundary planes (3 planes × 3 values) |

## Dependencies

No direct dependencies on other chunks.

## Implementation Notes

1. The AFBO chunk contains two 3×3 grids of short values that define the upper and lower boundaries for flying in the ADT tile.

2. The size of this chunk is fixed at 0x48 (72) bytes: 18 shorts × 2 bytes each, plus 8 bytes for chunk header.

3. Unlike the MFBO chunk in v18 which uses floats, AFBO uses shorts for more compact storage, potentially sacrificing some precision.

4. The three planes for each boundary are defined by 3 shorts each, representing coefficients in the plane equation Ax + By + Cz + D = 0.

5. To convert these shorts to actual world coordinates, they likely need to be scaled and translated based on the ADT tile's position.

6. These boundaries are used by the game client to determine where players on flying mounts can and cannot fly, creating invisible barriers.

## Implementation Example

```csharp
public class AfboChunk
{
    // Store boundary planes
    public short[,] Maximum { get; private set; } = new short[3, 3];
    public short[,] Minimum { get; private set; } = new short[3, 3];
    
    // Scaling factor to convert shorts to world units
    private const float SCALE_FACTOR = 0.5f;
    
    public AfboChunk()
    {
        // Initialize with default values (no restrictions)
        for (int plane = 0; plane < 3; plane++)
        {
            for (int coefficient = 0; coefficient < 3; coefficient++)
            {
                Maximum[plane, coefficient] = 0;
                Minimum[plane, coefficient] = 0;
            }
        }
    }
    
    public void Load(BinaryReader reader)
    {
        // Read maximum (upper) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            for (int coefficient = 0; coefficient < 3; coefficient++)
            {
                Maximum[plane, coefficient] = reader.ReadInt16();
            }
        }
        
        // Read minimum (lower) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            for (int coefficient = 0; coefficient < 3; coefficient++)
            {
                Minimum[plane, coefficient] = reader.ReadInt16();
            }
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("AFBO".ToCharArray());
        writer.Write(0x48 - 8); // Chunk size (72 bytes - 8 for header)
        
        // Write maximum (upper) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            for (int coefficient = 0; coefficient < 3; coefficient++)
            {
                writer.Write(Maximum[plane, coefficient]);
            }
        }
        
        // Write minimum (lower) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            for (int coefficient = 0; coefficient < 3; coefficient++)
            {
                writer.Write(Minimum[plane, coefficient]);
            }
        }
    }
    
    // Check if a point is within flight boundaries
    public bool IsPointWithinBoundaries(Vector3 point)
    {
        // Check against maximum (upper) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            float a = Maximum[plane, 0] * SCALE_FACTOR;
            float b = Maximum[plane, 1] * SCALE_FACTOR;
            float c = Maximum[plane, 2] * SCALE_FACTOR;
            
            // Calculate distance to plane
            float distance = a * point.X + b * point.Y + c * point.Z;
            
            // If distance is positive, point is above the plane
            if (distance > 0)
                return false;
        }
        
        // Check against minimum (lower) boundary
        for (int plane = 0; plane < 3; plane++)
        {
            float a = Minimum[plane, 0] * SCALE_FACTOR;
            float b = Minimum[plane, 1] * SCALE_FACTOR;
            float c = Minimum[plane, 2] * SCALE_FACTOR;
            
            // Calculate distance to plane
            float distance = a * point.X + b * point.Y + c * point.Z;
            
            // If distance is negative, point is below the plane
            if (distance < 0)
                return false;
        }
        
        // Point is within all boundaries
        return true;
    }
    
    // Set a simple box-shaped boundary
    public void SetBoxBoundary(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
    {
        // Convert to short values
        short minXShort = (short)(minX / SCALE_FACTOR);
        short minYShort = (short)(minY / SCALE_FACTOR);
        short minZShort = (short)(minZ / SCALE_FACTOR);
        short maxXShort = (short)(maxX / SCALE_FACTOR);
        short maxYShort = (short)(maxY / SCALE_FACTOR);
        short maxZShort = (short)(maxZ / SCALE_FACTOR);
        
        // Set maximum (upper) boundary planes
        // X plane (facing negative X)
        Maximum[0, 0] = 1;
        Maximum[0, 1] = 0;
        Maximum[0, 2] = 0;
        
        // Y plane (facing negative Y)
        Maximum[1, 0] = 0;
        Maximum[1, 1] = 1;
        Maximum[1, 2] = 0;
        
        // Z plane (facing negative Z)
        Maximum[2, 0] = 0;
        Maximum[2, 1] = 0;
        Maximum[2, 2] = 1;
        
        // Set minimum (lower) boundary planes
        // X plane (facing positive X)
        Minimum[0, 0] = -1;
        Minimum[0, 1] = 0;
        Minimum[0, 2] = 0;
        
        // Y plane (facing positive Y)
        Minimum[1, 0] = 0;
        Minimum[1, 1] = -1;
        Minimum[1, 2] = 0;
        
        // Z plane (facing positive Z)
        Minimum[2, 0] = 0;
        Minimum[2, 1] = 0;
        Minimum[2, 2] = -1;
    }
}
```

## Usage Context

The AFBO chunk plays a crucial role in managing player movement in World of Warcraft's 3D environment, especially regarding flying mounts and vehicles. Its main functions include:

1. **Flight Restrictions**: Defines invisible barriers that prevent players from flying into areas that should be inaccessible, such as unfinished zones or areas intended for ground-only access.

2. **Area Boundaries**: Helps create the boundaries of the playable world, preventing players from flying too far away from the intended game space.

3. **Dungeon/Instance Ceilings**: Establishes height limits in outdoor areas that have instance portals or dungeon entrances overhead.

4. **Environmental Storytelling**: Supports world design by guiding players along intended paths or restricting access to areas that should be reached through specific means.

The v23 format's approach to flight boundaries using shorts instead of floats (as in v18's MFBO chunk) represents an optimization for memory usage, potentially at the cost of some precision. This change aligns with the overall theme of v23's experiments with more efficient data storage.

Though never used in any retail release, this experimental approach in the ADT v23 format provides insight into Blizzard's continued refinements to their world data format during the Cataclysm beta development period, balancing the needs of detailed 3D environments with performance considerations. 