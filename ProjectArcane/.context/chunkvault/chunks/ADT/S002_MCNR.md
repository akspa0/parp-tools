# S002: MCNR

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCNR (Map Chunk Normal Vectors) subchunk contains normal vectors for each vertex in the MCVT height map. These normals are used for lighting calculations when rendering the terrain.

## Structure
```csharp
struct MCNR
{
    /*0x00*/ int8_t normal_vectors[145][3];  // 145 normal vectors, 3 components each
    /*0x0??*/ uint8_t padding[?];            // Padding to 4-byte boundary
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| normal_vectors | int8_t[145][3] | Normal vectors for each terrain vertex |
| padding | uint8_t[?] | Variable padding to align to 4-byte boundary |

## Normal Vector Format
Each normal vector consists of 3 signed bytes (x, y, z), which represent a compressed normal vector:
- Each component is scaled from the range [-1.0, 1.0] to [-127, 127]
- To convert to a float normal vector, divide each component by 127.0
- The resulting vector should be normalized again for precision

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MCVT (S001) - Height values corresponding to these normal vectors

## Implementation Notes
- There is one normal vector for each height value in MCVT (145 total)
- The normal vectors follow the same layout as the height map (9×9 grid + midpoints)
- The vectors are stored in a compressed format to save space
- When rendering, these normals are used for lighting calculations
- Padding is added to ensure the chunk ends on a 4-byte boundary
- The amount of padding varies depending on the total size

## Implementation Example
```csharp
public class MCNR : IChunk
{
    public const int VERTICES_COUNT = 145;
    
    public Vector3[] Normals { get; set; } = new Vector3[VERTICES_COUNT];
    
    public void Parse(BinaryReader reader)
    {
        for (int i = 0; i < VERTICES_COUNT; i++)
        {
            // Read the compressed normal vector
            sbyte nx = reader.ReadSByte();
            sbyte ny = reader.ReadSByte();
            sbyte nz = reader.ReadSByte();
            
            // Convert to float vector and normalize
            float x = nx / 127.0f;
            float y = ny / 127.0f;
            float z = nz / 127.0f;
            
            // Store the normalized vector
            Normals[i] = new Vector3(x, y, z);
            float length = (float)Math.Sqrt(x * x + y * y + z * z);
            if (length > 0.01f)
            {
                Normals[i] = new Vector3(x / length, y / length, z / length);
            }
        }
        
        // Skip padding bytes (if any)
        int byteCount = VERTICES_COUNT * 3;
        int paddingBytes = (4 - (byteCount % 4)) % 4;
        if (paddingBytes > 0)
        {
            reader.ReadBytes(paddingBytes);
        }
    }
    
    public Vector3 GetNormal(int x, int y)
    {
        // Same indexing as MCVT
        if (x < 0 || x >= 9 || y < 0 || y >= 9)
            throw new ArgumentOutOfRangeException();
            
        return Normals[y * 9 + x];
    }
    
    public Vector3 GetMiddleNormal(int x, int y)
    {
        if (x < 0 || x >= 8 || y < 0 || y >= 8)
            throw new ArgumentOutOfRangeException();
            
        return Normals[81 + y * 8 + x];
    }
}
```

## Lighting Application
The normal vectors determine how light interacts with the terrain:
- Surface facing the light source will be bright
- Surface facing away from light will be darker
- The dot product of the normal and light direction determines the lighting intensity

## Usage Context
The MCNR subchunk is essential for terrain rendering in World of Warcraft. When combined with the MCVT height data, it provides the basis for rendering the 3D terrain with proper lighting. The normal vectors are used in the lighting calculations to determine how light reflects off the terrain, creating the visual effect of hills, valleys, and other terrain features. Without these normal vectors, the terrain would appear flat and unrealistic, even if the height data created the proper 3D shape. 