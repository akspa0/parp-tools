# MOTV - WMO Group Texture Coordinates

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOTV chunk contains texture coordinates (UV coordinates) for each vertex in the WMO group. These coordinates determine how textures are mapped onto the 3D surfaces of the model. Each texture coordinate corresponds to a vertex in the MOVT chunk and should be in the same order. The texture coordinates define how 2D textures wrap around the 3D model surfaces, allowing for proper texturing of the model.

## Structure

```csharp
public struct MOTV
{
    public Vector2[] textureCoordinates; // Array of texture coordinates
}

public struct Vector2
{
    public float u;
    public float v;
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | textureCoordinates | Vector2[] | Array of 2D vectors representing the texture coordinates for each vertex. Each coordinate takes 8 bytes (2 floats, 4 bytes each). The number of texture coordinates can be calculated as chunk size / 8. Coordinates are typically in the [0.0, 1.0] range but can go outside this range for repeating textures. |

## Dependencies
- **MOVT**: The texture coordinates in this chunk should correspond to the vertices in the MOVT chunk in the same order.
- **MOVI**: The indices in the MOVI chunk reference vertices and their corresponding texture coordinates.

## Implementation Notes
- The size of the chunk should be a multiple of 8 bytes (2 floats × 4 bytes each).
- Texture coordinates typically range from 0.0 to 1.0, where (0,0) represents the top-left corner of the texture and (1,1) represents the bottom-right corner.
- Values outside the [0,1] range are valid and indicate texture repetition (for tiling textures).
- The WMO may have multiple MOTV chunks if the model uses multiple texture coordinate sets (like for multi-texturing or lightmapping). The presence of multiple MOTV chunks is indicated by flags in the MOGP header.
- Not present in "antiportal" WMO groups as they don't need texturing.
- The client can load up to 3 MOTV chunks, which will be stored in an array. This allows for multiple texture coordinate sets per vertex for advanced rendering effects.
- There is a bug in some client versions where the count is overwritten if there are more than 3 MOTV chunks, potentially causing memory corruption.

## Implementation Example

```csharp
public class MOTVChunk : IWmoGroupChunk
{
    public string ChunkId => "MOTV";
    public List<Vector2> TextureCoordinates { get; private set; } = new List<Vector2>();
    
    // Used to track which MOTV set this is (0, 1, or 2) when multiple sets exist
    public int TextureCoordinateSetIndex { get; set; } = 0;

    public void Parse(BinaryReader reader, long size)
    {
        // Calculate the number of texture coordinates
        int coordCount = (int)(size / 8);
        
        // Read all texture coordinates
        for (int i = 0; i < coordCount; i++)
        {
            Vector2 texCoord = new Vector2
            {
                u = reader.ReadSingle(),
                v = reader.ReadSingle()
            };
            
            TextureCoordinates.Add(texCoord);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 8 != 0)
        {
            throw new InvalidDataException("MOTV chunk size is not a multiple of 8 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var texCoord in TextureCoordinates)
        {
            writer.Write(texCoord.u);
            writer.Write(texCoord.v);
        }
    }
}
```

## Usage Context
- The texture coordinates are used to map 2D textures onto the 3D surfaces of the WMO model.
- Multiple texture coordinate sets may be used for multi-texturing effects, such as applying a base texture, detail texture, and lightmap to the same surface.
- When rendering, these coordinates determine which part of the texture is displayed on each part of the 3D model.
- Proper texture mapping is essential for creating realistic and detailed world objects.
- Texture coordinates can be manipulated for special effects like scrolling textures (for water surfaces) or texture animation. 