# MBNV (Map Blend Normal Vectors)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MBNV chunk was introduced in Mists of Pandaria (MoP) and contains vertex data for blend meshes. It stores positions, normals, texture coordinates, and color information for vertices used in blend meshes that create smooth transitions between terrain and WMO objects.

## Structure

```csharp
public struct MBNV
{
    public Vector3 Position;         // 3D position of the vertex
    public Vector3 Normal;           // Normal vector for the vertex
    public Vector2 TextureCoords;    // Texture coordinates (UV)
    public Color Color1;             // Primary color
    public Color Color2;             // Secondary color (used in some vertex formats)
    public Color Color3;             // Tertiary color (used in some vertex formats)
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| Position | Vector3 | 3D position of the vertex in world space |
| Normal | Vector3 | Surface normal at this vertex for lighting calculations |
| TextureCoords | Vector2 | 2D texture coordinates (UV) for mapping textures |
| Color1 | Color | Primary color, used in all vertex formats with color |
| Color2 | Color | Secondary color, used in PNC2 and PNC2T vertex formats |
| Color3 | Color | Tertiary color, used in PNC2T vertex format |

## Color Usage by Vertex Format

| Vertex Format | Color1 | Color2 | Color3 |
|---------------|--------|--------|--------|
| PN (Position, Normal) | None | None | None |
| PNC (Position, Normal, Color) | Used | None | None |
| PNC2 (Position, Normal, 2 Colors) | Used | Used | None |
| PNC2T (Position, Normal, 2 Colors, Texture) | Used | None | Used |

## Dependencies

- **MBMH (C019)** - References MBNV entries for blend mesh vertices
- **MBMI (C022)** - Contains indices into the MBNV vertex array
- **MCBB (S010)** - Contains blend batches that reference vertices from MBNV

## Implementation Notes

- The vertex format determines which color channels are used (as shown in the table above)
- Vertex format is determined by the parent MCNK chunk flags
- MBNV data is found in both the root ADT file and LOD files
- Vertices in this chunk are indexed by the MBMI chunk to create triangles
- The MBMH chunk contains start offset and count values for accessing specific ranges of MBNV entries

## Implementation Example

```csharp
public class BlendMeshVertexProcessor
{
    public enum VertexFormat
    {
        PN,       // Position, Normal
        PNC,      // Position, Normal, Color
        PNC2,     // Position, Normal, 2 Colors
        PNC2T     // Position, Normal, 2 Colors, Texture
    }
    
    public List<BlendMeshVertex> ProcessVertices(List<MBNV> rawVertices, VertexFormat format)
    {
        var processedVertices = new List<BlendMeshVertex>();
        
        foreach (var vertex in rawVertices)
        {
            var processed = new BlendMeshVertex
            {
                Position = vertex.Position,
                Normal = vertex.Normal,
                TextureCoords = vertex.TextureCoords
            };
            
            // Apply colors based on vertex format
            switch (format)
            {
                case VertexFormat.PNC:
                    processed.Colors = new[] { vertex.Color1 };
                    break;
                    
                case VertexFormat.PNC2:
                    processed.Colors = new[] { vertex.Color1, vertex.Color2 };
                    break;
                    
                case VertexFormat.PNC2T:
                    processed.Colors = new[] { vertex.Color1, vertex.Color3 };
                    break;
                    
                case VertexFormat.PN:
                default:
                    processed.Colors = Array.Empty<Color>();
                    break;
            }
            
            processedVertices.Add(processed);
        }
        
        return processedVertices;
    }
}

public struct BlendMeshVertex
{
    public Vector3 Position;
    public Vector3 Normal;
    public Vector2 TextureCoords;
    public Color[] Colors;  // Array of colors, length depends on vertex format
}
```

## Usage Context

The MBNV chunk is a critical component of the blend mesh system introduced in Mists of Pandaria to improve the visual integration of WMO objects with terrain. These blend meshes create seamless transitions where structures meet the surrounding landscape.

The vertex data in this chunk includes not only positions and normals for geometry but also texture coordinates and color information used for blending. The different vertex formats (PN, PNC, PNC2, PNC2T) allow for varying levels of detail and visual complexity in different areas of the game world.

When combined with the indexing information in MBMI and the header data in MBMH, these vertices form complete triangle meshes that connect WMO geometry to the terrain with proper visual blending. This dramatically improves the visual quality of the game world by eliminating harsh, unnatural transitions between objects and terrain. 