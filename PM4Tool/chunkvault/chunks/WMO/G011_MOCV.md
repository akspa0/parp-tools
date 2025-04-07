# MOCV - WMO Group Vertex Colors

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MOCV (Map Object Colored Vertices) chunk contains vertex color information used primarily for indoor lighting in WMO groups. Each entry in this chunk corresponds to a vertex in the MOVT chunk and provides RGBA color values that influence how light interacts with the model surfaces. These colors are particularly important for creating ambient lighting effects, shadows, and other visual details in interior spaces.

## Structure

```csharp
public struct MOCV
{
    public VertexColor[] vertexColors; // Array of vertex colors
}

public struct VertexColor
{
    public byte r;   // Red component (0-255)
    public byte g;   // Green component (0-255)
    public byte b;   // Blue component (0-255)
    public byte a;   // Alpha component (0-255)
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | r | byte | Red color component (0-255) |
| 0x01 | g | byte | Green color component (0-255) |
| 0x02 | b | byte | Blue color component (0-255) |
| 0x03 | a | byte | Alpha component (0-255), affects transparency and indoor/outdoor transitions |

## Dependencies
- **MOVT**: Contains the vertex positions that correspond to these colors
- **MOGP**: The flag 0x4 (HasVertexColors) in the MOGP header indicates that this chunk is present

## Implementation Notes
- Each vertex color entry is 4 bytes (0x04) in size, one byte per RGBA component.
- The number of entries in this chunk should match the number of vertices in the MOVT chunk for the same WMO group.
- Vertex colors are primarily used for indoor lighting effects and are blended with texture colors during rendering.
- In some cases, the alpha component is used for special effects:
  - For vertices near portals, alpha values may be reduced to create a smooth transition between indoor and outdoor areas.
  - Alpha values can be adjusted to create shadowed areas or make certain parts of the model more or less transparent.
- The FixColorVertexAlpha function is sometimes used by the client to adjust alpha values of vertex colors for specific effects, particularly in WMOs with fireplaces or blacksmith forges.
- Some WMOs may have a second MOCV chunk (flag 0x1000000 in MOGP), which is used for additional lighting effects.
- When rendering, these vertex colors are typically multiplied with the texture color to produce the final color:
  ```
  finalColor = textureColor * vertexColor
  ```
- For optimal rendering performance, the values should be converted to floating-point normalized range (0.0 to 1.0) during loading.
- Implementations should be aware that some early client versions may handle vertex colors differently than later versions.

## Implementation Example

```csharp
public class MOCVChunk : IWmoGroupChunk
{
    public string ChunkId => "MOCV";
    public List<Color32> VertexColors { get; private set; } = new List<Color32>();

    public void Parse(BinaryReader reader, long size)
    {
        // Each vertex color is 4 bytes (RGBA)
        int colorCount = (int)(size / 4);
        
        for (int i = 0; i < colorCount; i++)
        {
            Color32 color = new Color32
            {
                R = reader.ReadByte(),
                G = reader.ReadByte(),
                B = reader.ReadByte(),
                A = reader.ReadByte()
            };
            
            VertexColors.Add(color);
        }
        
        // Ensure we've read all the data
        if (reader.BaseStream.Position % 4 != 0)
        {
            throw new InvalidDataException("MOCV chunk size is not a multiple of 4 bytes");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var color in VertexColors)
        {
            writer.Write(color.R);
            writer.Write(color.G);
            writer.Write(color.B);
            writer.Write(color.A);
        }
    }
    
    // Helper method to apply the FixColorVertexAlpha function as used by the client
    // This should only be used for WMOs with specific flags
    public void ApplyFixColorVertexAlpha(List<Vector3> vertices, List<PortalInfo> portals)
    {
        for (int i = 0; i < VertexColors.Count; i++)
        {
            if (i >= vertices.Count) break;
            
            var vertex = vertices[i];
            var color = VertexColors[i];
            
            float minDistSq = float.MaxValue;
            float sumOpacities = 0.0f;
            
            // Check distance to each portal
            foreach (var portal in portals)
            {
                // Calculate distance from vertex to portal center
                Vector3 portalCenter = portal.Center;
                float distSq = (vertex - portalCenter).LengthSquared();
                
                if (distSq < minDistSq)
                {
                    minDistSq = distSq;
                }
                
                // Accumulate opacity based on distance
                float dist = (float)Math.Sqrt(distSq);
                if (dist <= portal.Radius)
                {
                    // Portal affects this vertex
                    float opacity = Math.Max(0, 1.0f - dist / portal.Radius);
                    sumOpacities += opacity;
                }
            }
            
            // Adjust alpha value based on distance to portals
            if (sumOpacities > 0)
            {
                // Limit maximum impact of portal proximity
                sumOpacities = Math.Min(1.0f, sumOpacities);
                
                // Reduce alpha for vertices close to portals
                color.A = (byte)(color.A * (1.0f - sumOpacities * 0.5f));
                VertexColors[i] = color;
            }
        }
    }
    
    public struct Color32
    {
        public byte R;
        public byte G;
        public byte B;
        public byte A;
        
        // Helper conversion to normalized float values
        public Vector4 ToVector4()
        {
            return new Vector4(R / 255.0f, G / 255.0f, B / 255.0f, A / 255.0f);
        }
    }
    
    public struct PortalInfo
    {
        public Vector3 Center;
        public float Radius;
    }
}
```

## Usage Context
- Vertex colors are primarily used for indoor lighting in WMO models, providing ambient light and shadow information.
- The MOCV chunk is essential for creating atmospheric interior spaces, such as dimly lit taverns, shadowed corners in caves, or brightly lit hallways.
- These colors affect how textures appear in different parts of the model, allowing for subtle lighting variations without requiring additional light sources.
- The alpha component can be used to create smooth transitions between indoor and outdoor areas when a player passes through a portal.
- In some special WMOs like blacksmith buildings or those with fireplaces, adjusted vertex colors can enhance the visual effect of glowing embers or fire.
- Rendering engines typically blend these vertex colors with texture colors during the fragment shader stage, influencing the final appearance of the model surfaces. 