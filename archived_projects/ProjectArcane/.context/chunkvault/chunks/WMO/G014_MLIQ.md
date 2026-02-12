# MLIQ - WMO Group Liquid

## Type
WMO Group Chunk

## Source
WMO.md

## Description
The MLIQ (Map LIQuid) chunk contains information about liquid surfaces within a WMO group, such as water, lava, slime, or other fluid types. It defines the geometry and properties of liquid areas, including height maps, vertex positions, and material types. This chunk is essential for rendering water bodies, lakes, pools, and other liquid features within World of Warcraft's interior spaces and structures.

## Structure

```csharp
public struct MLIQ
{
    public MLIQHeader header;             // Liquid header information
    public float[] heightMap;             // Height values for each point in the liquid grid
    public byte[] renderFlags;            // Render flags for each liquid vertex
    public Vector3[] vertices;            // Positions of liquid surface vertices
    public SMOLTile[] tileFlags;          // Liquid tile flags
}

public struct MLIQHeader
{
    public uint liquidType;               // Type of liquid (see LiquidType table)
    public uint liquidObject;             // Used for liquid object assignment and behavior
    public float xBase;                   // X coordinate of the bottom-left corner
    public float yBase;                   // Y coordinate of the bottom-left corner
    public float zBase;                   // Z coordinate (height) of the lowest point
    public float width;                   // X-axis size of the liquid surface
    public float height;                  // Y-axis size of the liquid surface
    public ushort xVertices;              // Number of vertices along X-axis
    public ushort yVertices;              // Number of vertices along Y-axis
    public uint xTiles;                   // Number of liquid tiles along X-axis
    public uint yTiles;                   // Number of liquid tiles along Y-axis
    public Vector3 basePosition;          // Base position of the liquid surface
}

public struct SMOLTile
{
    public byte flags;                    // Flags for each liquid tile (see TileFlags table)
}
```

## Properties

### MLIQHeader Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | liquidType | uint | Type of liquid (see LiquidType table) |
| 0x04 | liquidObject | uint | Used for liquid object assignment and behavior |
| 0x08 | xBase | float | X coordinate of the bottom-left corner |
| 0x0C | yBase | float | Y coordinate of the bottom-left corner |
| 0x10 | zBase | float | Z coordinate (height) of the lowest point |
| 0x14 | width | float | X-axis size of the liquid surface |
| 0x18 | height | float | Y-axis size of the liquid surface |
| 0x1C | xVertices | ushort | Number of vertices along X-axis |
| 0x1E | yVertices | ushort | Number of vertices along Y-axis |
| 0x20 | xTiles | uint | Number of liquid tiles along X-axis |
| 0x24 | yTiles | uint | Number of liquid tiles along Y-axis |
| 0x28 | basePosition | Vector3 | Base position of the liquid surface (12 bytes) |
| 0x34 | --- | --- | End of header, start of height map |

### LiquidType (Offset 0x00)

| Value | Description |
|-------|-------------|
| 0 | Water |
| 1 | Ocean |
| 2 | Magma/Lava |
| 3 | Slime |
| 4 | Naxxramas (Green Slime) |
| 5 | Underwater |
| ... | Other liquid types defined in client DBC files |

### TileFlags (SMOLTile)

| Bit | Name | Description |
|-----|------|-------------|
| 0x01 | LIQUID_TILE_VISIBLE | Tile is visible |
| 0x02 | LIQUID_TILE_HAS_GEOMETRY | Tile has liquid geometry |
| 0x04 | LIQUID_TILE_OCEAN | Tile is ocean water |
| 0x08 | LIQUID_TILE_MAGMA | Tile is magma/lava |

## Dependencies
- **MOGP**: The flag 0x1000 (HasLiquid) in the MOGP header indicates that this chunk is present

## Implementation Notes
- The MLIQ chunk starts with a header section containing basic information about the liquid area.
- After the header comes the height map, which is a 2D grid of float values. The number of height values is xVertices × yVertices.
- Each height value represents the Z-coordinate (height) of the liquid surface at that point in the grid.
- The render flags array contains one byte per vertex in the grid, potentially indicating special rendering properties.
- The vertices array contains the actual 3D positions of each point in the liquid grid.
- The tile flags array contains one byte per tile, with each tile covering a 1×1 section of the grid. The number of tiles is xTiles × yTiles.
- Liquid surfaces are typically rendered as a grid of triangles, with two triangles per tile.
- The liquid type determines the visual appearance, sound effects, and physical properties of the liquid (e.g., movement speed, damage).
- Some liquid types have special visual effects, such as waves, reflection, or particle effects.
- The liquidObject field may be used to link to database entries for special liquid behavior or effects.
- When rendering liquids, implementations should consider effects like:
  - Transparency and refraction for water
  - Reflection for calm water surfaces
  - Ripple and wave effects
  - Particle effects for lava or slime
  - Special shader effects based on the liquid type
- The basePosition combined with width and height defines the bounding box of the liquid area.
- For collision detection with liquid surfaces, use the height map to determine the exact height at any X,Y position.

## Implementation Example

```csharp
public class MLIQChunk : IWmoGroupChunk
{
    public string ChunkId => "MLIQ";
    
    public uint LiquidType { get; private set; }
    public uint LiquidObject { get; private set; }
    public float XBase { get; private set; }
    public float YBase { get; private set; }
    public float ZBase { get; private set; }
    public float Width { get; private set; }
    public float Height { get; private set; }
    public ushort XVertices { get; private set; }
    public ushort YVertices { get; private set; }
    public uint XTiles { get; private set; }
    public uint YTiles { get; private set; }
    public Vector3 BasePosition { get; private set; }
    
    public float[] HeightMap { get; private set; }
    public byte[] RenderFlags { get; private set; }
    public Vector3[] Vertices { get; private set; }
    public byte[] TileFlags { get; private set; }
    
    public void Parse(BinaryReader reader, long size)
    {
        long startPosition = reader.BaseStream.Position;
        
        // Read header
        LiquidType = reader.ReadUInt32();
        LiquidObject = reader.ReadUInt32();
        XBase = reader.ReadSingle();
        YBase = reader.ReadSingle();
        ZBase = reader.ReadSingle();
        Width = reader.ReadSingle();
        Height = reader.ReadSingle();
        XVertices = reader.ReadUInt16();
        YVertices = reader.ReadUInt16();
        XTiles = reader.ReadUInt32();
        YTiles = reader.ReadUInt32();
        
        // Read base position
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        BasePosition = new Vector3(x, y, z);
        
        // Calculate sizes
        int vertexCount = XVertices * YVertices;
        int tileCount = (int)(XTiles * YTiles);
        
        // Read height map
        HeightMap = new float[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            HeightMap[i] = reader.ReadSingle();
        }
        
        // Read render flags
        RenderFlags = new byte[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            RenderFlags[i] = reader.ReadByte();
        }
        
        // Generate vertices based on the grid
        Vertices = new Vector3[vertexCount];
        for (int y = 0; y < YVertices; y++)
        {
            for (int x = 0; x < XVertices; x++)
            {
                int index = y * XVertices + x;
                float xPos = XBase + (x * Width / (XVertices - 1));
                float yPos = YBase + (y * Height / (YVertices - 1));
                float zPos = ZBase + HeightMap[index];
                
                Vertices[index] = new Vector3(xPos, yPos, zPos);
            }
        }
        
        // Read tile flags
        TileFlags = new byte[tileCount];
        for (int i = 0; i < tileCount; i++)
        {
            TileFlags[i] = reader.ReadByte();
        }
        
        // Ensure we've read all the data
        long bytesRead = reader.BaseStream.Position - startPosition;
        if (bytesRead != size)
        {
            throw new InvalidDataException($"MLIQ chunk size mismatch. Expected {size} bytes, read {bytesRead} bytes.");
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write header
        writer.Write(LiquidType);
        writer.Write(LiquidObject);
        writer.Write(XBase);
        writer.Write(YBase);
        writer.Write(ZBase);
        writer.Write(Width);
        writer.Write(Height);
        writer.Write(XVertices);
        writer.Write(YVertices);
        writer.Write(XTiles);
        writer.Write(YTiles);
        
        // Write base position
        writer.Write(BasePosition.X);
        writer.Write(BasePosition.Y);
        writer.Write(BasePosition.Z);
        
        // Write height map
        foreach (float height in HeightMap)
        {
            writer.Write(height);
        }
        
        // Write render flags
        writer.Write(RenderFlags);
        
        // Write tile flags
        writer.Write(TileFlags);
    }
    
    // Helper method to check if a point is within the liquid area
    public bool IsPointOverLiquid(float x, float y)
    {
        // Check if the point is within the bounding box of the liquid
        if (x < XBase || x > XBase + Width || y < YBase || y > YBase + Height)
            return false;
            
        // Convert world position to tile indices
        int tileX = (int)((x - XBase) / Width * XTiles);
        int tileY = (int)((y - YBase) / Height * YTiles);
        
        // Clamp to ensure we're within bounds
        tileX = Math.Max(0, Math.Min(tileX, (int)XTiles - 1));
        tileY = Math.Max(0, Math.Min(tileY, (int)YTiles - 1));
        
        // Get tile index
        int tileIndex = tileY * (int)XTiles + tileX;
        
        // Check if the tile is visible and has geometry
        return (TileFlags[tileIndex] & 0x03) == 0x03;
    }
    
    // Helper method to get the liquid height at a specific position
    public float GetLiquidHeight(float x, float y)
    {
        // Check if the point is within the liquid area
        if (!IsPointOverLiquid(x, y))
            return float.MinValue;
            
        // Convert world position to vertex grid coordinates
        float gridX = (x - XBase) / Width * (XVertices - 1);
        float gridY = (y - YBase) / Height * (YVertices - 1);
        
        // Get the four surrounding vertices
        int x1 = (int)gridX;
        int y1 = (int)gridY;
        int x2 = Math.Min(x1 + 1, XVertices - 1);
        int y2 = Math.Min(y1 + 1, YVertices - 1);
        
        // Calculate interpolation factors
        float fracX = gridX - x1;
        float fracY = gridY - y1;
        
        // Get heights of the four corners
        float h00 = HeightMap[y1 * XVertices + x1];
        float h10 = HeightMap[y1 * XVertices + x2];
        float h01 = HeightMap[y2 * XVertices + x1];
        float h11 = HeightMap[y2 * XVertices + x2];
        
        // Bilinear interpolation
        float h0 = h00 * (1 - fracX) + h10 * fracX;
        float h1 = h01 * (1 - fracX) + h11 * fracX;
        float height = h0 * (1 - fracY) + h1 * fracY;
        
        return ZBase + height;
    }
    
    // Helper method to get the liquid type name
    public string GetLiquidTypeName()
    {
        switch (LiquidType)
        {
            case 0: return "Water";
            case 1: return "Ocean";
            case 2: return "Magma/Lava";
            case 3: return "Slime";
            case 4: return "Naxxramas (Green Slime)";
            case 5: return "Underwater";
            default: return $"Unknown ({LiquidType})";
        }
    }
}
```

## Usage Context
- Liquid surfaces add realism and visual appeal to interior spaces like castles, caves, and dungeons.
- Different liquid types provide environmental storytelling and gameplay elements:
  - Water areas may allow swimming or provide a refreshing effect
  - Lava areas often cause damage and create visual hazards
  - Slime areas might apply debuffs or slow movement
- The MLIQ chunk enables accurate collision detection with liquid surfaces, allowing characters to:
  - Float on water surfaces
  - Swim within water volumes
  - Take damage from harmful liquids like lava
- Rendering systems use this data to apply appropriate visual effects for each liquid type:
  - Water reflection and refraction
  - Lava glow and particle effects
  - Slime bubbling and opacity
- Sound systems use the liquid type to trigger appropriate ambient sounds and footstep effects
- The height map allows for realistic wave animation and ripple effects when objects interact with the liquid
- Game mechanics can use liquid areas for gameplay elements like:
  - Quest objectives requiring diving or swimming
  - Environmental hazards that players must navigate around
  - Special abilities or items that interact with specific liquid types 