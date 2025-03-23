# ACNK - Map Chunk Data Container

## Type
ADT v22 Chunk

## Source
Referenced from `ADT_v22.md`

## Description
The ACNK (chunk) structure is the primary container for terrain data in the ADT v22 format. Each ACNK chunk represents a 16×16 yard piece of terrain and contains information about heightmaps, textures, shadows, and object placement. The ADT v22 format contains 256 ACNK chunks (16×16 grid) covering the entire ADT tile.

This chunk serves a similar purpose to the MCNK chunk in v18 but uses a different internal structure and naming convention.

## Structure

```csharp
public struct ACNK
{
    // ACNK Header (present if chunk size > 0x40)
    public int indexX;                   // X position in the ADT grid
    public int indexY;                   // Y position in the ADT grid
    public uint flags;                   // Various flags for the chunk
    public int areaId;                   // Area ID for this chunk (from AreaTable.dbc)
    public ushort lowDetailTextureMapping; // Low detail texture information
    public uint padding1;                // Padding
    public uint padding2;                // Padding
    public uint padding3;                // Padding
    public uint padding4;                // Padding
    public ushort padding5;              // Padding
    public uint padding6;                // Padding
    public uint padding7;                // Padding
    public uint padding8;                // Padding
    public uint padding9;                // Padding
    public uint padding10;               // Padding
    public uint padding11;               // Padding
    public uint padding12;               // Padding
    
    // ACNK may contain subchunks:
    // - ALYR (Texture layer information)
    // - ASHD (Shadow map)
    // - ACDO (Object definitions)
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| indexX | int | X-axis position in the ADT grid (0-15) |
| indexY | int | Y-axis position in the ADT grid (0-15) |
| flags | uint | Flags controlling features and rendering of the chunk |
| areaId | int | Reference to the AreaTable.dbc for gameplay information |
| lowDetailTextureMapping | ushort | Information for low detail texture rendering |

## Dependencies

- AHDR (C001) - Header with grid structure information
- AVTX (C002) - For height data
- ANRM (C003) - For normal vector information
- ATEX (C004) - For texture information referenced by ALYR
- ADOO (C005) - For model information referenced by ACDO

## Subchunks

- ALYR (S001) - Contains texture layer information
- AMAP (S002) - May be embedded in ALYR, contains alpha map data
- ASHD (S003) - Contains shadow map data
- ACDO (S004) - Contains object placement information

## Implementation Notes

1. The ACNK chunk differs from MCNK (v18) by using a different header structure and internal organization.
   
2. Unlike v18 where all MCNK chunks have the same size, ACNK chunks can vary in size.

3. The header is only present if the chunk size is greater than 0x40 bytes.

4. This format separates its vertex data (AVTX) and normal data (ANRM) into separate chunks, unlike v18 which embeds them in MCNK subchunks.

5. The ACNK chunk consolidates both doodad (M2) and object (WMO) references into a single ACDO subchunk, rather than having separate chunks like in v18.

## Implementation Example

```csharp
public class AcnkChunk
{
    public int IndexX { get; set; }
    public int IndexY { get; set; }
    public uint Flags { get; set; }
    public int AreaId { get; set; }
    public ushort LowDetailTextureMapping { get; set; }
    
    public List<AlyrSubChunk> TextureLayers { get; set; } = new List<AlyrSubChunk>();
    public AshdSubChunk ShadowMap { get; set; }
    public AcdoSubChunk ObjectDefinitions { get; set; }

    public AcnkChunk()
    {
    }

    public void Load(BinaryReader reader, long size)
    {
        long startPosition = reader.BaseStream.Position;

        // Only read header if chunk is large enough
        if (size > 0x40)
        {
            IndexX = reader.ReadInt32();
            IndexY = reader.ReadInt32();
            Flags = reader.ReadUInt32();
            AreaId = reader.ReadInt32();
            LowDetailTextureMapping = reader.ReadUInt16();
            
            // Skip padding fields
            reader.BaseStream.Position += 46; // Skip 11.5 DWORDs
        }

        // Parse subchunks
        while (reader.BaseStream.Position < startPosition + size)
        {
            string chunkName = new string(reader.ReadChars(4));
            uint chunkSize = reader.ReadUInt32();
            long chunkStart = reader.BaseStream.Position;
            
            switch (chunkName)
            {
                case "ALYR":
                    var textureLayer = new AlyrSubChunk();
                    textureLayer.Load(reader, chunkSize);
                    TextureLayers.Add(textureLayer);
                    break;
                    
                case "ASHD":
                    ShadowMap = new AshdSubChunk();
                    ShadowMap.Load(reader, chunkSize);
                    break;
                    
                case "ACDO":
                    ObjectDefinitions = new AcdoSubChunk();
                    ObjectDefinitions.Load(reader, chunkSize);
                    break;
                    
                default:
                    // Skip unknown chunk
                    reader.BaseStream.Position += chunkSize;
                    break;
            }
            
            // Ensure we're at the end of the chunk
            reader.BaseStream.Position = chunkStart + chunkSize;
        }
    }
}
```

## Usage Context

The ACNK chunk is a fundamental part of the ADT v22 format, representing portions of terrain within a map tile. Each ACNK contains information about:

1. **Terrain Surface**: The ACNK references vertex and normal data from the global AVTX and ANRM chunks.

2. **Texturing**: Through its ALYR subchunks, it defines which textures are applied to the terrain surface and how they blend together. When combined with AMAP subchunks, these create detailed terrain appearances.

3. **Object Placement**: The ACDO subchunk defines which models and structures appear in this section of terrain.

4. **Environment Features**: Through various flags and properties, it controls lighting, shadows, and other environmental effects.

The v22 format was an experimental design that appeared only in the Cataclysm beta, showing Blizzard's exploration of alternative terrain data organizations. While it was ultimately abandoned in favor of continuing with modified v18 formats, it provides insight into how the developers were thinking about reorganizing terrain data to potentially improve performance or add new features.

A key difference from v18 is the separation of vertex and normal data from the chunk definitions, allowing for potentially more efficient memory management and data access patterns. The unification of M2 and WMO references in ACDO also suggests an attempt to simplify the object placement system. 