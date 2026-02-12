# ACNK - Map Chunk Data Container

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The ACNK (Chunk) chunk serves as a container for terrain chunk data in the ADT v23 format. Each ACNK chunk represents a specific section of terrain within the ADT tile grid, typically a 16×16 yard area. The ACNK chunk contains a header followed by various subchunks that define texture layers, shadow maps, and object placements for that specific terrain section.

## Structure

```csharp
public struct ACNK
{
    // Header (if present - non-root ADTs may have zero-sized headers)
    public ACNKHeader header;    // Only present if chunk size > 0x40

    // Subchunks (variable number and order)
    // Can include ALYR, ASHD, ACDO, and potentially any MCNK subchunk from v18
}

public struct ACNKHeader
{
    public int indexX;               // X position in the chunk grid (0-15)
    public int indexY;               // Y position in the chunk grid (0-15)
    public uint flags;               // Various flags defining chunk properties
    public int areaId;               // Zone ID for this chunk
    public ushort holes_low_res;     // Low resolution holes mask
    public uint[] lowDetailTexturingMap; // 4 DWORDs containing low detail texture mapping
    public ushort mcdd;              // Unknown
    public byte[] padding1;          // 6 bytes of padding
    public ushort unknown;           // Same as MCNK's unknown at 0x03E
    public byte[] highResHoles;      // 8 bytes of high resolution holes data (if flags & 0x10000)
    public byte[] padding2;          // 6 bytes of padding
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| header.indexX | int | X position of this chunk in the grid (0-15) |
| header.indexY | int | Y position of this chunk in the grid (0-15) |
| header.flags | uint | Bit field defining various chunk properties |
| header.areaId | int | Zone ID for this chunk (defines minimap color, allowed activities, etc.) |
| header.holes_low_res | ushort | 16-bit mask for low-resolution terrain holes (1 bit per quad) |
| header.lowDetailTexturingMap | uint[4] | Defines the low-detail texture mapping for distant viewing |
| header.mcdd | ushort | Unknown field |
| header.unknown | ushort | Same as MCNK's unknown value at offset 0x03E |
| header.highResHoles | byte[8] | 64-bit mask for high-resolution terrain holes (only if flags & 0x10000) |

## ACNK Flags

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x1 | HasMCCV | Has vertex shading (MCCV) data |
| 0x2 | UnkFlag0x2 | Unknown |
| 0x4 | HasMCLQ | Has liquid layer (MCLQ) data |
| 0x8 | HasMCSH | Has shadow map (MCSH) data |
| 0x10 | Impassable | Marks terrain as impassable |
| 0x20 | LakeOrOcean | Marks chunk as lake or ocean |
| 0x40 | UnkFlag0x40 | Unknown |
| 0x80 | UnkFlag0x80 | Unknown |
| 0x10000 | HasHighResHoles | Uses high-resolution hole mask |

## Dependencies

- AHDR (C001) - Defines the grid dimensions this chunk exists within
- AVTX (C002) - Contains vertex height data referenced by this chunk
- ANRM (C003) - Contains normal data referenced by this chunk
- ATEX (C004) - Contains texture filenames referenced by ALYR subchunks
- ADOO (C005) - Contains model filenames referenced by ACDO subchunks
- ACVT (C008) - May contain vertex shading information for this chunk

## Implementation Notes

1. The ACNK chunk serves as a container for various subchunks that define a specific section of terrain.

2. The header may be zero-sized for non-root ADTs due to the parser behavior in the client, but typically contains 0x40 bytes of data.

3. The actual size of the ACNK header may vary depending on the flags. If the HasHighResHoles flag (0x10000) is set, the high-resolution holes data is included in the header.

4. The ACNK chunk can contain subchunks in any order, including ALYR (texture layers), ASHD (shadow map), and ACDO (object definitions).

5. Due to parser behavior in the client, any MCNK subchunk from v18 can also appear within ACNK, though this is likely not intended.

6. Unlike v18 where each MCNK contains its own height data, in v23 the heights are stored centrally in AVTX, and each ACNK references a portion of that data based on its grid position.

7. The holes_low_res and highResHoles fields define areas of the terrain that have been "punched through," creating areas where players can fall through the terrain (used for caves and similar features).

## Implementation Example

```csharp
public class AcnkChunk
{
    // Header data
    public int IndexX { get; set; }
    public int IndexY { get; set; }
    public uint Flags { get; set; }
    public int AreaId { get; set; }
    public ushort HolesLowRes { get; set; }
    public uint[] LowDetailTexturingMap { get; set; } = new uint[4];
    public ushort Mcdd { get; set; }
    public ushort Unknown { get; set; }
    public byte[] HighResHoles { get; set; } = new byte[8];
    
    // Subchunks collections
    public List<AlyrSubchunk> TextureLayers { get; set; } = new List<AlyrSubchunk>();
    public AshdSubchunk ShadowMap { get; set; }
    public List<AcdoSubchunk> Objects { get; set; } = new List<AcdoSubchunk>();
    
    // Other potential subchunks from v18
    public Dictionary<string, byte[]> OtherSubchunks { get; set; } = new Dictionary<string, byte[]>();
    
    public AcnkChunk(int indexX, int indexY)
    {
        IndexX = indexX;
        IndexY = indexY;
        Flags = 0;
        AreaId = 0;
        HolesLowRes = 0;
        Mcdd = 0;
        Unknown = 0;
    }
    
    public void Load(BinaryReader reader, long chunkSize)
    {
        long startPosition = reader.BaseStream.Position;
        
        // Check if the chunk has a header
        if (chunkSize > 0x40)
        {
            // Read header
            IndexX = reader.ReadInt32();
            IndexY = reader.ReadInt32();
            Flags = reader.ReadUInt32();
            AreaId = reader.ReadInt32();
            HolesLowRes = reader.ReadUInt16();
            
            // Read low detail texturing map
            for (int i = 0; i < 4; i++)
                LowDetailTexturingMap[i] = reader.ReadUInt32();
                
            Mcdd = reader.ReadUInt16();
            
            // Skip padding
            reader.BaseStream.Position += 6;
            
            Unknown = reader.ReadUInt16();
            
            // Read high-res holes if present
            if ((Flags & 0x10000) != 0)
            {
                HighResHoles = reader.ReadBytes(8);
            }
            else
            {
                HighResHoles = new byte[8]; // Zero-filled
                reader.BaseStream.Position += 8;
            }
            
            // Skip padding
            reader.BaseStream.Position += 6;
        }
        
        // Now read subchunks until we reach the end of the ACNK chunk
        long endPosition = startPosition + chunkSize;
        while (reader.BaseStream.Position < endPosition)
        {
            // Read subchunk header
            char[] magic = reader.ReadChars(4);
            string subchunkName = new string(magic);
            uint subchunkSize = reader.ReadUInt32();
            
            long subchunkStart = reader.BaseStream.Position;
            
            // Process subchunk based on type
            switch (subchunkName)
            {
                case "ALYR":
                    // Load texture layer
                    var layer = new AlyrSubchunk();
                    layer.Load(reader, subchunkSize);
                    TextureLayers.Add(layer);
                    break;
                    
                case "ASHD":
                    // Load shadow map
                    ShadowMap = new AshdSubchunk();
                    ShadowMap.Load(reader, subchunkSize);
                    break;
                    
                case "ACDO":
                    // Load object definition
                    var obj = new AcdoSubchunk();
                    obj.Load(reader, subchunkSize);
                    Objects.Add(obj);
                    break;
                    
                default:
                    // Store unknown subchunk
                    byte[] data = reader.ReadBytes((int)subchunkSize);
                    OtherSubchunks[subchunkName] = data;
                    break;
            }
            
            // Make sure we're at the end of the subchunk
            reader.BaseStream.Position = subchunkStart + subchunkSize;
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ACNK".ToCharArray());
        
        // We need to calculate the total size, but don't know it yet
        // So remember this position to come back and write it later
        long sizePosition = writer.BaseStream.Position;
        writer.Write(0); // Placeholder for size
        
        long startPosition = writer.BaseStream.Position;
        
        // Write header
        writer.Write(IndexX);
        writer.Write(IndexY);
        writer.Write(Flags);
        writer.Write(AreaId);
        writer.Write(HolesLowRes);
        
        // Write low detail texturing map
        for (int i = 0; i < 4; i++)
            writer.Write(LowDetailTexturingMap[i]);
            
        writer.Write(Mcdd);
        
        // Write padding
        for (int i = 0; i < 6; i++)
            writer.Write((byte)0);
            
        writer.Write(Unknown);
        
        // Write high-res holes
        writer.Write(HighResHoles);
        
        // Write padding
        for (int i = 0; i < 6; i++)
            writer.Write((byte)0);
            
        // Write texture layers
        foreach (var layer in TextureLayers)
        {
            layer.Save(writer);
        }
        
        // Write shadow map if present
        if (ShadowMap != null)
        {
            ShadowMap.Save(writer);
        }
        
        // Write objects
        foreach (var obj in Objects)
        {
            obj.Save(writer);
        }
        
        // Write other subchunks
        foreach (var kvp in OtherSubchunks)
        {
            writer.Write(kvp.Key.ToCharArray());
            writer.Write(kvp.Value.Length);
            writer.Write(kvp.Value);
        }
        
        // Now go back and write the correct size
        long endPosition = writer.BaseStream.Position;
        uint size = (uint)(endPosition - startPosition);
        
        writer.BaseStream.Position = sizePosition;
        writer.Write(size);
        
        // Return to the end of the chunk
        writer.BaseStream.Position = endPosition;
    }
    
    // Helper methods
    
    // Check if chunk has a specific hole
    public bool HasHole(int quadX, int quadY)
    {
        if ((Flags & 0x10000) != 0)
        {
            // Using high-res holes (8x8 grid per chunk)
            int index = quadY * 8 + quadX;
            int byteIndex = index / 8;
            int bitIndex = index % 8;
            
            if (byteIndex < HighResHoles.Length)
                return (HighResHoles[byteIndex] & (1 << bitIndex)) != 0;
        }
        else
        {
            // Using low-res holes (4x4 grid per chunk)
            int index = quadY * 4 + quadX;
            return (HolesLowRes & (1 << index)) != 0;
        }
        
        return false;
    }
    
    // Set a specific hole
    public void SetHole(int quadX, int quadY, bool hasHole)
    {
        if ((Flags & 0x10000) != 0)
        {
            // Using high-res holes (8x8 grid per chunk)
            int index = quadY * 8 + quadX;
            int byteIndex = index / 8;
            int bitIndex = index % 8;
            
            if (byteIndex < HighResHoles.Length)
            {
                if (hasHole)
                    HighResHoles[byteIndex] |= (byte)(1 << bitIndex);
                else
                    HighResHoles[byteIndex] &= (byte)~(1 << bitIndex);
            }
        }
        else
        {
            // Using low-res holes (4x4 grid per chunk)
            int index = quadY * 4 + quadX;
            
            if (hasHole)
                HolesLowRes |= (ushort)(1 << index);
            else
                HolesLowRes &= (ushort)~(1 << index);
        }
    }
    
    // Enable high-res holes
    public void EnableHighResHoles()
    {
        if ((Flags & 0x10000) == 0)
        {
            Flags |= 0x10000;
            
            // Convert low-res holes to high-res
            for (int y = 0; y < 4; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    bool hasHole = (HolesLowRes & (1 << (y * 4 + x))) != 0;
                    
                    // Set 4 high-res holes for each low-res hole
                    SetHole(x * 2, y * 2, hasHole);
                    SetHole(x * 2 + 1, y * 2, hasHole);
                    SetHole(x * 2, y * 2 + 1, hasHole);
                    SetHole(x * 2 + 1, y * 2 + 1, hasHole);
                }
            }
        }
    }
}
```

## Usage Context

The ACNK chunk plays a central role in the ADT v23 format's terrain representation, providing detailed information about specific terrain sections. Its main functions include:

1. **Terrain Subdivision**: Each ACNK chunk represents a specific section of the overall terrain grid, allowing for localized detail and efficient culling.

2. **Texture Layering**: Through its ALYR subchunks, ACNK defines which textures are applied to the terrain and how they blend together.

3. **Object Placement**: ACDO subchunks define where models (doodads and WMOs) are placed within the chunk's boundaries.

4. **Terrain Features**: The holes mask defines areas where the terrain has been "cut out" to create caves, tunnels, or other terrain features.

5. **Environmental Settings**: The areaId field helps define the environmental properties of the chunk, such as PvP status, weather effects, and minimap colors.

The v23 format's approach to chunk data differs significantly from v18. While v18 embeds vertex data directly in each MCNK chunk, v23 centralizes vertex data in global chunks (AVTX, ANRM, ACVT) and uses ACNK merely as a container for localized data like texture layers and object placements. This approach might have been intended to reduce data duplication and improve memory access patterns.

Though never used in any retail release, this experimental approach in the ADT v23 format provides insight into how Blizzard was exploring ways to optimize terrain data organization during the Cataclysm beta development period. 