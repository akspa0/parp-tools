# ALYR - Texture Layer Information

## Type
ADT v22 ACNK Subchunk

## Source
Referenced from `ADT_v22.md`

## Description
The ALYR (Layer) subchunk contains texture layer information for an ACNK chunk in the ADT v22 format. It defines how textures are applied to the terrain and how they blend with other layers. Each ACNK chunk can have multiple texture layers that are blended together to create the final terrain appearance.

## Structure

```csharp
public struct ALYR
{
    public int textureID;     // Index into the ATEX array for this texture
    public int flags;         // Texture application flags
    public uint padding1;     // Padding/reserved
    public uint padding2;     // Padding/reserved
    public uint padding3;     // Padding/reserved
    public uint padding4;     // Padding/reserved
    public uint padding5;     // Padding/reserved
    public uint padding6;     // Padding/reserved
    
    // The ALYR chunk may include an embedded AMAP (alpha map) if flags & 0x100
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| textureID | int | Index into the ATEX array for the texture to use |
| flags | int | Flags controlling texture behavior and alpha map usage |

## Flags

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x100 | HasAlphaMap | Contains an embedded AMAP subchunk with alpha data |

## Dependencies

- ACNK (C006) - Parent chunk that contains this subchunk
- ATEX (C004) - Texture list referenced by the textureID field
- AMAP (S002) - May be embedded within an ALYR if flags & 0x100

## Implementation Notes

1. The ALYR subchunk is similar in purpose to the MCLY subchunk in ADT v18, but with a different structure.

2. When the HasAlphaMap flag (0x100) is set, the ALYR chunk will be followed by an embedded AMAP chunk containing the alpha map data.

3. Each ACNK can have multiple ALYR subchunks, representing different texture layers to be blended together.

4. The texture layers are applied in order, with each subsequent layer blending with the previous layers according to its alpha map.

5. The textureID field references an entry in the ATEX chunk list, which provides the filename of the texture to use.

6. The standard size of an ALYR chunk is 0x20 bytes, but it can be larger if it contains an embedded AMAP chunk.

## Implementation Example

```csharp
[Flags]
public enum AlyrFlags
{
    None = 0x0,
    HasAlphaMap = 0x100
}

public class AlyrSubChunk
{
    public int TextureID { get; set; }
    public AlyrFlags Flags { get; set; }
    
    // The alpha map if present
    public AmapSubChunk AlphaMap { get; set; }
    
    // Helper properties
    public bool HasAlphaMap => (Flags & AlyrFlags.HasAlphaMap) != 0;
    
    public void Load(BinaryReader reader, long size)
    {
        long startPosition = reader.BaseStream.Position;
        
        TextureID = reader.ReadInt32();
        Flags = (AlyrFlags)reader.ReadInt32();
        
        // Skip padding fields (6 DWORDs)
        reader.BaseStream.Position += 24;
        
        // If we have an alpha map and there's still data to read
        if (HasAlphaMap && reader.BaseStream.Position < startPosition + size)
        {
            // Verify this is an AMAP chunk
            string chunkName = new string(reader.ReadChars(4));
            uint chunkSize = reader.ReadUInt32();
            
            if (chunkName == "AMAP")
            {
                AlphaMap = new AmapSubChunk();
                AlphaMap.Load(reader, chunkSize);
            }
            else
            {
                // Unexpected chunk type, rewind and let the parent chunk handle it
                reader.BaseStream.Position -= 8;
            }
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        writer.Write("ALYR".ToCharArray());
        
        // Calculate size (base size + AMAP size if present)
        uint size = 0x20; // Base size
        if (HasAlphaMap && AlphaMap != null)
        {
            // Add 8 bytes for AMAP header plus AMAP data size
            size += 8 + (uint)AlphaMap.AlphaData.Length;
        }
        
        writer.Write(size);
        writer.Write(TextureID);
        writer.Write((int)Flags);
        
        // Write padding fields
        for (int i = 0; i < 6; i++)
        {
            writer.Write(0); // 6 DWORDs of padding
        }
        
        // Write AMAP chunk if present
        if (HasAlphaMap && AlphaMap != null)
        {
            AlphaMap.Save(writer);
        }
    }
}
```

## Usage Context

The ALYR subchunk is essential for controlling the visual appearance of terrain in the ADT v22 format. Each ALYR defines a layer of texture that is applied to the terrain, and multiple layers can be stacked to create complex terrain surfaces with different materials (grass, dirt, rock, etc.).

The system works as follows:

1. The base layer (first ALYR) provides the foundation texture for the terrain chunk.

2. Additional layers are blended on top using their alpha maps (provided in an AMAP subchunk) to control transparency.

3. The alpha map determines where and how strongly each texture shows through the previous layers.

This layering system allows map designers to create detailed and realistic terrain surfaces by combining multiple textures in various ways. For example:

- A rocky mountain might use a base rock texture with patches of snow (second layer) and moss (third layer)
- A shoreline might blend between sand, grass, and dirt textures
- A path through a forest might blend between grass and dirt textures along the trail

The v22 format's approach to texture layering is more centralized than v18, with all layer information directly in the ACNK chunk rather than spread across multiple chunks and subchunks. This experimental approach in the Cataclysm beta was ultimately abandoned, but it offers insight into alternate ways of organizing terrain texture data. 