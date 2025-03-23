# C013: MDID

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
List of diffuse texture file IDs used for texturing the terrain in this map tile (8.1.0.27826+).

## Applicability
This section only applies to versions ≥ (8.1.0.27826). Replaces MTEX for newer versions.

## Structure
```csharp
struct MDID
{
    uint32_t file_data_id[]; // _s.blp
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| file_data_id | uint32[] | Array of file data IDs for diffuse textures (_s.blp) |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk

## Implementation Notes
- Split files: appears in tex0
- Replaces MTEX in newer versions of the game (8.1.0.27826+)
- Contains file data IDs instead of filenames for diffuse textures
- Each entry is an ID that can be used to load the texture directly
- The texture files use the _s.blp suffix (diffuse textures)
- Referenced in MCLY sub-chunks of MCNK

## Implementation Example
```csharp
public class MDID
{
    public List<uint> FileDataIds { get; set; } = new List<uint>();
}
```

## Parsing Example
```csharp
public MDID ParseMDID(byte[] data)
{
    var mdid = new MDID();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            mdid.FileDataIds.Add(reader.ReadUInt32());
        }
    }
    return mdid;
}
```

## Usage Context
The MDID chunk provides the file data IDs for diffuse textures used for the terrain in the map tile. These IDs can be used to load the textures directly without needing string paths. The MCLY sub-chunks in each MCNK reference these textures by their index in this array. 