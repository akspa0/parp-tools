# C014: MHID

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
List of height texture file IDs used for texturing the terrain in this map tile (8.1.0.27826+).

## Applicability
This section only applies to versions ≥ (8.1.0.27826). Used alongside MDID in newer versions.

## Structure
```csharp
struct MHID
{
    uint32_t file_data_id[]; // _h.blp; 0 if there is none
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| file_data_id | uint32[] | Array of file data IDs for height textures (_h.blp), 0 if there is none |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MDID (C013) - Must match the number of entries in MDID

## Implementation Notes
- Split files: appears in tex0
- Introduced in newer versions of the game (8.1.0.27826+)
- Contains file data IDs instead of filenames for height textures
- Each entry is an ID that can be used to load the height texture directly
- The texture files use the _h.blp suffix (height textures)
- A value of 0 indicates that there is no height texture for the corresponding diffuse texture
- The number of entries must match the number of entries in MDID

## Implementation Example
```csharp
public class MHID
{
    public List<uint> FileDataIds { get; set; } = new List<uint>();
}
```

## Parsing Example
```csharp
public MHID ParseMHID(byte[] data)
{
    var mhid = new MHID();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            mhid.FileDataIds.Add(reader.ReadUInt32());
        }
    }
    return mhid;
}
```

## Usage Context
The MHID chunk provides the file data IDs for height textures used for the terrain in the map tile. These IDs can be used to load the textures directly without needing string paths. Each entry corresponds to the diffuse texture with the same index in the MDID chunk. If an entry is 0, it means there is no height texture for that diffuse texture. 