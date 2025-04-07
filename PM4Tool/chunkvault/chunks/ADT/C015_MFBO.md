# C015: MFBO

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Flight boundaries information for the map tile, introduced in Burning Crusade.

## Applicability
This section only applies to versions ≥ Burning Crusade (BC+).

## Structure
```csharp
struct MFBO
{
    uint32_t max_height_count;
    uint32_t min_height_count;
    int16_t max_heights[max_height_count];   // Maximum flight ceiling
    int16_t min_heights[min_height_count];   // Minimum flight floor
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| max_height_count | uint32 | Number of maximum height entries |
| min_height_count | uint32 | Number of minimum height entries |
| max_heights | int16[] | Array of maximum flight ceilings |
| min_heights | int16[] | Array of minimum flight floors |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk and flags indicating its presence

## Implementation Notes
- Presence is indicated by MHDR.flags & 1 (mhdr_MFBO flag)
- The offset is only set in MHDR if the flag is present
- Height values are limited by signed shorts (int16), with maximum value of 2^15
- Defines the flight boundaries (ceiling and floor) for the map tile
- Commonly used to prevent players from flying too high or too low in certain areas

## Implementation Example
```csharp
public class MFBO
{
    public List<short> MaxHeights { get; set; } = new List<short>();
    public List<short> MinHeights { get; set; } = new List<short>();
}
```

## Parsing Example
```csharp
public MFBO ParseMFBO(byte[] data)
{
    var mfbo = new MFBO();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        uint maxHeightCount = reader.ReadUInt32();
        uint minHeightCount = reader.ReadUInt32();
        
        for (int i = 0; i < maxHeightCount; i++)
        {
            mfbo.MaxHeights.Add(reader.ReadInt16());
        }
        
        for (int i = 0; i < minHeightCount; i++)
        {
            mfbo.MinHeights.Add(reader.ReadInt16());
        }
    }
    return mfbo;
}
```

## Usage Context
The MFBO chunk provides flight boundaries for the map tile, restricting how high or low players can fly in certain areas. This was introduced in Burning Crusade with the addition of flying mounts. The presence of this chunk is indicated by the mhdr_MFBO flag in the MHDR chunk. 