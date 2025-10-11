# W009: MDID

## Type
WDT Chunk

## Source
WDT.md

## Description
The MDID (Map Doodad Index) chunk contains offsets into the MDNM chunk, providing an efficient way to reference specific M2 doodad filenames. This chunk works with the MDNM and MDDF chunks to define global M2 model placements in the map.

## Structure
```csharp
struct MDID
{
    /*0x00*/ uint32_t offsets[];  // Array of offsets into the MDNM chunk
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offsets | uint32_t[] | Array of offset values pointing into the MDNM chunk |

## Dependencies
- MDNM (W007) - Contains the M2 filenames this chunk references
- MDDF (W008) - Uses the indices from this chunk to reference M2 filenames

## Implementation Notes
- The MDID chunk is only present if the map contains global doodad objects
- Each entry in the `offsets` array is a byte offset into the MDNM chunk
- These offsets point to the start of the corresponding null-terminated string
- The number of entries in this chunk should match the number of doodad placements in the MDDF chunk
- This chunk uses the same string table pattern seen in MWID and other similar chunks
- The offsets are relative to the start of the string data in the MDNM chunk

## Implementation Example
```csharp
public class MDID : IChunk
{
    public List<uint> Offsets { get; private set; } = new List<uint>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate number of offsets (each offset is 4 bytes)
        int count = (int)(size / 4);
        Offsets.Clear();
        
        for (int i = 0; i < count; i++)
        {
            uint offset = reader.ReadUInt32();
            Offsets.Add(offset);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (uint offset in Offsets)
        {
            writer.Write(offset);
        }
    }
    
    public int GetFilenameCount()
    {
        return Offsets.Count;
    }
    
    public uint GetOffset(int index)
    {
        if (index < 0 || index >= Offsets.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        return Offsets[index];
    }
    
    // Helper method to add a new offset
    public void AddOffset(uint offset)
    {
        Offsets.Add(offset);
    }
    
    // Get string from MDNM chunk using the offset at given index
    public string GetFilename(int index, MDNM mdnmChunk)
    {
        if (index < 0 || index >= Offsets.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        uint offset = Offsets[index];
        return mdnmChunk.GetFilenameByOffset((int)offset);
    }
}
```

## String Table Pattern
The MDID chunk is part of the "String Table Pattern" that appears throughout WoW file formats:
1. A string table chunk (MDNM) contains concatenated null-terminated strings
2. An index chunk (MDID) contains offsets into the string table
3. A placement chunk (MDDF) references strings by using indices into the offset table

This three-part system allows for efficient storage and lookup of strings while minimizing duplication. It's the same pattern used by MWMO/MWID/MODF for WMO objects, but applied to M2 doodad models.

## Version Information
- Present in later versions of WDT files (version 18+) if global doodad objects are included
- The structure remains consistent across all WDT versions that include it
- May be absent in earlier versions of the WDT format
- In version 22+, the references may be altered to work with FileDataIDs

## Usage Context
The MDID chunk works together with the MDNM and MDDF chunks to define global M2 model placements:

1. MDNM contains the actual M2 model filenames as strings
2. MDID contains byte offsets to efficiently reference these filenames
3. MDDF contains placement information and references filenames via indices into MDID

This system allows for:
- Efficient storage of M2 filenames (each unique filename is stored only once)
- Easy lookup of filenames when needed
- Compact references in the MDDF placement data

For example, if a WDT needs to place the same M2 model in multiple locations, the filename is stored once in MDNM, referenced once in MDID, and then the MDDF entries can all reference the same index. 