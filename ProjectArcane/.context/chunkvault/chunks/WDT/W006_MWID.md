# W006: MWID

## Type
WDT Chunk

## Source
WDT.md

## Description
The MWID (WMO Index) chunk contains offsets into the MWMO chunk, providing an efficient way to reference specific WMO filenames. This chunk works with the MWMO and MODF chunks to define global WMO placements in the map.

## Structure
```csharp
struct MWID
{
    /*0x00*/ uint32_t offsets[];  // Array of offsets into the MWMO chunk
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offsets | uint32_t[] | Array of offset values pointing into the MWMO chunk |

## Dependencies
- MWMO (W004) - Contains the WMO filenames this chunk references
- MODF (W005) - Uses the indices from this chunk to reference WMO filenames

## Implementation Notes
- The MWID chunk is only present if the map contains global WMO objects
- Each entry in the `offsets` array is a byte offset into the MWMO chunk
- These offsets point to the start of the corresponding null-terminated string
- The number of entries in this chunk should match the number of WMO placements in the MODF chunk
- This chunk uses the same string table pattern seen in other chunks like MMID in ADT files
- The offsets are relative to the start of the string data in the MWMO chunk

## Implementation Example
```csharp
public class MWID : IChunk
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
    
    // Get string from MWMO chunk using the offset at given index
    public string GetFilename(int index, MWMO mwmoChunk)
    {
        if (index < 0 || index >= Offsets.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        uint offset = Offsets[index];
        return mwmoChunk.GetFilenameByOffset((int)offset);
    }
}
```

## String Table Pattern
The MWID chunk is part of the "String Table Pattern" that appears throughout WoW file formats:
1. A string table chunk (MWMO) contains concatenated null-terminated strings
2. An index chunk (MWID) contains offsets into the string table
3. Data chunks (MODF) reference strings by using indices into the offset table

This three-part system allows for efficient storage and lookup of strings while minimizing duplication.

## Version Information
- Present in all versions of WDT files if the map contains global WMO objects
- Structure remains consistent across all WDT versions
- The functionality is identical to similar index chunks in other file formats

## Usage Context
The MWID chunk works together with the MWMO and MODF chunks to define global WMO placements:

1. MWMO contains the actual WMO filenames as strings
2. MWID contains byte offsets to efficiently reference these filenames
3. MODF contains placement information and references filenames via indices into MWID

This system allows for:
- Efficient storage of WMO filenames (each unique filename is stored only once)
- Easy lookup of filenames when needed
- Compact references in the MODF placement data

For example, if a WDT needs to place the same WMO in multiple locations, the filename is stored once in MWMO, referenced once in MWID, and then the MODF entries can all reference the same index. 