# L003: MWID

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MWID (Map WMO Index) chunk contains indices that reference filenames in the MWMO chunk, creating a lookup table for global WMO model names used in the low-resolution map. These indices are used by the MODF chunk to specify which WMO model is placed at a particular location. The MWID chunk optimizes storage by allowing the MODF chunk to reference WMO names by index rather than storing the full filename for each placement.

## Structure
```csharp
struct MWID
{
    /*0x00*/ uint32_t[] offsets; // Array of offsets into the MWMO chunk
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| offsets | uint32_t[] | Array of offsets into the MWMO chunk where each filename begins |

## Offset Values
- Each offset is a 32-bit unsigned integer (uint32_t)
- The offset represents the position within the MWMO chunk where a specific filename starts
- The offset is relative to the start of the MWMO chunk's data section
- The number of elements in the array equals the number of WMO filenames in the MWMO chunk

## Dependencies
- MWMO (L002) - Contains the actual filenames that MWID references
- MODF (L004) - Uses the indices from MWID to reference WMO filenames

## Implementation Notes
- The MWID chunk may not be present if there are no global WMO objects in the low-resolution map
- The indices in this array are used by the MODF chunk to specify which WMO model is placed at a given location
- The number of entries in the MWID array should match the number of null-terminated strings in the MWMO chunk
- The values represent byte offsets into the MWMO chunk, not linear indices
- Using this offset-based system allows for efficient lookup of WMO filenames during rendering

## Implementation Example
```csharp
public class MWID : IChunk
{
    public List<uint> Offsets { get; private set; } = new List<uint>();
    
    public void Parse(BinaryReader reader, long size)
    {
        int numOffsets = (int)(size / sizeof(uint));
        Offsets.Clear();
        
        for (int i = 0; i < numOffsets; i++)
        {
            Offsets.Add(reader.ReadUInt32());
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (uint offset in Offsets)
        {
            writer.Write(offset);
        }
    }
    
    // Helper method to get the number of indices
    public int GetCount()
    {
        return Offsets.Count;
    }
    
    // Helper method to get an offset by index
    public uint GetOffset(int index)
    {
        if (index < 0 || index >= Offsets.Count)
            return 0;
            
        return Offsets[index];
    }
    
    // Helper method to add a new offset
    public void AddOffset(uint offset)
    {
        Offsets.Add(offset);
    }
    
    // Helper method to get the filename from the MWMO chunk using an index
    public string GetFilename(int index, MWMO mwmo)
    {
        if (index < 0 || index >= Offsets.Count)
            return string.Empty;
            
        // In a real implementation, you would use the offset to look up the filename in the MWMO chunk
        // For simplicity, we'll assume mwmo has a method to get filenames by index
        return mwmo.GetFilename(index);
    }
    
    // Helper method to build MWID offsets from a MWMO chunk
    public static MWID BuildFromMWMO(MWMO mwmo)
    {
        MWID mwid = new MWID();
        uint currentOffset = 0;
        
        foreach (string filename in mwmo.Filenames)
        {
            mwid.AddOffset(currentOffset);
            // Move to the next filename (including null terminator)
            currentOffset += (uint)(filename.Length + 1);
        }
        
        return mwid;
    }
}
```

## Relationship to WMO Placement
The MWID chunk creates an indirection layer between WMO placement and filenames:

1. MODF chunks reference WMOs by their indices in the MWID array
2. MWID contains offsets into the MWMO chunk
3. MWMO contains the actual filenames

This design pattern optimizes storage by:
- Avoiding repetition of identical filenames when the same WMO is placed multiple times
- Creating a compact form of reference (32-bit index) rather than storing full paths
- Allowing for efficient string comparison and lookup during rendering

## Relationship to WDT
The MWID chunk in WDL shares the same structure and purpose as its counterpart in WDT files:

- Both contain offsets into their respective MWMO chunks
- Both are used to reference WMO filenames by index
- WDL typically contains a subset of the WMOs defined in WDT
- The indices in WDL MWID may not correspond to the same indices in WDT MWID

## Usage with MODF
The MWID indices are used in the MODF chunk's `nameId` field:

1. MODF contains a `nameId` value
2. This value is an index into the MWID array
3. MWID provides an offset into MWMO
4. MWMO contains the actual WMO filename

This creates the full reference chain: MODF → MWID → MWMO → filename

## Version Information
- The MWID chunk format remains consistent across different versions
- The chunk may be absent in maps with no distant WMO objects 