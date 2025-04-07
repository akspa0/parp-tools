# W004: MWMO

## Type
WDT Chunk

## Source
WDT.md

## Description
The MWMO (Map World Model Objects) chunk contains filenames of global WMO (World Map Object) models placed in the map. These are typically large structures that span multiple ADT tiles, such as major buildings, bridges, or other significant structures. This chunk works in conjunction with MWID and MODF chunks to define global WMO placements.

## Structure
```csharp
struct MWMO
{
    /*0x00*/ char filenames[];  // Concatenated null-terminated strings
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Array of characters containing all filenames as concatenated, null-terminated strings |

## String Format
- Each filename is stored as a null-terminated string
- Multiple filenames are concatenated together
- Example format: "path/to/file1.wmo\0path/to/file2.wmo\0"
- Filenames are typically relative paths within the WoW data hierarchy
- Common path format: "World\\wmo\\path\\to\\filename.wmo"
- Backslashes are used as path separators (Windows style)

## Dependencies
- MWID (W006) - Contains offsets into this chunk
- MODF (W005) - Contains doodad placement data that references these filenames via MWID

## Implementation Notes
- The MWMO chunk is only present if the map contains global WMO objects
- The chunk contains no header besides the standard chunk header
- It is effectively a raw string table
- String offsets in the MWID chunk are relative to the start of the string data (position 0x00)
- Filenames are typically stored in lowercase
- The total size of the chunk is the sum of all string lengths, including null terminators

## Implementation Example
```csharp
public class MWMO : IChunk
{
    public List<string> Filenames { get; private set; } = new List<string>();
    private Dictionary<int, string> _offsetToNameMap = new Dictionary<int, string>();
    
    public void Parse(BinaryReader reader, long size)
    {
        Filenames.Clear();
        _offsetToNameMap.Clear();
        
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        while (reader.BaseStream.Position < endPosition)
        {
            // Record current offset from the start of the chunk
            int currentOffset = (int)(reader.BaseStream.Position - startPosition);
            
            // Read null-terminated string
            StringBuilder sb = new StringBuilder();
            char c;
            while ((c = reader.ReadChar()) != '\0' && reader.BaseStream.Position < endPosition)
            {
                sb.Append(c);
            }
            
            // Add the null terminator back when computing offset to next string
            string filename = sb.ToString();
            Filenames.Add(filename);
            _offsetToNameMap[currentOffset] = filename;
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Rebuild offset map during write
        _offsetToNameMap.Clear();
        int currentOffset = 0;
        
        foreach (var filename in Filenames)
        {
            _offsetToNameMap[currentOffset] = filename;
            
            // Write filename with null terminator
            foreach (char c in filename)
            {
                writer.Write(c);
            }
            writer.Write('\0');
            
            // Update offset for next filename
            currentOffset += filename.Length + 1; // +1 for null terminator
        }
    }
    
    public string GetFilename(int index)
    {
        if (index < 0 || index >= Filenames.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        return Filenames[index];
    }
    
    // Get filename by byte offset from start of chunk data
    public string GetFilenameByOffset(int offset)
    {
        if (_offsetToNameMap.TryGetValue(offset, out string filename))
            return filename;
            
        throw new ArgumentException($"No filename found at offset {offset}");
    }
    
    // Calculate offset for a filename (used when building MWID)
    public int GetOffsetForFilename(string filename)
    {
        int currentOffset = 0;
        
        foreach (var name in Filenames)
        {
            if (name.Equals(filename, StringComparison.OrdinalIgnoreCase))
                return currentOffset;
                
            currentOffset += name.Length + 1; // +1 for null terminator
        }
        
        return -1; // Not found
    }
    
    // Add a new filename and return its offset
    public int AddFilename(string filename)
    {
        // Check if already exists
        int existingOffset = GetOffsetForFilename(filename);
        if (existingOffset >= 0)
            return existingOffset;
            
        // Calculate new offset
        int newOffset = 0;
        foreach (var name in Filenames)
        {
            newOffset += name.Length + 1;
        }
        
        Filenames.Add(filename);
        _offsetToNameMap[newOffset] = filename;
        return newOffset;
    }
}
```

## Version Information
- Present in all versions of WDT files if global WMO objects are included
- In version 18, all WMO names will always have their M2 (doodad set) files loaded
- In version 22+, only referenced M2s from a given WMO will be loaded
- The structure remains consistent across all WDT versions

## Usage Context
The MWMO chunk provides filenames for global WMO objects in the world:

- Global WMOs are structures that can span multiple map tiles
- Unlike ADT-specific WMOs, global WMOs are defined at the world level
- Examples include major cities, large bridges, and other significant structures
- The filenames stored here point to .wmo model files that contain the 3D models
- These WMO models can contain multiple grouped objects, materials, and lighting information
- The WMO files themselves can reference M2 models as doodads placed within the WMO

The MWMO chunk works together with the MWID and MODF chunks to define the complete WMO placement system:
1. MWMO stores unique filenames
2. MWID provides lookup indices
3. MODF stores positioning, rotation, and scale information for each WMO instance 