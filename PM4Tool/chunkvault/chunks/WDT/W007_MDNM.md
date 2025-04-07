# W007: MDNM

## Type
WDT Chunk

## Source
WDT.md

## Description
The MDNM (Map Doodad Names) chunk contains filenames of M2 model files (doodads) that are placed globally in the world map. Similar to how MWMO contains global WMO filenames, MDNM provides a string table for global doodad models. This chunk is typically paired with MDDF and MDID to define global doodad placements.

## Structure
```csharp
struct MDNM
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
- Example format: "path/to/file1.m2\0path/to/file2.m2\0"
- Filenames are typically relative paths within the WoW data hierarchy
- Common path format: "World\\model\\path\\to\\filename.m2"
- Backslashes are used as path separators (Windows style)

## Dependencies
- MDID (not documented yet) - Contains offsets into this chunk
- MDDF (not documented yet) - Contains doodad placement data that references these filenames via MDID

## Implementation Notes
- The MDNM chunk is only present if the map contains global doodad objects
- The chunk contains no header besides the standard chunk header
- It is effectively a raw string table
- String offsets in the MDID chunk are relative to the start of the string data (position 0x00)
- Filenames are typically stored in lowercase
- The total size of the chunk is the sum of all string lengths, including null terminators
- This chunk mirrors the functionality of MWMO but for M2 models instead of WMO objects

## Implementation Example
```csharp
public class MDNM : IChunk
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
    
    // Calculate offset for a filename (used when building MDID)
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
- Present in later versions of WDT files (version 18+) if global doodad objects are included
- The structure remains consistent across all WDT versions that include it
- May be absent in earlier versions of the WDT format
- In version 22+, the references may be replaced with FileDataIDs in the MDDF chunk

## String Table System
The MDNM chunk is part of the "String Table System" used throughout WoW file formats:
1. A string table chunk (MDNM) contains concatenated null-terminated strings
2. An index chunk (MDID) contains offsets into the string table
3. A placement chunk (MDDF) references strings by using indices into the offset table

This three-part system is the same pattern used by MWMO/MWID/MODF for WMO objects.

## Usage Context
The MDNM chunk provides filenames for global M2 model objects in the world:

- Global doodads are M2 model objects placed directly in the world (not part of an ADT tile)
- Unlike ADT-specific doodads, global doodads are defined at the world level
- Examples include large statues, special objects, or other notable features
- The filenames stored here point to .m2 model files that contain the 3D models
- These M2 models typically include their own textures, animations, and properties

Global doodads serve several purposes:
1. Objects that should be visible from long distances
2. Important objects that span multiple ADT tiles
3. Objects that are referenced by world scripts
4. Special interactive elements

The MDNM chunk works together with the MDID and MDDF chunks in the global doodad system:
1. MDNM stores unique filenames
2. MDID provides lookup indices
3. MDDF stores positioning, rotation, and scale information for each doodad instance 