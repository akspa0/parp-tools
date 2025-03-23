# O009: MODN

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MODN (Map Object DoodadNames) chunk contains a list of null-terminated strings that provide filenames for M2 models (doodads) placed within the WMO. These doodad models are small objects like furniture, decorations, or fixtures that enhance the visual detail of the WMO. The MODD chunk references these filenames by offset to define where and how each doodad is placed in the world.

## Structure
```csharp
struct MODN
{
    char[] filenames;  // Concatenated list of null-terminated doodad filenames
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | A continuous block of null-terminated strings containing M2 model filenames |

## Dependencies
- MOHD: The nDoodadNames field indicates how many doodad filenames should be present
- MODD: References filenames in this chunk by their offset to define doodad placement

## Implementation Notes
- Doodad filenames are stored as null-terminated strings in a continuous block
- Filenames are referenced by offset (in bytes) from the beginning of this chunk
- All filenames are relative paths from the base WoW directory (e.g., "World/Expansion01/doodads/silvermoon/sm_chandelier_01.m2")
- The total number of unique doodad filenames is defined in the MOHD chunk's nDoodadNames field
- Each filename is a null-terminated string, meaning it ends with a '\0' character
- The chunk itself doesn't store any count or indexing information—this is maintained by other chunks
- Filenames always have the .m2 extension, as they reference M2 model files
- Empty strings are represented as a single null byte
- Multiple consecutive null terminators may indicate empty strings or padding
- This chunk follows the same storage pattern as MOTX (textures) and MOGN (group names)

## Implementation Example
```csharp
public class MODN : IChunk
{
    public byte[] RawData { get; private set; }
    public List<string> DoodadFilenames { get; private set; }
    private Dictionary<string, int> _offsetLookup;
    
    public MODN()
    {
        DoodadFilenames = new List<string>();
        _offsetLookup = new Dictionary<string, int>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Store the raw data for calculating offsets later
        RawData = reader.ReadBytes((int)size);
        
        // Extract doodad filenames from the raw data
        DoodadFilenames.Clear();
        _offsetLookup.Clear();
        
        int offset = 0;
        while (offset < RawData.Length)
        {
            // Find the null terminator
            int stringStart = offset;
            int stringEnd = stringStart;
            
            while (stringEnd < RawData.Length && RawData[stringEnd] != 0)
            {
                stringEnd++;
            }
            
            // Extract the filename as a string
            if (stringEnd > stringStart)
            {
                string filename = Encoding.ASCII.GetString(RawData, stringStart, stringEnd - stringStart);
                DoodadFilenames.Add(filename);
                _offsetLookup[filename] = stringStart;
            }
            else
            {
                // Empty string (just a null terminator)
                DoodadFilenames.Add(string.Empty);
                _offsetLookup[string.Empty] = stringStart;
            }
            
            // Move past the null terminator to the next string
            offset = stringEnd + 1;
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Calculate the total size needed
        int totalSize = 0;
        foreach (string filename in DoodadFilenames)
        {
            totalSize += filename.Length + 1; // +1 for null terminator
        }
        
        // Create a buffer to hold all filenames
        byte[] buffer = new byte[totalSize];
        int offset = 0;
        
        // Reset the offset lookup
        _offsetLookup.Clear();
        
        // Write each filename to the buffer
        foreach (string filename in DoodadFilenames)
        {
            _offsetLookup[filename] = offset;
            
            if (!string.IsNullOrEmpty(filename))
            {
                byte[] filenameBytes = Encoding.ASCII.GetBytes(filename);
                Buffer.BlockCopy(filenameBytes, 0, buffer, offset, filenameBytes.Length);
                offset += filenameBytes.Length;
            }
            
            // Add null terminator
            buffer[offset++] = 0;
        }
        
        // Write the buffer to the output
        writer.Write(buffer);
        
        // Update the raw data
        RawData = buffer;
    }
    
    public int GetOffsetForFilename(string filename)
    {
        if (_offsetLookup.TryGetValue(filename, out int offset))
        {
            return offset;
        }
        
        return -1; // Not found
    }
    
    public string GetFilenameByOffset(int offset)
    {
        if (offset < 0 || offset >= RawData.Length)
        {
            return string.Empty;
        }
        
        // Find end of string (null terminator)
        int end = offset;
        while (end < RawData.Length && RawData[end] != 0)
        {
            end++;
        }
        
        // Extract the string
        int length = end - offset;
        if (length > 0)
        {
            return Encoding.ASCII.GetString(RawData, offset, length);
        }
        
        return string.Empty;
    }
    
    public void AddFilename(string filename)
    {
        DoodadFilenames.Add(filename);
        // Note: offsets will be calculated during Write
    }
}
```

## Validation Requirements
- All filenames should be properly null-terminated
- The filenames should be valid relative paths within the WoW directory structure
- The number of unique filenames should match the nDoodadNames field in the MOHD chunk
- Each filename should have the .m2 extension
- Filenames referenced from the MODD chunk should exist within this chunk
- The chunk should be present if nDoodadNames > 0 in the MOHD header

## Usage Context
The MODN chunk serves several important functions in the WMO format:

1. **Doodad Definition**: Provides the model filenames for all doodads used in the WMO
2. **Asset Management**: Centralizes the list of external M2 models needed for the WMO
3. **Memory Efficiency**: Allows the same model to be referenced multiple times with different placements
4. **Enhanced Detail**: Enables the addition of small detailed objects without increasing WMO complexity

Common types of doodads include:
- Furniture (chairs, tables, beds)
- Light fixtures (torches, lamps, chandeliers)
- Decorative elements (banners, books, pottery)
- Structural details (columns, beams, ornaments)
- Foliage (potted plants, small trees)

When rendering a WMO, the client:
1. Reads the doodad filenames from this chunk
2. Loads the referenced M2 models into memory
3. Places instances of these models according to the transformations in the MODD chunk
4. Renders the doodads as part of the complete WMO scene

The doodad system allows WMO creators to add significant detail and variety to buildings without having to model every small object as part of the main structure. Instead, reusable M2 models can be placed throughout the WMO, improving both visual quality and performance. 