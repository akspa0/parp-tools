# O003: MOTX

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOTX (Map Object TeXture names) chunk contains a list of null-terminated filenames for textures used in the WMO. Each filename is referenced by offset within this chunk from the MOMT (materials) chunk. This chunk stores the texture paths as relative paths from the WoW base directory.

## Structure
```csharp
struct MOTX
{
    char[] filenames;  // Concatenated list of null-terminated filenames
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | A continuous block of null-terminated strings containing texture filenames |

## Dependencies
- MOHD: The nTextures field in MOHD indicates how many distinct textures should be referenced by MOMT
- MOMT: References textures in this chunk by their offset from the start of MOTX

## Implementation Notes
- Texture filenames are stored as null-terminated strings in a continuous block
- Filenames are referenced by offset (in bytes) from the beginning of this chunk
- All filenames are relative paths from the base WoW directory (e.g., "textures/stone/granite01.blp")
- The total number of textures is defined in the MOHD chunk's nTextures field
- Each filename is a null-terminated string, meaning it ends with a '\0' character
- The chunk itself doesn't store any count or indexing information—this is maintained by other chunks that reference the filenames
- Texture path strings must end with a null-terminator (0x00)
- Empty strings are represented as a single null byte
- Multiple consecutive null terminators may indicate empty strings or padding

## Implementation Example
```csharp
public class MOTX : IChunk
{
    public byte[] RawData { get; private set; }
    public List<string> Filenames { get; private set; }
    private Dictionary<string, int> _offsetLookup;
    
    public MOTX()
    {
        Filenames = new List<string>();
        _offsetLookup = new Dictionary<string, int>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Store the raw data for calculating offsets later
        RawData = reader.ReadBytes((int)size);
        
        // Extract filenames from the raw data
        Filenames.Clear();
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
                Filenames.Add(filename);
                _offsetLookup[filename] = stringStart;
            }
            else
            {
                // Empty string (just a null terminator)
                Filenames.Add(string.Empty);
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
        foreach (string filename in Filenames)
        {
            totalSize += filename.Length + 1; // +1 for null terminator
        }
        
        // Create a buffer to hold all filenames
        byte[] buffer = new byte[totalSize];
        int offset = 0;
        
        // Reset the offset lookup
        _offsetLookup.Clear();
        
        // Write each filename to the buffer
        foreach (string filename in Filenames)
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
        Filenames.Add(filename);
        // Note: offsets will be calculated during Write
    }
}
```

## Validation Requirements
- All filenames should be properly null-terminated
- The filenames should be valid relative paths within the WoW directory structure
- The number of unique filenames should match the nTextures field in the MOHD chunk
- Each filename should have a valid extension (typically .blp for WoW textures)
- The chunk should contain at least one filename (even empty WMOs typically have at least a default texture)

## Usage Context
The MOTX chunk serves as a central repository for texture filenames used throughout the WMO. Its main purposes are:

1. **Centralized Storage**: Consolidates all texture paths in one place, reducing redundancy
2. **Material Definition**: Provides texture paths for materials defined in the MOMT chunk
3. **Resource Management**: Allows the engine to preload textures before rendering
4. **Memory Efficiency**: Allows referencing the same texture multiple times by offset

When rendering the WMO, the engine:
1. Reads texture filenames from this chunk
2. Loads the textures into video memory
3. Applies them to corresponding materials as specified in the MOMT chunk
4. Uses these materials for rendering the model geometry

This architecture allows the same texture to be referenced multiple times within the model, saving both file size and memory usage. 