# C004: MTEX

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
List of textures used for texturing the terrain in this map tile.

## Applicability
This section only applies to versions < (8.1.0.28294). MTEX has been replaced with file data ids in MDID and MHID chunks in later versions.

## Structure
```csharp
struct MTEX 
{ 
    char filenames[0]; // zero-terminated strings with complete paths to textures. Referenced in MCLY.
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Variable-length array of zero-terminated strings with complete paths to textures |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk

## Implementation Notes
- Split files: appears in tex file
- Contains multiple null-terminated strings concatenated together
- The strings are complete paths to texture files
- Referenced in MCLY sub-chunks of MCNK
- The texture indices are used by MCLY chunks to specify which textures are used in that map chunk

## Implementation Example
```csharp
public class MTEX
{
    public List<string> Filenames { get; set; } = new List<string>();
}
```

## Parsing Example
```csharp
public MTEX ParseMTEX(byte[] data)
{
    var mtex = new MTEX();
    var currentOffset = 0;
    
    while (currentOffset < data.Length)
    {
        // Find the null terminator
        var stringEnd = Array.IndexOf(data, (byte)0, currentOffset);
        if (stringEnd == -1) break;
        
        // Extract the string
        var stringLength = stringEnd - currentOffset;
        var textureFilename = Encoding.ASCII.GetString(data, currentOffset, stringLength);
        mtex.Filenames.Add(textureFilename);
        
        // Move past the null terminator
        currentOffset = stringEnd + 1;
    }
    
    return mtex;
}
```

## Usage Context
The MTEX chunk provides the textures used for the terrain in the map tile. The MCLY sub-chunks in each MCNK reference these textures by their index in this array. 