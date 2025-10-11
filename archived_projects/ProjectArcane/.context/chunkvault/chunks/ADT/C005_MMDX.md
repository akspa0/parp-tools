# C005: MMDX

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
List of filenames for M2 models (doodads) used in this map tile.

## Structure
```csharp
struct MMDX 
{ 
    char filenames[0]; // zero-terminated strings with complete paths to models. Referenced in MDDF and MMID.
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Variable-length array of zero-terminated strings with complete paths to M2 model files |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk

## Implementation Notes
- Split files: appears in obj file
- Contains multiple null-terminated strings concatenated together
- The strings are complete paths to M2 model files
- Referenced by MDDF and MMID chunks
- Similar structure to MTEX, but for models instead of textures

## Implementation Example
```csharp
public class MMDX
{
    public List<string> Filenames { get; set; } = new List<string>();
}
```

## Parsing Example
```csharp
public MMDX ParseMMDX(byte[] data)
{
    var mmdx = new MMDX();
    var currentOffset = 0;
    
    while (currentOffset < data.Length)
    {
        // Find the null terminator
        var stringEnd = Array.IndexOf(data, (byte)0, currentOffset);
        if (stringEnd == -1) break;
        
        // Extract the string
        var stringLength = stringEnd - currentOffset;
        var modelFilename = Encoding.ASCII.GetString(data, currentOffset, stringLength);
        mmdx.Filenames.Add(modelFilename);
        
        // Move past the null terminator
        currentOffset = stringEnd + 1;
    }
    
    return mmdx;
}
```

## Usage Context
The MMDX chunk provides the list of M2 model filenames used in the map tile. These models are the doodads (objects) placed in the world, like trees, rocks, and other decorative elements. The MDDF and MMID chunks reference these models by their index in this list. 