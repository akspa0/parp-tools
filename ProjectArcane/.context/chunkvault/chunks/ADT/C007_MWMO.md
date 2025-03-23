# C007: MWMO

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
List of filenames for WMO (World Model Object) models used in this map tile.

## Structure
```csharp
struct MWMO 
{ 
    char filenames[0]; // zero-terminated strings with complete paths to WMO models. Referenced in MODF and MWID.
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Variable-length array of zero-terminated strings with complete paths to WMO model files |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk

## Implementation Notes
- Split files: appears in obj file
- Contains multiple null-terminated strings concatenated together
- The strings are complete paths to WMO model files
- Referenced by MODF and MWID chunks
- Similar structure to MTEX and MMDX, but for WMO models

## Implementation Example
```csharp
public class MWMO
{
    public List<string> Filenames { get; set; } = new List<string>();
}
```

## Parsing Example
```csharp
public MWMO ParseMWMO(byte[] data)
{
    var mwmo = new MWMO();
    var currentOffset = 0;
    
    while (currentOffset < data.Length)
    {
        // Find the null terminator
        var stringEnd = Array.IndexOf(data, (byte)0, currentOffset);
        if (stringEnd == -1) break;
        
        // Extract the string
        var stringLength = stringEnd - currentOffset;
        var modelFilename = Encoding.ASCII.GetString(data, currentOffset, stringLength);
        mwmo.Filenames.Add(modelFilename);
        
        // Move past the null terminator
        currentOffset = stringEnd + 1;
    }
    
    return mwmo;
}
```

## Usage Context
The MWMO chunk provides the list of WMO (World Model Object) filenames used in the map tile. WMO models represent more complex structures like buildings, caves, and other large environmental objects. The MODF and MWID chunks reference these WMO models by their index in this list. 