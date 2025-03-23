# L002: MWMO

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MWMO (Map WMO) chunk contains the filenames of global WMO (World Map Object) models that appear in the low-resolution version of the map. These are typically large structures that span multiple map areas or are visible from a great distance. The MWMO chunk in WDL is similar to its counterpart in WDT files but contains only the subset of WMO models that need to be rendered at a distance.

## Structure
```csharp
struct MWMO
{
    /*0x00*/ char[] filenames; // Sequence of null-terminated strings
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Array of null-terminated strings containing WMO filenames |

## String Format
- Each filename is stored as a null-terminated string
- Multiple filenames are stored consecutively
- Example format: "World\\wmo\\path\\object1.wmo\0World\\wmo\\path\\object2.wmo\0"
- Filenames are relative to the World of Warcraft installation directory
- Typically using Windows path format with backslashes
- The chunk size minus the total length of all strings (including null terminators) should be zero

## Dependencies
- MWID (L003) - Contains indices to the WMO filenames stored in this chunk
- MODF (L004) - Contains placement information for the WMO models

## Implementation Notes
- The MWMO chunk may not be present if there are no global WMO objects in the low-resolution map
- The string data must be parsed carefully to extract individual filenames
- Some filenames may contain special characters
- The index in the array corresponds to the index used in the MWID chunk
- In the WDL format, the MWMO chunk typically contains a subset of the WMOs listed in the equivalent WDT chunk
- The presence of a WMO in this chunk indicates it's significant enough to be rendered even at great distances

## Implementation Example
```csharp
public class MWMO : IChunk
{
    public List<string> Filenames { get; private set; } = new List<string>();
    
    public void Parse(BinaryReader reader, long size)
    {
        long endPos = reader.BaseStream.Position + size;
        Filenames.Clear();
        
        // Read null-terminated strings until we reach the end of the chunk
        while (reader.BaseStream.Position < endPos)
        {
            List<char> chars = new List<char>();
            char c;
            
            // Read characters until we hit a null terminator
            while ((c = reader.ReadChar()) != '\0' && reader.BaseStream.Position < endPos)
            {
                chars.Add(c);
            }
            
            // Convert the character array to a string and add it to our list
            if (chars.Count > 0)
            {
                Filenames.Add(new string(chars.ToArray()));
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write each filename as a null-terminated string
        foreach (string filename in Filenames)
        {
            // Write the characters of the filename
            foreach (char c in filename)
            {
                writer.Write(c);
            }
            
            // Write the null terminator
            writer.Write('\0');
        }
    }
    
    // Helper method to get a filename by index
    public string GetFilename(int index)
    {
        if (index < 0 || index >= Filenames.Count)
            return string.Empty;
            
        return Filenames[index];
    }
    
    // Helper method to add a new filename
    public int AddFilename(string filename)
    {
        Filenames.Add(filename);
        return Filenames.Count - 1;
    }
    
    // Helper method to check if a filename exists and get its index
    public bool TryGetFilenameIndex(string filename, out int index)
    {
        index = Filenames.IndexOf(filename);
        return index >= 0;
    }
}
```

## Relationship to WDT
The MWMO chunk in WDL serves a similar purpose to its counterpart in WDT files, but with some key differences:

- WDL MWMO contains a subset of the WMOs listed in WDT MWMO
- Only WMOs visible from a distance are included
- The formatting and structure remain the same across both formats
- The indices in WDL MWID may not directly correspond to the indices in WDT MWID

## Low-Resolution Rendering Context
WMOs in the WDL MWMO chunk are selected for distant rendering:

- Typically large structures visible from far away
- May use simplified models or LOD (Level of Detail) variants
- Positioned to provide recognizable landmarks when viewing the world from a distance
- Examples include major buildings, mountains, and other significant structures

## Version Information
- The MWMO chunk format remains consistent across different versions
- The chunk may be absent in maps with no distant WMO objects 