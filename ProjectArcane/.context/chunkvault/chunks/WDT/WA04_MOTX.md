# WA04: MOTX

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MOTX (Map Object Texture) chunk contains a list of texture filenames used by the map objects in the Alpha WDT format. This chunk stores texture paths in a continuous block of null-terminated strings, which are referenced by objects in the MAOI chunk through offsets.

## Structure
```csharp
struct MOTX
{
    // Variable-length block of null-terminated texture filename strings
    // Strings are referenced by offset from other chunks
    char[] texture_filenames;
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| texture_filenames | char[] | A continuous block of null-terminated texture filename strings |

## String Format
Texture filenames are stored as null-terminated strings concatenated together in a single block:

```
Texture1.blp\0Texture2.blp\0Texture3.blp\0...
```

Where `\0` represents a null terminator (byte value 0).

## String Referencing
Other chunks (primarily MAOI) reference these strings by their offset from the start of the MOTX data. For example:
- Offset 0 would reference the first texture ("Texture1.blp")
- Offset 12 (length of "Texture1.blp\0") would reference the second texture ("Texture2.blp")

## Dependencies
- MAOI (WA02) - Contains texture references that point to this chunk
- MAOT (WA01) - Indirectly related, as it indexes the MAOI entries that reference these textures

## Implementation Notes
- The MOTX chunk serves as a string table, optimizing storage by eliminating duplicate texture paths
- Texture paths are typically relative to the client's texture directory
- Common formats include .BLP (Blizzard Picture) files
- Most offsets in an Alpha WDT file that end with _name_offset or similar reference this chunk
- The chunk is present only when the map contains textured objects

## Implementation Example
```csharp
public class MOTX : IChunk
{
    private byte[] rawData;
    private Dictionary<int, string> offsetToString = new Dictionary<int, string>();
    private Dictionary<string, int> stringToOffset = new Dictionary<string, int>();
    
    public IReadOnlyDictionary<int, string> TexturesByOffset => offsetToString;
    public IReadOnlyDictionary<string, int> OffsetsByTexture => stringToOffset;
    
    public void Parse(BinaryReader reader, long size)
    {
        // Read the raw data
        rawData = reader.ReadBytes((int)size);
        
        // Parse the null-terminated strings and build lookup dictionaries
        int offset = 0;
        while (offset < rawData.Length)
        {
            string textureName = ReadNullTerminatedString(rawData, ref offset);
            if (string.IsNullOrEmpty(textureName))
                break;
                
            offsetToString[offset - textureName.Length - 1] = textureName;
            stringToOffset[textureName] = offset - textureName.Length - 1;
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(rawData);
    }
    
    public string GetTextureNameByOffset(int offset)
    {
        if (offsetToString.TryGetValue(offset, out string textureName))
            return textureName;
        return null;
    }
    
    public int GetOffsetByTextureName(string textureName)
    {
        if (stringToOffset.TryGetValue(textureName, out int offset))
            return offset;
        return -1;
    }
    
    private string ReadNullTerminatedString(byte[] data, ref int offset)
    {
        int startOffset = offset;
        
        // Find the null terminator
        while (offset < data.Length && data[offset] != 0)
            offset++;
            
        if (offset >= data.Length)
            return string.Empty;
            
        // Extract the string
        string result = System.Text.Encoding.ASCII.GetString(data, startOffset, offset - startOffset);
        
        // Skip past the null terminator
        offset++;
        
        return result;
    }
    
    // Helper method to add a new texture filename and get its offset
    public int AddTextureName(string textureName)
    {
        // If the texture already exists, return its offset
        if (stringToOffset.TryGetValue(textureName, out int existingOffset))
            return existingOffset;
            
        // Otherwise, add it to the end
        int newOffset = rawData.Length;
        byte[] textureBytes = System.Text.Encoding.ASCII.GetBytes(textureName + '\0');
        
        // Resize rawData to accommodate the new string
        Array.Resize(ref rawData, rawData.Length + textureBytes.Length);
        Array.Copy(textureBytes, 0, rawData, newOffset, textureBytes.Length);
        
        // Update dictionaries
        offsetToString[newOffset] = textureName;
        stringToOffset[textureName] = newOffset;
        
        return newOffset;
    }
}
```

## Texture Naming Conventions
Texture filenames typically follow these patterns:
- Environment textures: `environment/texture_name.blp`
- Terrain textures: `terrain/texture_name.blp`
- Structure textures: `buildings/texture_name.blp`
- Item textures: `items/texture_name.blp`

## Version Information
- Present only in the Alpha version of the WDT format
- Similar in function to the MWID and MWMO chunks in modern WDT files, but specifically for textures
- In later versions, texture references were moved to separate model files (M2, WMO)

## Architectural Significance
The MOTX chunk exemplifies the Alpha WDT format's self-contained approach:

1. **Centralized Texture Library**: All map textures are referenced in a single location
2. **Offset-Based References**: Efficient string storage using offsets instead of duplicating strings
3. **Direct Access**: Textures can be accessed directly without loading external files

This differs from the modern approach where:
- Textures are typically referenced within WMO and M2 model files
- The WDT file mainly contains references to external files rather than texture data
- Texture information is distributed across multiple files based on where they are used 