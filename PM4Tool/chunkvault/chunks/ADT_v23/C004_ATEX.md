# ATEX - Texture Filenames

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The ATEX (Textures) chunk contains filenames for all textures used in the ADT v23 format. It stores an array of chunks, with each chunk containing the filename of a texture that can be referenced by ALYR subchunks within ACNK chunks. This chunk provides a centralized registry of all texture assets needed to render the terrain.

## Structure

```csharp
public struct ATEX
{
    // Array of texture filenames
    public TextureEntry[] entries;
    
    // Individual texture entry structure
    public struct TextureEntry
    {
        public char[] filename;  // Null-terminated filename
    }
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| entries | TextureEntry[] | Array of entries, each containing a texture filename |

## Dependencies

No direct dependencies, but referenced by:
- ALYR (S001) - Texture layer subchunks reference ATEX entries by index

## Implementation Notes

1. The ATEX chunk contains an array of texture filenames, each stored as a null-terminated string.

2. The size of this chunk is variable, depending on the number of unique textures used in the ADT tile and the length of each filename.

3. Each filename typically references a BLP texture file in the World of Warcraft directory structure.

4. Texture indices in ALYR subchunks (textureID field) are 0-based, referencing the position in this array.

5. The number of entries is not explicitly stored; it must be determined by parsing the chunk data until the end is reached.

6. All texture filenames are stored with backslashes as directory separators, following Windows convention.

## Implementation Example

```csharp
public class AtexChunk
{
    // List of texture filenames
    public List<string> TextureFilenames { get; private set; } = new List<string>();
    
    public AtexChunk()
    {
    }
    
    public void Load(BinaryReader reader, long size)
    {
        TextureFilenames.Clear();
        
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        // Read texture entries until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            string filename = ReadNullTerminatedString(reader);
            
            // Skip empty entries at the end of the chunk
            if (string.IsNullOrWhiteSpace(filename) && reader.BaseStream.Position >= endPosition - 1)
                break;
                
            TextureFilenames.Add(filename);
            
            // Ensure we're at the start of the next entry (align to byte boundary)
            long currentPos = reader.BaseStream.Position;
            if (currentPos < endPosition && currentPos % 4 != 0)
            {
                reader.BaseStream.Position += 4 - (currentPos % 4);
            }
        }
    }
    
    public void Save(BinaryWriter writer)
    {
        // Calculate the total size needed
        uint totalSize = 0;
        foreach (string filename in TextureFilenames)
        {
            totalSize += (uint)(filename.Length + 1); // +1 for null terminator
            
            // Padding to 4-byte boundary if needed
            if ((filename.Length + 1) % 4 != 0)
                totalSize += (uint)(4 - ((filename.Length + 1) % 4));
        }
        
        writer.Write("ATEX".ToCharArray());
        writer.Write(totalSize);
        
        // Write each filename
        foreach (string filename in TextureFilenames)
        {
            WriteNullTerminatedString(writer, filename);
            
            // Pad to 4-byte boundary if needed
            long currentPos = writer.BaseStream.Position;
            if (currentPos % 4 != 0)
            {
                byte[] padding = new byte[4 - (currentPos % 4)];
                writer.Write(padding);
            }
        }
    }
    
    // Helper method to read a null-terminated string
    private string ReadNullTerminatedString(BinaryReader reader)
    {
        List<char> chars = new List<char>();
        char c;
        while ((c = reader.ReadChar()) != '\0')
        {
            chars.Add(c);
        }
        return new string(chars.ToArray());
    }
    
    // Helper method to write a null-terminated string
    private void WriteNullTerminatedString(BinaryWriter writer, string str)
    {
        foreach (char c in str)
        {
            writer.Write(c);
        }
        writer.Write('\0');
    }
    
    // Helper method to get a texture filename by index
    public string GetFilename(int index)
    {
        if (index >= 0 && index < TextureFilenames.Count)
            return TextureFilenames[index];
        else
            return string.Empty;
    }
    
    // Helper method to add a new texture and get its index
    public int AddTexture(string filename)
    {
        // Check if texture already exists
        for (int i = 0; i < TextureFilenames.Count; i++)
        {
            if (TextureFilenames[i].Equals(filename, StringComparison.OrdinalIgnoreCase))
                return i;
        }
        
        // Add new texture
        TextureFilenames.Add(filename);
        return TextureFilenames.Count - 1;
    }
}
```

## Usage Context

The ATEX chunk serves as a central registry of all textures used in the ADT tile, playing several important roles in the terrain rendering system:

1. **Texture Asset Management**: By providing a single list of all required textures, the ATEX chunk allows the game client to efficiently load and manage texture assets.

2. **Texture Sharing**: Multiple terrain chunks can reference the same texture entries, reducing memory usage and improving texture caching.

3. **Layer Definition**: Texture layers in ALYR subchunks reference these entries by index, defining which textures are applied to different parts of the terrain.

4. **Material System**: The textures referenced by ATEX typically include diffuse maps, and potentially normal maps or specular maps, forming the basis of the terrain material system.

The v23 format's approach to texture management is similar to v18's MTEX chunk, but with the A-prefix naming convention. However, the relationship between ATEX and ALYR creates a more centralized approach to texture mapping compared to v18, where texture references were more distributed.

While this format was never used in any retail release, it shows how the World of Warcraft developers were considering different approaches to terrain texturing during the Cataclysm beta development period, potentially seeking improvements in performance, memory usage, or visual quality. 