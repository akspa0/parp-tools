# ADOO - Doodad/Object Filenames

## Type
ADT v23 Chunk

## Source
Referenced from `ADT_v23.md`

## Description
The ADOO (Doodad/Object) chunk contains filenames for all M2 models and WMO objects used in the ADT v23 format. Unlike the v18 format which uses separate MMDX and MWMO chunks for M2 and WMO filenames respectively, the v23 format consolidates both model types into a single ADOO chunk. This chunk provides a centralized registry of all model assets needed to render decorative objects in the terrain.

## Structure

```csharp
public struct ADOO
{
    // Array of model filenames
    public ModelEntry[] entries;
    
    // Individual model entry structure
    public struct ModelEntry
    {
        public char[] filename;  // Null-terminated filename
    }
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| entries | ModelEntry[] | Array of entries, each containing a model filename |

## Dependencies

No direct dependencies, but referenced by:
- ACDO (S004) - Doodad/object placement subchunks reference ADOO entries by index

## Implementation Notes

1. The ADOO chunk contains an array of model filenames, each stored as a null-terminated string.

2. The size of this chunk is variable, depending on the number of unique models used in the ADT tile and the length of each filename.

3. Each filename typically references either an M2 model file or a WMO object file in the World of Warcraft directory structure.

4. Model indices in ACDO subchunks (modelID field) are 0-based, referencing the position in this array.

5. The number of entries is not explicitly stored; it must be determined by parsing the chunk data until the end is reached.

6. The v23 format does not distinguish between M2 and WMO filenames in this chunk, unlike v18 which stores them separately.

7. All model filenames are stored with backslashes as directory separators, following Windows convention.

## Implementation Example

```csharp
public class AdooChunk
{
    // List of model filenames
    public List<string> ModelFilenames { get; private set; } = new List<string>();
    
    public AdooChunk()
    {
    }
    
    public void Load(BinaryReader reader, long size)
    {
        ModelFilenames.Clear();
        
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        // Read model entries until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            string filename = ReadNullTerminatedString(reader);
            
            // Skip empty entries at the end of the chunk
            if (string.IsNullOrWhiteSpace(filename) && reader.BaseStream.Position >= endPosition - 1)
                break;
                
            ModelFilenames.Add(filename);
            
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
        foreach (string filename in ModelFilenames)
        {
            totalSize += (uint)(filename.Length + 1); // +1 for null terminator
            
            // Padding to 4-byte boundary if needed
            if ((filename.Length + 1) % 4 != 0)
                totalSize += (uint)(4 - ((filename.Length + 1) % 4));
        }
        
        writer.Write("ADOO".ToCharArray());
        writer.Write(totalSize);
        
        // Write each filename
        foreach (string filename in ModelFilenames)
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
    
    // Helper method to get a model filename by index
    public string GetFilename(int index)
    {
        if (index >= 0 && index < ModelFilenames.Count)
            return ModelFilenames[index];
        else
            return string.Empty;
    }
    
    // Helper method to determine if a filename is an M2 or WMO
    public bool IsM2Model(int index)
    {
        string filename = GetFilename(index);
        return !string.IsNullOrEmpty(filename) && filename.EndsWith(".m2", StringComparison.OrdinalIgnoreCase);
    }
    
    // Helper method to determine if a filename is an M2 or WMO
    public bool IsWmoModel(int index)
    {
        string filename = GetFilename(index);
        return !string.IsNullOrEmpty(filename) && filename.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
    }
    
    // Helper method to add a new model and get its index
    public int AddModel(string filename)
    {
        // Check if model already exists
        for (int i = 0; i < ModelFilenames.Count; i++)
        {
            if (ModelFilenames[i].Equals(filename, StringComparison.OrdinalIgnoreCase))
                return i;
        }
        
        // Add new model
        ModelFilenames.Add(filename);
        return ModelFilenames.Count - 1;
    }
}
```

## Usage Context

The ADOO chunk serves as a central registry of all models used in the ADT tile, playing several important roles in the world rendering system:

1. **Model Asset Management**: By providing a single list of all required models, the ADOO chunk allows the game client to efficiently load and manage model assets.

2. **Model Reuse**: Multiple instances of the same model (placed by ACDO subchunks) can reference the same ADOO entry, reducing memory usage and improving resource management.

3. **Unified Model Types**: By storing both M2 and WMO filenames in the same chunk, v23 simplifies the model loading system and potentially enables more unified handling of different model types.

4. **Visual Enrichment**: The models referenced by ADOO form the basis of the world's visual enrichment, adding details like trees, rocks, buildings, and other objects to the terrain.

The v23 format's unified approach to M2 and WMO filenames represents a significant departure from v18's separate MMDX and MWMO chunks. This approach might have been intended to simplify the model loading pipeline and potentially improve performance by treating all models more consistently.

Though never used in any retail release, this experimental approach provides insight into how Blizzard was considering more unified system designs during the Cataclysm beta development period, possibly seeking to reduce complexity and improve efficiency in their world rendering systems. 