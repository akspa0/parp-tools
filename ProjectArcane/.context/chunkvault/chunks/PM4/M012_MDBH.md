# M012: MDBH (Destructible Building Header)

## Type
PM4 Container Chunk

## Source
PM4 Format Documentation

## Description
The MDBH chunk is a container for information about destructible buildings. It contains both embedded chunks (MDBF and MDBI) and additional metadata. This chunk is specific to the PM4 format and is not present in PD4 files. The structure provides information about how buildings can be destroyed in the game world.

## Structure
The MDBH chunk has the following structure:

```csharp
struct MDBH
{
    /*0x00*/ mdbh_entry[] m_destructible_building_header;
}

struct mdbh_entry
{
    /*0x00*/ CHUNK index;      // MDBI chunk
    /*0x??*/ CHUNK filename[3]; // MDBF chunks (3 per entry)
}

// Where CHUNK is a generic chunk structure:
struct CHUNK
{
    /*0x00*/ char[4] magic;   // Four CC chunk identifier
    /*0x04*/ uint32_t size;   // Size of the chunk data
    /*0x08*/ byte[] data;     // Chunk data of 'size' bytes
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| m_destructible_building_header | mdbh_entry[] | Array of entries for destructible buildings |
| index | CHUNK (MDBI) | Embedded MDBI chunk containing index data |
| filename | CHUNK[3] (MDBF) | Three embedded MDBF chunks containing filename data for different states of destruction |

## Dependencies
None directly, but this chunk contains:
- Embedded **MDBI** (Destructible Building Index) chunks
- Embedded **MDBF** (Destructible Building Filename) chunks

## Implementation Notes
- Each entry contains one MDBI chunk and three MDBF chunks
- The MDBI chunk contains indices related to the destructible building
- The three MDBF chunks likely contain filenames for different states of the building (intact, damaged, destroyed)
- Parsing requires handling the embedded chunk structure correctly
- Size calculations need to account for the variable-length embedded chunks
- Each embedded chunk has its own magic identifier, size field, and data section

## C# Implementation Example

```csharp
public class MdbhChunk : IChunk
{
    public const string Signature = "MDBH";
    public List<MdbhEntry> Entries { get; private set; }

    public MdbhChunk()
    {
        Entries = new List<MdbhEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        Entries.Clear();

        // Continue reading entries until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var entry = new MdbhEntry();
            
            // Read the MDBI chunk
            entry.IndexChunk = ReadEmbeddedChunk(reader);
            
            // Verify that it's an MDBI chunk
            if (entry.IndexChunk.Magic != "MDBI")
            {
                throw new InvalidDataException($"Expected MDBI chunk but found {entry.IndexChunk.Magic}");
            }

            // Read the three MDBF chunks
            entry.FilenameChunks = new List<ChunkData>();
            for (int i = 0; i < 3; i++)
            {
                var filenameChunk = ReadEmbeddedChunk(reader);
                
                // Verify that it's an MDBF chunk
                if (filenameChunk.Magic != "MDBF")
                {
                    throw new InvalidDataException($"Expected MDBF chunk but found {filenameChunk.Magic}");
                }
                
                entry.FilenameChunks.Add(filenameChunk);
            }
            
            Entries.Add(entry);
        }
    }

    private ChunkData ReadEmbeddedChunk(BinaryReader reader)
    {
        string magic = new string(reader.ReadChars(4));
        uint size = reader.ReadUInt32();
        byte[] data = reader.ReadBytes((int)size);
        
        return new ChunkData
        {
            Magic = magic,
            Size = size,
            Data = data
        };
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            // Write the MDBI chunk
            WriteEmbeddedChunk(writer, entry.IndexChunk);
            
            // Write the three MDBF chunks
            foreach (var filenameChunk in entry.FilenameChunks)
            {
                WriteEmbeddedChunk(writer, filenameChunk);
            }
        }
    }

    private void WriteEmbeddedChunk(BinaryWriter writer, ChunkData chunk)
    {
        writer.Write(chunk.Magic.ToCharArray());
        writer.Write(chunk.Size);
        writer.Write(chunk.Data);
    }

    // Parse the MDBI chunk data
    public MdbiData ParseMdbi(ChunkData mdbiChunk)
    {
        using (var memoryStream = new MemoryStream(mdbiChunk.Data))
        using (var reader = new BinaryReader(memoryStream))
        {
            return new MdbiData
            {
                Index = reader.ReadUInt32()
                // Add additional fields as documented
            };
        }
    }

    // Parse the MDBF chunk data
    public string ParseMdbf(ChunkData mdbfChunk)
    {
        // MDBF data contains a null-terminated string
        int stringLength = 0;
        while (stringLength < mdbfChunk.Data.Length && mdbfChunk.Data[stringLength] != 0)
        {
            stringLength++;
        }
        
        return System.Text.Encoding.ASCII.GetString(mdbfChunk.Data, 0, stringLength);
    }
}

public class MdbhEntry
{
    public ChunkData IndexChunk { get; set; }        // MDBI chunk
    public List<ChunkData> FilenameChunks { get; set; }  // Three MDBF chunks
}

public class ChunkData
{
    public string Magic { get; set; }    // Four-character code (e.g., "MDBI" or "MDBF")
    public uint Size { get; set; }      // Size of the data
    public byte[] Data { get; set; }    // Raw chunk data
}

public class MdbiData
{
    public uint Index { get; set; }
    // Add additional fields as documented
}
```

## Related Information
- This chunk is specific to the PM4 format and not present in PD4 files
- The structure contains embedded chunks, which is a unique pattern compared to other chunks
- The three MDBF chunks likely represent different states of destruction (intact, partially destroyed, fully destroyed)
- The embedded MDBI chunk may contain indices that reference other data in the file
- This pattern of embedded chunks allows for a hierarchical data structure
- This chunk likely interacts with the game's destruction system to show building damage states 