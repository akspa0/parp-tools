using System;
using System.IO;
using System.Text.Json;

public interface IChunk
{
    /// <summary>
    /// Gets the chunk ID in documentation format (little-endian)
    /// </summary>
    string ChunkId { get; }
    
    /// <summary>
    /// Parses the chunk data from a binary reader
    /// </summary>
    void Parse(BinaryReader reader, uint size);
    
    /// <summary>
    /// Writes the chunk data to a binary writer
    /// </summary>
    void Write(BinaryWriter writer);
    
    /// <summary>
    /// Converts the chunk data to a human-readable format
    /// </summary>
    string ToHumanReadable();
}

/// <summary>
/// Base class for all chunk handlers that provides common functionality
/// </summary>
public abstract class ChunkBase : IChunk
{
    public abstract string ChunkId { get; }

    public abstract void Parse(BinaryReader reader, uint size);

    public virtual void Write(BinaryWriter writer)
    {
        // Write chunk ID in big-endian format
        ChunkUtils.WriteChunkId(writer, ChunkId);
        
        // Get chunk data size and content
        using (var ms = new MemoryStream())
        using (var tempWriter = new BinaryWriter(ms))
        {
            WriteContent(tempWriter);
            var data = ms.ToArray();
            
            // Write size and content
            writer.Write((uint)data.Length);
            writer.Write(data);
        }
    }

    /// <summary>
    /// Writes the actual chunk content (excluding ID and size)
    /// </summary>
    protected abstract void WriteContent(BinaryWriter writer);

    public virtual string ToHumanReadable()
    {
        // Default implementation uses JSON serialization
        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        return JsonSerializer.Serialize(this, GetType(), options);
    }
}

/// <summary>
/// Represents a chunk header as found in the file
/// </summary>
public class ChunkHeader
{
    /// <summary>
    /// The chunk ID in documentation format (little-endian)
    /// </summary>
    public string Id { get; }
    
    /// <summary>
    /// The size of the chunk data in bytes (excluding header)
    /// </summary>
    public uint Size { get; }
    
    /// <summary>
    /// The position in the stream where the chunk data begins
    /// </summary>
    public long DataPosition { get; }

    public ChunkHeader(string id, uint size, long dataPosition)
    {
        Id = id;
        Size = size;
        DataPosition = dataPosition;
    }

    /// <summary>
    /// Reads a chunk header from the current position in the stream
    /// </summary>
    public static ChunkHeader Read(BinaryReader reader)
    {
        string id = ChunkUtils.ReadChunkId(reader);
        uint size = reader.ReadUInt32();
        long dataPosition = reader.BaseStream.Position;
        
        return new ChunkHeader(id, size, dataPosition);
    }
} 