namespace GillijimProject.WowFiles;

/// <summary>
/// Interface for chunks that preserve raw binary data for 1:1 recompilation
/// </summary>
public interface IChunkData
{
    /// <summary>
    /// Original chunk tag (4-byte identifier)
    /// </summary>
    uint Tag { get; }
    
    /// <summary>
    /// Raw chunk payload data (without 8-byte header)
    /// </summary>
    ReadOnlyMemory<byte> RawData { get; }
    
    /// <summary>
    /// Absolute offset of chunk header in source file
    /// </summary>
    long SourceOffset { get; }
    
    /// <summary>
    /// Serialize chunk back to binary format (header + payload)
    /// </summary>
    byte[] ToBytes();
}

/// <summary>
/// Generic chunk for unknown/unimplemented chunk types
/// </summary>
public sealed class UnknownChunk : IChunkData
{
    public uint Tag { get; }
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    public UnknownChunk(uint tag, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        Tag = tag;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public byte[] ToBytes()
    {
        var result = new byte[8 + RawData.Length];
        BitConverter.GetBytes(Tag).CopyTo(result, 0);
        BitConverter.GetBytes((uint)RawData.Length).CopyTo(result, 4);
        RawData.Span.CopyTo(result.AsSpan(8));
        return result;
    }
}

/// <summary>
/// Helper methods for chunk serialization
/// </summary>
public static class ChunkSerializer
{
    /// <summary>
    /// Create chunk header bytes
    /// </summary>
    public static byte[] CreateHeader(uint tag, uint size)
    {
        var header = new byte[8];
        BitConverter.GetBytes(tag).CopyTo(header, 0);
        BitConverter.GetBytes(size).CopyTo(header, 4);
        return header;
    }
    
    /// <summary>
    /// Read chunk tag as string for debugging
    /// </summary>
    public static string TagToString(uint tag)
    {
        var bytes = BitConverter.GetBytes(tag);
        return System.Text.Encoding.ASCII.GetString(bytes);
    }
}
