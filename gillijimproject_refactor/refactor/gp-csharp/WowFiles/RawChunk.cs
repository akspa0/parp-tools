namespace GillijimProject.WowFiles;

/// <summary>
/// Raw chunk for preserving unparseable data bytes
/// </summary>
public sealed class RawChunk : IChunkData
{
    public uint Tag { get; }
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    public RawChunk(uint tag, byte[] rawData, long sourceOffset)
    {
        Tag = tag;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public byte[] ToBytes()
    {
        // For raw chunks, just return the raw data as-is
        return RawData.ToArray();
    }
}
