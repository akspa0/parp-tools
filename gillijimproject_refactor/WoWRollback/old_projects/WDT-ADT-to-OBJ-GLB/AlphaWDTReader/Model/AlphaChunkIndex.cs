namespace AlphaWDTReader.Model;

public sealed class AlphaChunkIndex
{
    public long ChunkFileOffset { get; init; }
    public uint ChunkSize { get; init; }

    // Subchunks (absolute file offsets), 0 if not present
    public long OfsMCVT { get; init; }
    public long OfsMCNR { get; init; }
    public long OfsMCLQ { get; init; }

    // Subchunk sizes (bytes), 0 if not present
    public uint SizeMCVT { get; init; }
    public uint SizeMCNR { get; init; }
    public uint SizeMCLQ { get; init; }
}
