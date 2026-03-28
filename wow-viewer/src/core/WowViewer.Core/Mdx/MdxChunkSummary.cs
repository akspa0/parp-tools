using WowViewer.Core.Chunks;

namespace WowViewer.Core.Mdx;

public sealed class MdxChunkSummary
{
    public MdxChunkSummary(FourCC id, uint payloadSizeBytes, long headerOffset, long dataOffset, bool isKnownChunk)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(headerOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(dataOffset);

        Id = id;
        PayloadSizeBytes = payloadSizeBytes;
        HeaderOffset = headerOffset;
        DataOffset = dataOffset;
        IsKnownChunk = isKnownChunk;
    }

    public FourCC Id { get; }

    public uint PayloadSizeBytes { get; }

    public long HeaderOffset { get; }

    public long DataOffset { get; }

    public bool IsKnownChunk { get; }
}