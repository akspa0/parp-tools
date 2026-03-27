namespace WowViewer.Core.Wmo;

public sealed class WmoVisibleBlockSummary
{
    public WmoVisibleBlockSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int blockCount,
        int totalVertexRefs,
        int minVerticesPerBlock,
        int maxVerticesPerBlock,
        int minFirstVertex,
        int maxFirstVertex,
        int maxVertexEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(blockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalVertexRefs);
        ArgumentOutOfRangeException.ThrowIfNegative(minVerticesPerBlock);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVerticesPerBlock);
        ArgumentOutOfRangeException.ThrowIfNegative(minFirstVertex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFirstVertex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVertexEnd);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        BlockCount = blockCount;
        TotalVertexRefs = totalVertexRefs;
        MinVerticesPerBlock = minVerticesPerBlock;
        MaxVerticesPerBlock = maxVerticesPerBlock;
        MinFirstVertex = minFirstVertex;
        MaxFirstVertex = maxFirstVertex;
        MaxVertexEnd = maxVertexEnd;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int BlockCount { get; }
    public int TotalVertexRefs { get; }
    public int MinVerticesPerBlock { get; }
    public int MaxVerticesPerBlock { get; }
    public int MinFirstVertex { get; }
    public int MaxFirstVertex { get; }
    public int MaxVertexEnd { get; }
}
