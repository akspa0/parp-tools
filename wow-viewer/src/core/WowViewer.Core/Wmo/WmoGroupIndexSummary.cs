namespace WowViewer.Core.Wmo;

public sealed class WmoGroupIndexSummary
{
    public WmoGroupIndexSummary(
        string sourcePath,
        uint? version,
        string chunkId,
        int payloadSizeBytes,
        int indexCount,
        int triangleCount,
        int distinctIndexCount,
        int minIndex,
        int maxIndex,
        int degenerateTriangleCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(chunkId);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(indexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(triangleCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(degenerateTriangleCount);

        SourcePath = sourcePath;
        Version = version;
        ChunkId = chunkId;
        PayloadSizeBytes = payloadSizeBytes;
        IndexCount = indexCount;
        TriangleCount = triangleCount;
        DistinctIndexCount = distinctIndexCount;
        MinIndex = minIndex;
        MaxIndex = maxIndex;
        DegenerateTriangleCount = degenerateTriangleCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public string ChunkId { get; }

    public int PayloadSizeBytes { get; }

    public int IndexCount { get; }

    public int TriangleCount { get; }

    public int DistinctIndexCount { get; }

    public int MinIndex { get; }

    public int MaxIndex { get; }

    public int DegenerateTriangleCount { get; }
}
