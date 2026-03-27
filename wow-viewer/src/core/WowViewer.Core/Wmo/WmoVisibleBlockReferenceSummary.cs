namespace WowViewer.Core.Wmo;

public sealed class WmoVisibleBlockReferenceSummary
{
    public WmoVisibleBlockReferenceSummary(
        string sourcePath,
        uint? version,
        int blockCount,
        int visibleVertexCount,
        int zeroVertexBlockCount,
        int coveredBlockCount,
        int outOfRangeBlockCount,
        int maxVertexEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(blockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(visibleVertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(zeroVertexBlockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredBlockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeBlockCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVertexEnd);

        SourcePath = sourcePath;
        Version = version;
        BlockCount = blockCount;
        VisibleVertexCount = visibleVertexCount;
        ZeroVertexBlockCount = zeroVertexBlockCount;
        CoveredBlockCount = coveredBlockCount;
        OutOfRangeBlockCount = outOfRangeBlockCount;
        MaxVertexEnd = maxVertexEnd;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int BlockCount { get; }
    public int VisibleVertexCount { get; }
    public int ZeroVertexBlockCount { get; }
    public int CoveredBlockCount { get; }
    public int OutOfRangeBlockCount { get; }
    public int MaxVertexEnd { get; }
}
