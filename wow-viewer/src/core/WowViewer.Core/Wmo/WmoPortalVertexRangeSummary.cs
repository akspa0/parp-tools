namespace WowViewer.Core.Wmo;

public sealed class WmoPortalVertexRangeSummary
{
    public WmoPortalVertexRangeSummary(
        string sourcePath,
        uint? version,
        int entryCount,
        int vertexCount,
        int zeroVertexPortalCount,
        int coveredPortalCount,
        int outOfRangePortalCount,
        int maxVertexEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(zeroVertexPortalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredPortalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangePortalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVertexEnd);

        SourcePath = sourcePath;
        Version = version;
        EntryCount = entryCount;
        VertexCount = vertexCount;
        ZeroVertexPortalCount = zeroVertexPortalCount;
        CoveredPortalCount = coveredPortalCount;
        OutOfRangePortalCount = outOfRangePortalCount;
        MaxVertexEnd = maxVertexEnd;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int EntryCount { get; }
    public int VertexCount { get; }
    public int ZeroVertexPortalCount { get; }
    public int CoveredPortalCount { get; }
    public int OutOfRangePortalCount { get; }
    public int MaxVertexEnd { get; }
}
