namespace WowViewer.Core.Wmo;

public sealed class WmoGroupBspFaceRangeSummary
{
    public WmoGroupBspFaceRangeSummary(
        string sourcePath,
        uint? version,
        int nodeCount,
        int faceRefCount,
        int zeroFaceNodeCount,
        int coveredNodeCount,
        int outOfRangeNodeCount,
        int maxFaceEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(nodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(faceRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(zeroFaceNodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredNodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeNodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFaceEnd);

        SourcePath = sourcePath;
        Version = version;
        NodeCount = nodeCount;
        FaceRefCount = faceRefCount;
        ZeroFaceNodeCount = zeroFaceNodeCount;
        CoveredNodeCount = coveredNodeCount;
        OutOfRangeNodeCount = outOfRangeNodeCount;
        MaxFaceEnd = maxFaceEnd;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int NodeCount { get; }

    public int FaceRefCount { get; }

    public int ZeroFaceNodeCount { get; }

    public int CoveredNodeCount { get; }

    public int OutOfRangeNodeCount { get; }

    public int MaxFaceEnd { get; }
}