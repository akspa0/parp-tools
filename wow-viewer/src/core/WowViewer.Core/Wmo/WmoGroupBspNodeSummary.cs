namespace WowViewer.Core.Wmo;

public sealed class WmoGroupBspNodeSummary
{
    public WmoGroupBspNodeSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int nodeCount,
        int leafNodeCount,
        int branchNodeCount,
        int childReferenceCount,
        int noChildReferenceCount,
        int outOfRangeChildReferenceCount,
        int minFaceCount,
        int maxFaceCount,
        int minFaceStart,
        int maxFaceStart,
        int maxFaceEnd,
        float minPlaneDistance,
        float maxPlaneDistance)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(nodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(leafNodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(branchNodeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(childReferenceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(noChildReferenceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeChildReferenceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minFaceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFaceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minFaceStart);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFaceStart);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFaceEnd);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        NodeCount = nodeCount;
        LeafNodeCount = leafNodeCount;
        BranchNodeCount = branchNodeCount;
        ChildReferenceCount = childReferenceCount;
        NoChildReferenceCount = noChildReferenceCount;
        OutOfRangeChildReferenceCount = outOfRangeChildReferenceCount;
        MinFaceCount = minFaceCount;
        MaxFaceCount = maxFaceCount;
        MinFaceStart = minFaceStart;
        MaxFaceStart = maxFaceStart;
        MaxFaceEnd = maxFaceEnd;
        MinPlaneDistance = minPlaneDistance;
        MaxPlaneDistance = maxPlaneDistance;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int NodeCount { get; }

    public int LeafNodeCount { get; }

    public int BranchNodeCount { get; }

    public int ChildReferenceCount { get; }

    public int NoChildReferenceCount { get; }

    public int OutOfRangeChildReferenceCount { get; }

    public int MinFaceCount { get; }

    public int MaxFaceCount { get; }

    public int MinFaceStart { get; }

    public int MaxFaceStart { get; }

    public int MaxFaceEnd { get; }

    public float MinPlaneDistance { get; }

    public float MaxPlaneDistance { get; }
}