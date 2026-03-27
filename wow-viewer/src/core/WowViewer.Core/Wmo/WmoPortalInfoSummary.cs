namespace WowViewer.Core.Wmo;

public sealed class WmoPortalInfoSummary
{
    public WmoPortalInfoSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int maxStartVertex,
        int maxVertexCount,
        float minPlaneD,
        float maxPlaneD)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxStartVertex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxVertexCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        MaxStartVertex = maxStartVertex;
        MaxVertexCount = maxVertexCount;
        MinPlaneD = minPlaneD;
        MaxPlaneD = maxPlaneD;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int EntryCount { get; }
    public int MaxStartVertex { get; }
    public int MaxVertexCount { get; }
    public float MinPlaneD { get; }
    public float MaxPlaneD { get; }
}
