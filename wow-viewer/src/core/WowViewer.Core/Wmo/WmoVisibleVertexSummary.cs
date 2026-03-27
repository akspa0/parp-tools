using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoVisibleVertexSummary
{
    public WmoVisibleVertexSummary(string sourcePath, uint? version, int payloadSizeBytes, int vertexCount, Vector3 boundsMin, Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        VertexCount = vertexCount;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int VertexCount { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
}
