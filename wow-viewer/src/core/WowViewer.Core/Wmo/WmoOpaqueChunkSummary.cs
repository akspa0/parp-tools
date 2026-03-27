namespace WowViewer.Core.Wmo;

public sealed class WmoOpaqueChunkSummary
{
    public WmoOpaqueChunkSummary(string sourcePath, uint? version, string chunkId, int payloadSizeBytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(chunkId);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);

        SourcePath = sourcePath;
        Version = version;
        ChunkId = chunkId;
        PayloadSizeBytes = payloadSizeBytes;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public string ChunkId { get; }
    public int PayloadSizeBytes { get; }
}
