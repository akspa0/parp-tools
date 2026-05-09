namespace WowViewer.Core.Maps;

public sealed class TerrainRawChunkBlob
{
    public string EntryName { get; init; } = string.Empty;

    public string SourceKind { get; init; } = string.Empty;

    public string SourcePath { get; init; } = string.Empty;

    public string Scope { get; init; } = string.Empty;

    public string ChunkId { get; init; } = string.Empty;

    public int? ChunkIndex { get; init; }

    public int? ChunkX { get; init; }

    public int? ChunkY { get; init; }

    public byte[] Data { get; init; } = [];
}