namespace WowViewer.Core.Maps;

public sealed class MapFileSummary
{
    public MapFileSummary(string sourcePath, MapFileKind kind, uint? version, IReadOnlyList<MapChunkLocation> chunks)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(chunks);

        SourcePath = sourcePath;
        Kind = kind;
        Version = version;
        Chunks = chunks;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public uint? Version { get; }

    public IReadOnlyList<MapChunkLocation> Chunks { get; }

    public int ChunkCount => Chunks.Count;

    public bool HasChunk(WowViewer.Core.Chunks.FourCC id)
    {
        return Chunks.Any(chunk => chunk.Id == id);
    }

    public int CountChunks(WowViewer.Core.Chunks.FourCC id)
    {
        return Chunks.Count(chunk => chunk.Id == id);
    }
}