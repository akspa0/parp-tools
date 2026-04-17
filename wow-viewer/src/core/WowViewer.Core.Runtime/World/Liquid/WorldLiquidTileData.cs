using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Liquid;

public sealed class WorldLiquidTileData
{
    public WorldLiquidTileData(string sourcePath, MapFileKind kind, IReadOnlyList<WorldLiquidChunkData> chunks)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(chunks);

        SourcePath = sourcePath;
        Kind = kind;
        Chunks = chunks;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public IReadOnlyList<WorldLiquidChunkData> Chunks { get; }

    public int ActiveChunkCount => Chunks.Count;

    public int LayerCount => Chunks.Sum(static chunk => chunk.Layers.Count);

    public int VisibleTileCount => Chunks.Sum(static chunk => chunk.VisibleTileCount);
}