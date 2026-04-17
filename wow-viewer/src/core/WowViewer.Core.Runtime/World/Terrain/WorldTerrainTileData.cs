using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainTileData
{
    public WorldTerrainTileData(string sourcePath, MapFileKind kind, IReadOnlyList<WorldTerrainChunkData> chunks, WorldTerrainHeightmapData? heightmap)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(chunks);

        SourcePath = sourcePath;
        Kind = kind;
        Chunks = chunks;
        Heightmap = heightmap;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public IReadOnlyList<WorldTerrainChunkData> Chunks { get; }

    public WorldTerrainHeightmapData? Heightmap { get; }

    public int ChunkCount => Chunks.Count;

    public int ChunksWithHeights => Chunks.Count(static chunk => chunk.HasHeights);

    public int HoleChunkCount => Chunks.Count(static chunk => chunk.HasHoles);

    public int LiquidFlagChunkCount => Chunks.Count(static chunk => chunk.HasLiquidFlags);

    public int VertexColorChunkCount => Chunks.Count(static chunk => chunk.HasVertexColors);

    public int DistinctAreaIdCount => Chunks.Select(static chunk => chunk.AreaId).Distinct().Count();

    public bool HasHeightmap => Heightmap is not null;
}