using WowViewer.Core.IO.Files;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.App;

internal static class AlphaEmbeddedAdtReader
{
    public static bool TryReadTile(
        string clientRoot,
        string mapDirectory,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out AlphaEmbeddedAdtTileData? alphaTile)
    {
        alphaTile = null;
        return false;
    }
}

internal sealed class AlphaEmbeddedAdtTileData(
    string sourcePath,
    WorldTerrainTileData terrainTileData,
    AdtPlacementCatalog placementCatalog,
    AlphaEmbeddedLiquidTileData liquidTileData)
{
    public string SourcePath { get; } = sourcePath;
    public WorldTerrainTileData TerrainTileData { get; } = terrainTileData;
    public AdtPlacementCatalog PlacementCatalog { get; } = placementCatalog;
    public AlphaEmbeddedLiquidTileData LiquidTileData { get; } = liquidTileData;
}

internal sealed class AlphaEmbeddedLiquidTileData(IReadOnlyList<AdtLiquidChunk> chunks)
{
    public IReadOnlyList<AdtLiquidChunk> Chunks { get; } = chunks;
    public int ActiveChunkCount { get; } = chunks.Count(static chunk => chunk.Layers.Count > 0);
}

