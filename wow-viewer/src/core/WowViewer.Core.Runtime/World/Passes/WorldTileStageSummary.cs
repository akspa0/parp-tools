using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Passes;

public sealed class WorldTileStageSummary
{
    public WorldTileStageSummary(
        string sourcePath,
        MapFileKind kind,
        int wdlVisibleTileCount,
        int terrainChunkCount,
        int terrainHoleChunkCount,
        int liquidChunkCount,
        int liquidLayerCount,
        int visibleLiquidTileCount,
        bool hasWater)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(wdlVisibleTileCount);
        ArgumentOutOfRangeException.ThrowIfNegative(terrainChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(terrainHoleChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(liquidChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(liquidLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(visibleLiquidTileCount);

        SourcePath = sourcePath;
        Kind = kind;
        WdlVisibleTileCount = wdlVisibleTileCount;
        TerrainChunkCount = terrainChunkCount;
        TerrainHoleChunkCount = terrainHoleChunkCount;
        LiquidChunkCount = liquidChunkCount;
        LiquidLayerCount = liquidLayerCount;
        VisibleLiquidTileCount = visibleLiquidTileCount;
        HasWater = hasWater;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public int WdlVisibleTileCount { get; }

    public int TerrainChunkCount { get; }

    public int TerrainHoleChunkCount { get; }

    public int LiquidChunkCount { get; }

    public int LiquidLayerCount { get; }

    public int VisibleLiquidTileCount { get; }

    public bool HasWater { get; }
}