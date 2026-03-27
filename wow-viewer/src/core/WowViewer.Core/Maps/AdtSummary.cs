namespace WowViewer.Core.Maps;

public sealed class AdtSummary
{
    public AdtSummary(
        string sourcePath,
        MapFileKind kind,
        int terrainChunkCount,
        int textureNameCount,
        int modelNameCount,
        int worldModelNameCount,
        int modelPlacementCount,
        int worldModelPlacementCount,
        bool hasFlightBounds,
        bool hasWater,
        bool hasTextureParams,
        bool hasTextureFlags)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(terrainChunkCount);
        ArgumentOutOfRangeException.ThrowIfNegative(textureNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(modelNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(worldModelNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(modelPlacementCount);
        ArgumentOutOfRangeException.ThrowIfNegative(worldModelPlacementCount);

        SourcePath = sourcePath;
        Kind = kind;
        TerrainChunkCount = terrainChunkCount;
        TextureNameCount = textureNameCount;
        ModelNameCount = modelNameCount;
        WorldModelNameCount = worldModelNameCount;
        ModelPlacementCount = modelPlacementCount;
        WorldModelPlacementCount = worldModelPlacementCount;
        HasFlightBounds = hasFlightBounds;
        HasWater = hasWater;
        HasTextureParams = hasTextureParams;
        HasTextureFlags = hasTextureFlags;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public int TerrainChunkCount { get; }

    public int TextureNameCount { get; }

    public int ModelNameCount { get; }

    public int WorldModelNameCount { get; }

    public int ModelPlacementCount { get; }

    public int WorldModelPlacementCount { get; }

    public bool HasFlightBounds { get; }

    public bool HasWater { get; }

    public bool HasTextureParams { get; }

    public bool HasTextureFlags { get; }
}
