using System.Numerics;

namespace WowViewer.Core.Maps;

public sealed record AlphaModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float Scale);

public sealed record AlphaWorldModelPlacement(
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    ushort Flags);

public sealed record AlphaLiquidChunk(
    int ChunkIndex,
    int IndexX,
    int IndexY,
    float MinHeight,
    float MaxHeight,
    byte[]? TileFlags,
    uint McnkFlags);

public sealed record AlphaTileDiagnostics(
    bool HasResidualData,
    bool HasSparseChunks,
    int ResidualDataBytes,
    int ActiveChunkCount,
    bool McshSunOrientationUpperRight,
    int McshDataSize);

public sealed class AlphaTileData
{
    public AlphaTileData(
        string sourcePath,
        float[,] heightmap,
        float[,,]? mcalAlphaPack,
        int[,,] mclyTextureIds,
        bool[,,] mclyLayerMask,
        bool[,] holeMask,
        IReadOnlyList<string> textureNames,
        IReadOnlyList<AlphaModelPlacement> modelPlacements,
        IReadOnlyList<AlphaWorldModelPlacement> worldModelPlacements,
        IReadOnlyList<AlphaLiquidChunk> liquidChunks,
        AlphaTileDiagnostics? diagnostics = null)
    {
        SourcePath = sourcePath;
        Heightmap = heightmap;
        McalAlphaPack = mcalAlphaPack;
        MclyTextureIds = mclyTextureIds;
        MclyLayerMask = mclyLayerMask;
        HoleMask = holeMask;
        TextureNames = textureNames;
        ModelPlacements = modelPlacements;
        WorldModelPlacements = worldModelPlacements;
        LiquidChunks = liquidChunks;
        Diagnostics = diagnostics;
    }

    public string SourcePath { get; }
    public float[,] Heightmap { get; }
    public float[,,]? McalAlphaPack { get; }
    public int[,,] MclyTextureIds { get; }
    public bool[,,] MclyLayerMask { get; }
    public bool[,] HoleMask { get; }
    public IReadOnlyList<string> TextureNames { get; }
    public IReadOnlyList<AlphaModelPlacement> ModelPlacements { get; }
    public IReadOnlyList<AlphaWorldModelPlacement> WorldModelPlacements { get; }
    public IReadOnlyList<AlphaLiquidChunk> LiquidChunks { get; }
    public AlphaTileDiagnostics? Diagnostics { get; }

    public AdtPlacementCatalog ToPlacementCatalog()
    {
        return new AdtPlacementCatalog(
            SourcePath,
            MapFileKind.Adt,
            ModelPlacements.Select(static p => p.ModelPath).Distinct().OrderBy(static p => p).ToList(),
            WorldModelPlacements.Select(static p => p.ModelPath).Distinct().OrderBy(static p => p).ToList(),
            ModelPlacements.Select(static p => new AdtModelPlacement(
                p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation, p.Scale)).ToList(),
            WorldModelPlacements.Select(static p => new AdtWorldModelPlacement(
                p.NameId, p.ModelPath, p.UniqueId, p.Position, p.Rotation,
                p.BoundsMin, p.BoundsMax, p.Flags)).ToList());
    }
}
