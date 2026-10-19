using System.Numerics;

namespace WowViewer.Core.Archaeology;

public enum PlacementOriginKind
{
    M2 = 0,
    Wmo = 1,
    Creature = 2,
    GameObject = 3
}

public sealed record UniqueIdRecord
{
    public required int UniqueId { get; init; }
    public required string ModelPath { get; init; }
    public string? DisplayName { get; init; }
    public required Vector3 Position { get; init; }
    public Vector3 Rotation { get; init; }
    public float Scale { get; init; } = 1.0f;
    public PlacementOriginKind Kind { get; init; }
    public int MapId { get; init; }
    public int TileX { get; init; }
    public int TileY { get; init; }
    public bool HasTileCoordinates { get; init; }
}

public sealed record UniqueIdLayerSummary(
    int LayerNumber,
    int StartId,
    int EndId,
    int TotalCount,
    int M2Count,
    int WmoCount,
    int CreatureCount,
    int GameObjectCount,
    string? EraLabel = null);

public sealed record TileStratigraphySummary(
    int TileX,
    int TileY,
    int MinUniqueId,
    int MaxUniqueId,
    int MedianUniqueId,
    int TotalCount,
    int M2Count,
    int WmoCount,
    int SpawnCount);

public sealed record MapStratigraphyProfile
{
    public required int MapId { get; init; }
    public required string MapName { get; init; }
    public string? BuildVersion { get; init; }
    public int TotalPlacements { get; init; }
    public int DistinctUniqueIds { get; init; }
    public int MinUniqueId { get; init; }
    public int MaxUniqueId { get; init; }
    public double P10 { get; init; }
    public double P25 { get; init; }
    public double Median { get; init; }
    public double P75 { get; init; }
    public double P90 { get; init; }
    public IReadOnlyList<UniqueIdLayerSummary> Layers { get; init; } = [];
    public IReadOnlyList<TileStratigraphySummary> Tiles { get; init; } = [];
}

public enum DivergenceStatus
{
    InheritedFromOverworld = 0,
    PostForkAuthored = 1,
    Ambiguous = 2
}

public sealed record CrossMapPlacementMatch
{
    public required UniqueIdRecord TargetPlacement { get; init; }
    public UniqueIdRecord? ReferencePlacement { get; init; }
    public DivergenceStatus Status { get; init; }
    public float Distance { get; init; }
    public bool UniqueIdMatched { get; init; }
    public bool SpatialMatched { get; init; }
}

public sealed record MapDivergenceReport
{
    public required int TargetMapId { get; init; }
    public required string TargetMapName { get; init; }
    public required int ReferenceMapId { get; init; }
    public required string ReferenceMapName { get; init; }
    public int TargetTotalPlacements { get; init; }
    public int MatchedPlacementsCount { get; init; }
    public double SharedPercentage { get; init; }
    public int DivergenceCutoffUniqueId { get; init; }
    public IReadOnlyList<CrossMapPlacementMatch> Matches { get; init; } = [];
    public IReadOnlyList<string> Notes { get; init; } = [];
}
