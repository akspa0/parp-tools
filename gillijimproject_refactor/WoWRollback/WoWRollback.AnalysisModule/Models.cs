namespace WoWRollback.AnalysisModule;

/// <summary>
/// Main result from analysis orchestrator.
/// </summary>
public sealed record AnalysisResult(
    IReadOnlyList<string> UniqueIdCsvs,
    IReadOnlyList<string> TerrainCsvs,
    IReadOnlyList<string> PlacementCsvs,
    int OverlayCount,
    string? ManifestPath,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result from UniqueID analysis.
/// </summary>
public sealed record UniqueIdAnalysisResult(
    string CsvPath,
    string LayersJsonPath,
    int TileCount,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result from terrain CSV generation.
/// </summary>
public sealed record TerrainCsvResult(
    string TerrainCsvPath,
    string PropertiesCsvPath,
    int ChunkCount,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result from overlay generation.
/// </summary>
public sealed record OverlayGenerationResult(
    int TilesProcessed,
    int TerrainOverlays,
    int ObjectOverlays,
    int ShadowOverlays,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result from manifest building.
/// </summary>
public sealed record ManifestBuildResult(
    string ManifestPath,
    int OverlayCount,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Options for analysis operations.
/// </summary>
public sealed record AnalysisOptions
{
    public bool GenerateUniqueIdCsvs { get; init; } = true;
    public bool GenerateTerrainCsvs { get; init; } = true;
    public bool GenerateOverlays { get; init; } = true;
    public bool GenerateManifest { get; init; } = true;
    public int UniqueIdGapThreshold { get; init; } = 100; // IDs gap for layer detection
    public bool Verbose { get; init; } = false;
}

/// <summary>
/// Tile-level UniqueID distribution.
/// </summary>
public sealed record TileIdDistribution
{
    public required string MapName { get; init; }
    public required int TileX { get; init; }
    public required int TileY { get; init; }
    public IdDistribution? M2Distribution { get; init; }
    public IdDistribution? WmoDistribution { get; init; }
}

/// <summary>
/// UniqueID distribution for a single asset type.
/// </summary>
public sealed record IdDistribution
{
    public required uint MinId { get; init; }
    public required uint MaxId { get; init; }
    public required int Count { get; init; }
    public required List<LayerInfo> Layers { get; init; }
}

/// <summary>
/// Detected work layer information.
/// </summary>
public sealed record LayerInfo
{
    public required int LayerNumber { get; init; }
    public required uint IdRangeStart { get; init; }
    public required uint IdRangeEnd { get; init; }
    public required int ObjectCount { get; init; }
}

/// <summary>
/// Global layer information across all tiles.
/// </summary>
public sealed record GlobalLayerInfo
{
    public required string MapName { get; init; }
    public required int AnalyzedTiles { get; init; }
    public required List<LayerInfo> GlobalLayers { get; init; }
}

/// <summary>
/// MCNK terrain record for CSV export.
/// </summary>
public sealed record McnkTerrainRecord
{
    public required string MapName { get; init; }
    public required int TileX { get; init; }
    public required int TileY { get; init; }
    public required int ChunkX { get; init; }
    public required int ChunkY { get; init; }
    public required int AreaId { get; init; }
    public required uint Flags { get; init; }
    public required int TextureLayers { get; init; }
    public required bool HasLiquids { get; init; }
    public required bool HasHoles { get; init; }
    public required bool IsImpassible { get; init; }
}
