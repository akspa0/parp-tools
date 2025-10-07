namespace WoWRollback.AdtModule;

/// <summary>
/// Result of Alpha to Lich King ADT conversion operation.
/// </summary>
public sealed record AdtConversionResult(
    string AdtOutputDirectory,
    string? TerrainCsvPath,
    string? ShadowCsvPath,
    int TilesProcessed,
    int AreaIdsPatched,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Result of terrain CSV extraction from Lich King ADTs.
/// </summary>
public sealed record TerrainCsvResult(
    string TerrainCsvPath,
    string ShadowCsvPath,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Options for Alpha to Lich King ADT conversion.
/// </summary>
public sealed record ConversionOptions
{
    public string? CommunityListfilePath { get; init; }
    public string? LkListfilePath { get; init; }
    public string? DbdDir { get; init; }
    public string? CrosswalkDir { get; init; }
    public string? LkDbcDir { get; init; }
    
    public bool ConvertToMh2o { get; init; } = true;
    public bool AssetFuzzy { get; init; } = true;
    public bool UseFallbacks { get; init; } = true;
    public bool EnableFixups { get; init; } = true;
    public bool Verbose { get; init; }
    
    public string FallbackTileset { get; init; } = "Tileset\\Generic\\Checkers.blp";
    public string FallbackNonTilesetBlp { get; init; } = "Dungeons\\Textures\\temp\\64.blp";
    public string FallbackWmo { get; init; } = "wmo\\Dungeon\\test\\missingwmo.wmo";
    public string FallbackM2 { get; init; } = "World\\Scale\\HumanMaleScale.mdx";
}
