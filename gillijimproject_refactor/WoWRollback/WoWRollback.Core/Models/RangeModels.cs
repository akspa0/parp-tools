namespace WoWRollback.Core.Models;

public enum PlacementKind
{
    M2,
    WMO
}

public sealed record PlacementRange(
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    int Count,
    uint MinUniqueId,
    uint MaxUniqueId,
    string FilePath
);

public sealed record PlacementAsset(
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint? UniqueId,
    string AssetPath,
    string FilePath
);

public sealed record AlphaAnalysisResult(IReadOnlyList<PlacementRange> Ranges, IReadOnlyList<PlacementAsset> Assets);

public sealed record RangeCsvResult(string PerMapPath, string TimelinePath, string? TimelineAssetsPath, string? AssetLedgerPath);

public sealed record RangeRule(uint Min, uint Max);

public sealed class RangeConfig
{
    public string Map { get; set; } = string.Empty;
    // keep or drop determines inclusion logic for includeRanges when excludeRanges is empty
    public string Mode { get; set; } = "keep"; // or "drop"
    public List<RangeRule> IncludeRanges { get; set; } = new();
    public List<RangeRule> ExcludeRanges { get; set; } = new();
}

public sealed record PlacementEntry(
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string FilePath
);
