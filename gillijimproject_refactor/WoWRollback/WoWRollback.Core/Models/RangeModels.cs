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
    string FilePath,
    float WorldX,
    float WorldY,
    float WorldZ
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

public sealed record VersionRangeEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint MinUniqueId,
    uint MaxUniqueId,
    string FilePath,
    IReadOnlyList<string> Assets
);

public sealed record MapVersionSummary(
    string Version,
    string Map,
    int RangeCount,
    uint MinUniqueId,
    uint MaxUniqueId,
    int DistinctAssetCount
);

public sealed record RangeOverlapEntry(
    string VersionA,
    string MapA,
    int TileRowA,
    int TileColA,
    PlacementKind KindA,
    uint MinUniqueIdA,
    uint MaxUniqueIdA,
    string VersionB,
    string MapB,
    int TileRowB,
    int TileColB,
    PlacementKind KindB,
    uint MinUniqueIdB,
    uint MaxUniqueIdB,
    uint OverlapMin,
    uint OverlapMax
);

public sealed record AssetFirstSeenEntry(
    string AssetPath,
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint MinUniqueId,
    uint MaxUniqueId
);

public sealed record AssetFolderSummary(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    string Folder,
    int AssetCount
);

public sealed record AssetFolderTimelineEntry(
    string Version,
    string Folder,
    int Depth,
    int DistinctAssetCount,
    int DistinctMapCount,
    int DistinctTileCount,
    uint MinUniqueId,
    uint MaxUniqueId,
    IReadOnlyList<string> Maps,
    IReadOnlyList<string> Subfolders
);

public sealed record VersionComparisonResult(
    string RootDirectory,
    string ComparisonKey,
    IReadOnlyList<string> Versions,
    IReadOnlyList<VersionRangeEntry> RangeEntries,
    IReadOnlyList<MapVersionSummary> MapSummaries,
    IReadOnlyList<RangeOverlapEntry> Overlaps,
    IReadOnlyList<AssetFirstSeenEntry> AssetFirstSeen,
    IReadOnlyList<AssetFolderSummary> AssetFolderSummaries,
    IReadOnlyList<AssetFolderTimelineEntry> AssetFolderTimeline,
    IReadOnlyList<AssetTimelineEntry> AssetTimeline,
    IReadOnlyList<DesignKitAssetEntry> DesignKitAssets,
    IReadOnlyList<DesignKitRangeEntry> DesignKitRanges,
    IReadOnlyList<DesignKitSummaryEntry> DesignKitSummaries,
    IReadOnlyList<DesignKitTimelineEntry> DesignKitTimeline,
    IReadOnlyList<DesignKitAssetDetailEntry> DesignKitAssetDetails,
    IReadOnlyList<UniqueIdAssetEntry> UniqueIdAssets,
    IReadOnlyList<AssetTimelineDetailedEntry> AssetTimelineDetailed,
    IReadOnlyList<string> Warnings
);

public sealed record ComparisonOutputPaths(
    string ComparisonDirectory,
    string VersionRangesPath,
    string MapSummaryPath,
    string OverlapPath,
    string AssetFirstSeenPath,
    string AssetFolderSummaryPath,
    string AssetFolderTimelinePath,
    string AssetTimelinePath,
    string DesignKitAssetsPath,
    string DesignKitRangesPath,
    string DesignKitSummaryPath,
    string DesignKitTimelinePath,
    string DesignKitAssetDetailsPath,
    string UniqueIdAssetsPath,
    string AssetTimelineDetailedPath
);

public sealed record AssetTimelineEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string AssetPath,
    string Folder,
    string Category,
    string Subcategory
);

public sealed record DesignKitAssetEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string AssetPath,
    string DesignKit,
    string SourceRule
);

public sealed record DesignKitRangeEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint MinUniqueId,
    uint MaxUniqueId,
    string DesignKit,
    int AssetCount,
    int DistinctAssetCount,
    string SourceRule
);

public sealed record DesignKitSummaryEntry(
    string Version,
    string DesignKit,
    int DistinctAssetCount,
    int AssetCount,
    int DistinctMapCount,
    int DistinctTileCount,
    uint MinUniqueId,
    uint MaxUniqueId
);

public sealed record DesignKitTimelineEntry(
    string DesignKit,
    string Version,
    int AssetCount,
    int DistinctMapCount,
    int DistinctTileCount,
    uint MinUniqueId,
    uint MaxUniqueId
);

public sealed record MapEntries(
    IReadOnlyList<VersionRangeEntry> Ranges,
    IReadOnlyList<PlacementAsset> Assets
);

public sealed record DesignKitAssetDetailEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string AssetPath,
    string DesignKit,
    string SourceRule,
    string KitRoot,
    string SubkitPath,
    string SubkitTop,
    int SubkitDepth,
    string FileName,
    string FileStem,
    string Extension,
    int SegmentCount
);

public sealed record UniqueIdAssetEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string AssetPath,
    string DesignKit,
    string SubkitPath,
    string SourceRule,
    uint MatchedRangeMin,
    uint MatchedRangeMax,
    string MatchedRangeFile,
    int MatchedRangeCount,
    float WorldX,
    float WorldY,
    float WorldZ
);

public sealed record AssetTimelineDetailedEntry(
    string Version,
    string Map,
    int TileRow,
    int TileCol,
    PlacementKind Kind,
    uint UniqueId,
    string AssetPath,
    string Folder,
    string Category,
    string Subcategory,
    string DesignKit,
    string SourceRule,
    string KitRoot,
    string SubkitPath,
    string SubkitTop,
    int SubkitDepth,
    string FileName,
    string FileStem,
    string Extension,
    float WorldX,
    float WorldY,
    float WorldZ
);
